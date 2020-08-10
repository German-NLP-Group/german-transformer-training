import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.train import Trainer, EarlyStopping
from farm.utils import initialize_device_settings
from farm.modeling.tokenization import Tokenizer

n_epochs = 1
batch_size = 32
evaluate_every = 15
cp_num = 650_000
repeats = 31
do_lower_case = True
lang_model = "/home/phmay/data/nlp/checkpoints_256/model-electra"
#lang_model = "/home/phmay/data/nlp/checkpoints_128/model-electra"
#lang_model = "dbmdz/electra-base-german-europeana-cased-generator"
#lang_model = "bert-base-german-cased"


# https://huggingface.co/dbmdz/bert-base-german-cased
#lang_model = "dbmdz/bert-base-german-cased"

# https://huggingface.co/dbmdz/bert-base-german-uncased
#lang_model = "dbmdz/bert-base-german-uncased"

# https://huggingface.co/distilbert-base-german-cased
#lang_model = "distilbert-base-german-cased"

# https://huggingface.co/dbmdz/bert-base-german-europeana-cased
#lang_model = "dbmdz/bert-base-german-europeana-cased"

# https://huggingface.co/dbmdz/electra-base-german-europeana-cased-discriminator'
#lang_model = "dbmdz/electra-base-german-europeana-cased-discriminator"

# https://huggingface.co/dbmdz/bert-base-german-europeana-uncased
#lang_model = "dbmdz/bert-base-german-europeana-uncased"

use_amp = None
label_list = ["OTHER", "OFFENSE"]
metric = "f1_macro"

def doc_classifcation():
    device, n_gpu = initialize_device_settings(use_cuda=True, use_amp=use_amp)

    tokenizer = AutoTokenizer.from_pretrained(lang_model, strip_accents=False)
    #tokenizer = Tokenizer.load(
    #    pretrained_model_name_or_path=lang_model,
    #    do_lower_case=do_lower_case)

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=128,
                                            data_dir=Path("./data/germeval18"),
                                            label_list=label_list,
                                            metric=metric,
                                            dev_filename="test.tsv",  # we want to evaluate against test
                                            label_column_name="coarse_label",
                                            )

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = LanguageModel.load(lang_model)
    prediction_head = TextClassificationHead(
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
        num_labels=len(label_list))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=3e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        use_amp=use_amp)

    earlystopping = EarlyStopping(
        metric=metric, mode="max",
        #save_dir=Path("./saved_models"),
        patience=3
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        early_stopping=earlystopping,
        device=device)

    trainer.train()

    return earlystopping.best_so_far


if __name__ == "__main__":
    best_score_list = []
    for _ in tqdm(range(repeats)):
        best_score = doc_classifcation()
        best_score_list.append(best_score)

    print('all best scores:', best_score_list)
    score = np.mean(best_score_list)
    std = np.std(best_score_list)
    print('mean F1 macro of {} runs for checkpoint {}: {} with standard deviation: {}'.format(repeats, cp_num, score, std))
    print('on:', lang_model, "with do_lower_case:", do_lower_case)
