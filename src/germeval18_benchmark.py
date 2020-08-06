from pathlib import Path
import numpy as np

from transformers import AutoTokenizer

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.train import Trainer, EarlyStopping
from farm.utils import initialize_device_settings


def doc_classifcation():
    n_epochs = 1
    batch_size = 32
    evaluate_every = 20
    lang_model = "/home/phmay/data/nlp/checkpoints_256/model-electra"
    #lang_model = "dbmdz/electra-base-german-europeana-cased-generator"
    #lang_model = "bert-base-german-cased"
    use_amp = None
    label_list = ["OTHER", "OFFENSE"]
    metric = "f1_macro"

    device, n_gpu = initialize_device_settings(use_cuda=True, use_amp=use_amp)

    tokenizer = AutoTokenizer.from_pretrained(lang_model, strip_accents=False)
    #tokenizer = Tokenizer.load(
    #    pretrained_model_name_or_path=lang_model,
    #    do_lower_case=False)

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=128,
                                            data_dir=Path("./data/germeval18"),
                                            label_list=label_list,
                                            metric=metric,
                                            label_column_name="coarse_label"
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
    for _ in range(5):
        best_score = doc_classifcation()
        best_score_list.append(best_score)

    score = np.mean(best_score_list)
    print('final score:', score)
