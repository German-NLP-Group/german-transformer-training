from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

save_dir = "de-wiki-talk"
text_corpus_file = '/home/phmay/data/ml-data/gtt/dewiki-talk-20200620-split/xaa'

# hyperparameter
tokenizer_max_len = 512
vocab_size = 52_000  # must be same as in tokenizer-preprocessing
batch_size = 16

# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/configuration_roberta.py#L36
# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/configuration_bert.py#L53
config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,  # what does this mean?
    )

# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/tokenization_roberta.py#L261
# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/tokenization_utils_base.py#L1093
tokenizer = RobertaTokenizerFast.from_pretrained(
    save_dir,
    max_len=tokenizer_max_len,
    )

# make sure tokenizer is saved
tokenizer.save_pretrained(save_dir)

# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/modeling_roberta.py#L178
model = RobertaForMaskedLM(
    config=config,
    )

# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/data/datasets/language_modeling.py#L78
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=text_corpus_file,
    block_size=tokenizer_max_len,  # should this be the same as max_len of RobertaTokenizerFast.from_pretrained
    )

# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/data/data_collator.py#L69
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # this is the default value
    mlm_probability=0.15,  # this is the default value
    )

# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/training_args.py#L33
# tpu_num_cores
# max_steps
# learning_rate
# weight_decay
# adam_epsilon
# max_grad_norm
# warmup_steps
training_args = TrainingArguments(
    output_dir=save_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,  # &&& better set max_steps
    #max_steps=500_000,  # &&& set this and not num_train_epochs
    per_device_train_batch_size=batch_size,  # per_gpu_train_batch_size is depricated
    save_steps=1_00,  # must be changed
    # tpu_num_cores
    save_total_limit=2,  # must be changed
    )

# https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/src/transformers/trainer.py#L134
# what about optimizers parameter ?
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
    )

trainer.train()

trainer.save_model(save_dir)

# save tokenizer again
tokenizer.save_pretrained(save_dir)
