from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate  # 这也是教程里的一个库
import peft

""" prepare model """
model = AutoModelForSequenceClassification.from_pretrained("./bert-base-uncased", num_labels=5)
dataset = load_dataset("./yelp_review_full/data")
# print(dataset["train"][100])
tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
training_args = TrainingArguments(output_dir="check_point", evaluation_strategy="epoch")
training_args.num_train_epochs=5.0
print(training_args)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


""" parse dataset """
tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

""" define evaluation method """
metric = evaluate.load("./eval_accuracy/accuracy.py")  # 这个东西也会实时下载，看看怎么 本地化


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


""" define trainer """
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
