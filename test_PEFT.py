import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
)
import numpy as np

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, \
    TrainingArguments, Trainer, AutoModelForCausalLM
from tqdm import tqdm

import peft

dataset_name_or_path = "./yelp_review_full/data"
model_name_or_path = './TinyLlama-1.1B-intermediate-step-1431k-3T'
evalmetric_name_or_path = './eval_accuracy/accuracy.py'

# 微调策略
p_type = ("lora")

# 加载数据集
dataset = load_dataset(dataset_name_or_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=3)
print(training_args)
# 加载评估
metric = evaluate.load(evalmetric_name_or_path)
print(model.config)

if getattr(tokenizer, "pad_token_id") is None:
    print("\nlet pad_token = eos_token \n")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    # return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=490)
    return tokenizer(examples["text"],truncation=True, padding="max_length", max_length=120)
# 这里要修改默认max_length，因为基于prompt的方法会添加vtoken，导致embeding长度的不匹配

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)




# 训练器配置
peft_config = None
if p_type == "prefix-tuning":
    peft_type = PeftType.PREFIX_TUNING
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
elif p_type == "prompt-tuning":
    peft_type = PeftType.PROMPT_TUNING
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
elif p_type == "p-tuning":
    peft_type = PeftType.P_TUNING
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
elif p_type == "lora":
    peft_type = PeftType.LORA
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
# print(peft_type)


""" update model """
if peft_config is not None:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
else:
    print("error")


""" define trainer """
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()


