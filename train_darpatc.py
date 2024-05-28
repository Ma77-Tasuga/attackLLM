import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import evaluate
import numpy as np
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
    TaskType
)

split_ratio = 0.2

datasets_list = ["cadets", 'fivedirections', 'theia', 'trace']

chosed_dataset = 'cadets'
folder_train = './DARPA_T3/dataset_json/train'

assert chosed_dataset in datasets_list, 'unexpected dataset chosed \n'

data_attack = []
data_benign = []

for filename in os.listdir(folder_train):
    if chosed_dataset not in filename:
        continue

    with open(os.path.join(folder_train, filename), 'r', encoding='utf-8') as f:
        if 'attack' in filename:
            data_attack = json.load(f)
        elif 'benign' in filename:
            data_benign = json.load(f)
        else:
            print("error: wrong filename \n")


data = data_attack + data_benign
dataset = Dataset.from_list(data)
dataset = dataset.shuffle(seed=42)
# print(dataset[50:100])
data_size = dataset.shape[0]
test_size = int(data_size * split_ratio)

# test_dataset = dataset.select(indices=range(test_size))
# train_dataset = dataset.select(indices=range(test_size, data_size))

"""
    训练模型
"""
model_name_or_path = './TinyLlama-1.1B-intermediate-step-1431k-3T'
evalmetric_name_or_path = './eval_accuracy/accuracy.py'

# 微调策略
p_type = ("lora")


# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=1, per_device_train_batch_size=1)
# print(training_args)
# 加载评估
metric = evaluate.load(evalmetric_name_or_path)
# print(model.config)

if getattr(tokenizer, "pad_token_id") is None:
    print("\nlet pad_token = eos_token \n")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    # return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=490)

    inputs = tokenizer(examples["prompt"], return_tensors="pt", padding="max_length", truncation=True, max_length=300)
    targets = tokenizer(examples["response"], return_tensors="pt", padding="max_length", truncation=True, max_length=300)
    outputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "target_ids": targets["input_ids"],
        "target_attention_mask": targets["attention_mask"]
    }

    return outputs
# 这里要修改默认max_length，因为基于prompt的方法会添加vtoken，导致embeding长度的不匹配

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(['prompt', 'response'])
tokenized_datasets = tokenized_datasets.rename_column('target_ids', 'labels')
tokenized_datasets = tokenized_datasets.rename_column('target_attention_mask', 'decoder_attention_mask')

# print(tokenized_datasets.shape)
# print(tokenized_datasets.data[50:100])


small_train_dataset = tokenized_datasets.select(range(test_size,data_size))
small_eval_dataset = tokenized_datasets.select(range(test_size))
# print(small_train_dataset[50:100])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)




# 训练器配置
peft_config = None
if p_type == "prefix-tuning":
    peft_type = PeftType.PREFIX_TUNING
    peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
elif p_type == "prompt-tuning":
    peft_type = PeftType.PROMPT_TUNING
    peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
elif p_type == "p-tuning":
    peft_type = PeftType.P_TUNING
    peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20, encoder_hidden_size=128)
elif p_type == "lora":
    peft_type = PeftType.LORA
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
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
