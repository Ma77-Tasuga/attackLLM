import json
import os
import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
import evaluate
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from accelerate.utils import set_seed

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], return_tensors="pt", padding="max_length", truncation=True,
                       max_length=512)  #50:1100 30:600 20:400
    targets = tokenizer(examples["response"], return_tensors="pt", padding="max_length", truncation=True,
                        max_length=10)

    outputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "target_ids": targets["input_ids"],
        "target_attention_mask": targets["attention_mask"]
    }

    return outputs


# def compute_metrics(eval_pred):
#     print('-----------------------\n')
#     print(eval_pred)
#     logits, labels = eval_pred
#     print(logits)
#     print(labels)
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    set_seed(42)
    split_ratio = 0.2
    smaller_retio = 1  # 1 use full dataset

    datasets_list = ["cadets", 'fivedirections', 'theia', 'trace']
    filename_prefix_attack = '_attack.json'
    filename_prefix_benign = '_benign.json'
    # chosed_dataset = 'cadets'
    folder_train = './DARPA_T3/dataset_json/train'

    # assert chosed_dataset in datasets_list, 'unexpected dataset chosed \n'

    data = []

    for datasets_name in datasets_list:
        file_path_attack = os.path.join(folder_train, datasets_name+filename_prefix_attack)
        file_path_benign = os.path.join(folder_train, datasets_name+filename_prefix_benign)

        with open(file_path_attack, 'r', encoding='utf-8') as fa:
            data_attack = json.load(fa)
            # print(len(data_attack))

        with open(file_path_benign, 'r', encoding='utf-8') as fb:
            data_benign = json.load(fb)
            # print(len(data_benign))

        if len(data_attack) <= len(data_benign):
            data_benign_sampled = random.sample(data_benign, len(data_attack))
        else:
            print('data_attack > data_benign \n')
            data_benign_sampled = data_benign

        data += data_attack
        data += data_benign_sampled

    # print(len(data))

    dataset = Dataset.from_list(data)
    dataset = dataset.shuffle(seed=42)
    # print(dataset[50:100])
    data_size = int(dataset.shape[0] * smaller_retio)
    test_size = int(data_size * split_ratio)

    # test_dataset = dataset.select(indices=range(test_size))
    # train_dataset = dataset.select(indices=range(test_size, data_size))

    """
        训练模型
    """
    # model_name_or_path = './TinyLlama-1.1B-intermediate-step-1431k-3T'
    model_name_or_path = './T5-small'
    # evalmetric_name_or_path = './eval_accuracy/accuracy.py'

    # 加载模型
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    training_args = TrainingArguments(output_dir="check_point",
                                      evaluation_strategy="epoch",
                                      num_train_epochs=50,
                                      per_device_train_batch_size=64,
                                      # per_device_eval_batch_size=1,
                                      learning_rate=1e-03,
                                      )
    # training_args = TrainingArguments(output_dir="check_point",
    #                                   evaluation_strategy="epoch",
    #                                   num_train_epochs=1,
    #                                   )

    print(training_args)
    # 加载评估
    # metric = evaluate.load(evalmetric_name_or_path)
    # print(model.config)

    if getattr(tokenizer, "pad_token_id") is None:
        print("\nlet pad_token = eos_token \n")
        tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(['prompt', 'response'])
    tokenized_datasets = tokenized_datasets.rename_column('target_ids', 'labels')
    tokenized_datasets = tokenized_datasets.rename_column('target_attention_mask', 'decoder_attention_mask')

    # print(tokenized_datasets.shape)
    # print(tokenized_datasets.data[50:60])
    # print(tokenized_datasets['labels'][50:60])
    # print(tokenized_datasets['attention_mask'][50:60])
    # print(tokenized_datasets['decoder_attention_mask'][50:60])

    small_train_dataset = tokenized_datasets.select(range(test_size, data_size))
    small_eval_dataset = tokenized_datasets.select(range(test_size))
    # print(small_train_dataset[50:100])
    print(small_train_dataset.shape)
    print(small_eval_dataset.shape)

    # 训练器配置
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    """ define trainer """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained("output_lora_model")