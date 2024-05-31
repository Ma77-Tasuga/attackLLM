import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import evaluate
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model


def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], return_tensors="pt", padding="max_length", truncation=True,
                       max_length=400)  #50:1100 30:600 20:400
    targets = tokenizer(examples["response"], return_tensors="pt", padding="max_length", truncation=True,
                        max_length=400)
    outputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "target_ids": targets["input_ids"],
        "target_attention_mask": targets["attention_mask"]
    }

    return outputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

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
    training_args = TrainingArguments(output_dir="test_trainer",
                                      evaluation_strategy="epoch",
                                      num_train_epochs=1,
                                      per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1,
                                      )
    # training_args = TrainingArguments(output_dir="test_trainer",
    #                                   evaluation_strategy="epoch",
    #                                   num_train_epochs=1,
    #                                   )

    print(training_args)
    # 加载评估
    metric = evaluate.load(evalmetric_name_or_path)
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
    # print(tokenized_datasets.data[50:100])
    # print(tokenized_datasets['attention_mask'][50:60])

    small_train_dataset = tokenized_datasets.select(range(test_size, data_size))
    small_eval_dataset = tokenized_datasets.select(range(test_size))
    # print(small_train_dataset[50:100])
    # print(small_train_dataset.shape)
    # print(small_eval_dataset.shape)


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
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained("output_lora_model")
