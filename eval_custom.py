import json
import os

import numpy as np
from datasets import Dataset
from peft import PeftModel, PeftConfig
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer ,Text2TextGenerationPipeline, TrainingArguments
import torch
from sklearn.metrics import accuracy_score


# def tokenize_function(examples):
#     inputs = tokenizer(examples["prompt"], return_tensors="pt", padding="max_length", truncation=True,
#                        max_length=512)  #50:1100 30:600 20:400
#
#     batch_size = len(examples["prompt"])
#
#     # labels = [1 for _ in range(batch_size)]
#     labels = [1] * batch_size
#
#
#     outputs = {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"],
#         "labels": labels,
#     }
#
#     return outputs

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
def compute_metrics(eval_pred):
    print('-----------------------\n')

    logits, labels = eval_pred
    labels = [1]*len(logits[0])
    # print(logits[0].shape)
    # print(len(logits))
    # print(labels)
    predictions = np.argmax(logits[0], axis=-1)
    pred_labels = []
    for i in range(len(predictions)):
        if 3211 in predictions[i]: # 31144-benign 3211-attack
            pred_labels.append(1)
        else:
            pred_labels.append(0)


    # print(predictions)
    # print(predictions.shape)
    score = accuracy_score(pred_labels,labels)
    print(score)
    return {
        'acc':score,
        'bcc':score-0.5
    }

if __name__ == '__main__':
    set_seed(42)
    smeller_ratio = 0.01

    model = AutoModelForSeq2SeqLM.from_pretrained("./T5-small")
    lora_config = PeftConfig.from_pretrained("./output_lora_model/T5_fulldata_50ep_084_0603")

    model = PeftModel.from_pretrained(model, "./output_lora_model/T5_fulldata_50ep_084_0603")
    tokenizer = AutoTokenizer.from_pretrained("./T5-small")

    training_args = TrainingArguments(
        output_dir="check_point",
        evaluation_strategy="epoch",
        num_train_epochs=50,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=1,
        learning_rate=1e-03,
    )

    datasets_list = ["cadets", 'fivedirections', 'theia', 'trace']
    folder_test = './DARPA_T3/dataset_json/test'

    filename_prefix_attack = '_attack.json'
    filename_prefix_benign = '_benign.json'

    data_attack_all = []
    data_benign_all = []
    for datasets_name in datasets_list:
        file_path_attack = os.path.join(folder_test, datasets_name + filename_prefix_attack)
        file_path_benign = os.path.join(folder_test, datasets_name + filename_prefix_benign)

        with open(file_path_attack, 'r', encoding='utf-8') as fa:
            data_attack = json.load(fa)
            print(len(data_attack))

        with open(file_path_benign, 'r', encoding='utf-8') as fb:
            data_benign = json.load(fb)
            print(len(data_benign))

        data_attack_all += data_attack
        data_benign_all += data_benign

    print("num of data_attack is: " + str(len(data_attack_all)))
    print("num of data_benign is: " + str(len(data_benign_all)))

    dataset_attack = Dataset.from_list(data_attack_all)
    dataset_benign = Dataset.from_list(data_benign_all)

    # print(data_attack[50:51])
    # print(data_benign[50:51])
    # prompt = data_attack[5000]['prompt']
    # prompt = data_benign_all[5000]['prompt']
    # prompt = prompt[:int(len(prompt)/2)]
    # data_size = int(dataset.shape[0] * smaller_retio)
    # test_size = int(data_size * split_ratio)

    if getattr(tokenizer, "pad_token_id") is None:
        print("\nlet pad_token = eos_token \n")
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_attacks = dataset_attack.map(tokenize_function, batched=True)
    # tokenized_attacks = tokenized_attacks.remove_columns(['prompt', 'response'])

    tokenized_attacks = tokenized_attacks.remove_columns(['prompt', 'response'])
    tokenized_attacks = tokenized_attacks.rename_column('target_ids', 'labels')
    tokenized_attacks = tokenized_attacks.rename_column('target_attention_mask', 'decoder_attention_mask')

    # results = generator(dataset_attack["prompt"], max_length=10)
    # print(results)

    # tokenized_attacks = tokenizer(dataset_attack["prompt"], return_tensors="pt", padding="max_length", truncation=True,
    #                    max_length=512)  #50:1100 30:600 20:400)


    # print(tokenized_attacks[50])

    # print(inputs)
    #
    # outputs = model.generate(input_ids=tokenized_attacks.input_ids.to("cuda:0"), attention_mask = tokenized_attacks.attention_mask.to("cuda:0"), max_length=10)
    # print(outputs)
    # print(outputs.shape)
    # print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

    tokenized_attacks = tokenized_attacks.select(range(int(len(tokenized_attacks)*smeller_ratio)))

    trainer = Trainer(
        model=model,  # 要微调的模型
        args=training_args,  # 训练参数
        # train_dataset=dataset['train'],  # 训练数据集
        eval_dataset=tokenized_attacks,  # 验证数据集
        compute_metrics=compute_metrics  # 计算评估指标的函数
    )
    result = trainer.evaluate()
    print(result)
