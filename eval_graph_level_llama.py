import json
import os

import numpy as np
from datasets import Dataset
from peft import PeftModel, PeftConfig
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer ,Text2TextGenerationPipeline, TrainingArguments, AutoModelForCausalLM
import torch
from sklearn.metrics import accuracy_score


def tokenize_function(examples):
    q_list = []
    for p in examples["prompt"]:
        q_list.append("question: "+p)
    tokenizer.pad_token = tokenizer.unk_token
    inputs = tokenizer(q_list, return_tensors="pt", padding="max_length", truncation=True,
                       max_length=940)  #50:1100 30:600 20:400
    # tokenizer.pad_token = tokenizer.eos_token
    # targets = tokenizer(examples["response"], return_tensors="pt", padding="max_length", truncation=True,
    #                     max_length=10)

    return inputs
# def tokenize_function_llama(examples):
#     inputs = tokenizer(examples["prompt"], return_tensors="pt", padding="max_length", truncation=True,
#                        max_length=450)  #50:1100 30:600 20:400
#     targets = tokenizer(examples["response"], return_tensors="pt", padding="max_length", truncation=True,
#                         max_length=10)
#
#     outputs = {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"],
#         "target_ids": targets["input_ids"],
#         "target_attention_mask": targets["attention_mask"]
#     }
#
#     return outputs

def eval_function(model, input_data):
    model = model.to("cuda")
    model.eval()

    input_ids = torch.tensor(input_data['input_ids'])
    attention_mask = torch.tensor(input_data['attention_mask'])
    outputs = model.generate(input_ids=input_ids[:5].to('cuda'),attention_mask=attention_mask[:5].to('cuda'), max_length = 960)
    print(outputs.shape)
    print(outputs[:5])
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=False)[0])
    print("\n")

def compute_metrics(eval_pred):
    pred_labels = []
    true_labels = []
    logits, labels = eval_pred
    batch_size = len(labels)
    print(labels.shape)
    for i in range(batch_size):
        if 3211 in labels[i]:
            true_labels.append(1)
        else:
            true_labels.append(0)

    predictions = np.argmax(logits[0], axis=-1)

    for i in range(batch_size):
        if 3211 in predictions[i]: # 31144-benign 3211-attack
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    acc = (TP + TN) / (TP + TN + FP + FN)

    return {
        'acc':acc,
        "TP": TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

if __name__ == '__main__':
    set_seed(42)
    smaller_ratio = 1
    shards_size = 2000

    base_model_path = "./TinyLlama-1.1B-intermediate-step-1431k-3T"
    finetune_model_path = "./output_lora_model/Llama_0705_50len_10ep_trace_theia"

    # model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    lora_config = PeftConfig.from_pretrained(finetune_model_path)

    model = PeftModel.from_pretrained(model, finetune_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"base model is:{base_model_path}, fine tune model is:{finetune_model_path}.\n")



    training_args = TrainingArguments(
        output_dir="check_point",
        evaluation_strategy="epoch",
        num_train_epochs=50,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=1,
        learning_rate=1e-03,
    )

    datasets_list = ["fivedirections"]
    folder_test = './DARPA_T3/dataset_json/test'
    folder_train = './DARPA_T3/dataset_json/train'

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

    for datasets_name in datasets_list:
        file_path_attack = os.path.join(folder_train, datasets_name + filename_prefix_attack)
        file_path_benign = os.path.join(folder_train, datasets_name + filename_prefix_benign)

        with open(file_path_attack, 'r', encoding='utf-8') as fa:
            data_attack = json.load(fa)
            print(len(data_attack))

        with open(file_path_benign, 'r', encoding='utf-8') as fb:
            data_benign = json.load(fb)
            print(len(data_benign))

        data_attack_all += data_attack
        data_benign_all += data_benign

    # print("num of data_attack is: " + str(len(data_attack_all)))
    # print("num of data_benign is: " + str(len(data_benign_all)))

    # dataset_attack = Dataset.from_list(data_attack_all)
    # dataset_benign = Dataset.from_list(data_benign_all)

    # dataset_attack = dataset_attack.select(range(int(len(dataset_attack)*smaller_ratio)))
    # dataset_benign = dataset_benign.select(range(int(len(dataset_benign)*smaller_ratio)))

    data_all = data_attack_all+data_benign_all
    data_all = data_all[:1000]
    dataset_all = Dataset.from_list(data_all).shuffle(seed=42)
    dataset_all = dataset_all.select(range(int(len(dataset_all)*smaller_ratio)))
    print("dataset shape is: ",dataset_all.shape)

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

    # tokenized_attacks = dataset_attack.map(tokenize_function, batched=True)
    # tokenized_benigns = dataset_benign.map(tokenize_function, batched=True)
    tokenized_dataset = dataset_all.map(tokenize_function, batched=True)

    # tokenized_attacks = tokenized_attacks.remove_columns(['prompt', 'response'])
    # tokenized_attacks = tokenized_attacks.rename_column('target_ids', 'labels')
    # tokenized_attacks = tokenized_attacks.rename_column('target_attention_mask', 'decoder_attention_mask')
    # tokenized_benigns = tokenized_benigns.remove_columns(['prompt', 'response'])
    # tokenized_benigns = tokenized_benigns.rename_column('target_ids', 'labels')
    # tokenized_benigns = tokenized_benigns.rename_column('target_attention_mask', 'decoder_attention_mask')

    tokenized_dataset = tokenized_dataset.remove_columns(['prompt'])
    # tokenized_dataset = tokenized_dataset.rename_column('target_ids', 'labels')
    # tokenized_dataset = tokenized_dataset.rename_column('target_attention_mask', 'decoder_attention_mask')


    # print(inputs['input_ids'][:10])
    # print("\n")
    # print(labels[:10])

    cnt = 0
    print(len(tokenized_dataset['input_ids']))
    num_data = len(tokenized_dataset['input_ids'])
    num_left = int(num_data%shards_size)
    total_loop = int(num_data/shards_size)
    print('total loop:'+str(total_loop))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    now_loop = 0
    for i in range(int(num_data/shards_size)):
        input_data = tokenized_dataset.select(range(shards_size*i,shards_size*(i+1)))
        input_data = input_data.remove_columns(['attention_mask', 'decoder_attention_mask'])

        now_loop += 1
        print(f'loop count: {now_loop} / {total_loop}')
        trainer = Trainer(
            model=model,  # 要微调的模型
            args=training_args,  # 训练参数
            # eval_dataset=tokenized_attacks,  # 验证数据集
            eval_dataset=input_data,
            compute_metrics=compute_metrics  # 计算评估指标的函数
            )

        result = trainer.evaluate()
        TP += int(result['eval_TP'])
        TN += int(result['eval_TN'])
        FP += int(result['eval_FP'])
        FN += int(result['eval_FN'])
        print(result)

    input_data = tokenized_dataset.select(range(num_data-num_left, num_data))

    eval_function(model, input_data)
    # trainer = Trainer(
    #     model=model,  # 要微调的模型
    #     args=training_args,  # 训练参数
    #     # eval_dataset=tokenized_attacks,  # 验证数据集
    #     eval_dataset=input_data,
    #     compute_metrics=compute_metrics,  # 计算评估指标的函数
    #     generation_config="a"
    # )

    # result = trainer.evaluate()
    TP += int(result['eval_TP'])
    TN += int(result['eval_TN'])
    FP += int(result['eval_FP'])
    FN += int(result['eval_FN'])
    print(result)
    print(f'final result: TP={TP}, TN={TN}, FP={FP}, FN={FN}\n')
    acc = (TP + TN) / (TP + TN + FP + FN)
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
    print(f'final matric: acc={acc}, recall={rec}, precision={pre}, f1-score={f1}\n')
    with open('./logbuffer.txt','a') as f:
        f.write(f'final result: TP={TP}, TN={TN}, FP={FP}, FN={FN}\n')
        f.write(f'final matric: acc={acc}, recall={rec}, precision={pre}, f1-score={f1}\n')