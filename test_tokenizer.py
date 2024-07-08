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
import torch
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


def tokenize_function_llama(examples):

    q_list = []
    a_list = []

    for p,r in zip(examples["prompt"], examples["response"]):
        q_list.append("question: "+ p)
        a_list.append("answer: " + r)

    tokenizer.pad_token = tokenizer.unk_token  # eos 2 bos 1 unk 0
    tokenizer.padding_side = 'left'
    inputs_prompt = tokenizer(q_list, return_tensors="pt", padding="max_length", truncation=True,
                       max_length=940)
    tokenizer.pad_token = tokenizer.eos_token  # eos 2 bos 1 unk 0
    tokenizer.padding_side='right'
    inputs_targets = tokenizer(a_list, return_tensors="pt", padding="max_length", truncation=True,
                        max_length=10, add_special_tokens=False)
    qa_ids = []
    qa_mask = []
    for token_p_ids,token_p_mask,token_t_ids, token_t_mask in zip(inputs_prompt["input_ids"], inputs_prompt["attention_mask"],
                                                                inputs_targets["input_ids"], inputs_targets["attention_mask"]):
        qa_ids.append(torch.cat((token_p_ids,token_t_ids), dim=0))
        qa_mask.append(torch.cat((token_p_mask,token_t_mask), dim=0))

    outputs = {
        "input_ids": qa_ids,
        "attention_mask": qa_mask,
        "target_ids": qa_ids,
        "target_attention_mask": qa_mask
    }

    return outputs


if __name__ == "__main__":
    set_seed(42)
    benign_attack_ratio = 1
    smaller_ratio = 0.1  # 1 use full eval dataset

    datasets_list = ["cadets"]
    filename_prefix_attack = '_attack.json'
    filename_prefix_benign = '_benign.json'
    # chosed_dataset = 'cadets'
    folder_train = './DARPA_T3/dataset_json/train'
    folder_eval = "./DARPA_T3/dataset_json/test"
    # assert chosed_dataset in datasets_list, 'unexpected dataset chosed \n'

    data_train = []
    for datasets_name in datasets_list:
        train_file_path_attack = os.path.join(folder_train, datasets_name+filename_prefix_attack)
        with open(train_file_path_attack, 'r', encoding='utf-8') as fa:
            data_attack = json.load(fa)
            # print(len(data_attack))


        data_train += data_attack



    # print(len(data))

    dataset = Dataset.from_list(data_train[:1000])
    dataset = dataset.shuffle(seed=42)


    # test_dataset = dataset.select(indices=range(test_size))
    # train_dataset = dataset.select(indices=range(test_size, data_size))

    """
        训练模型
    """
    model_name_or_path = './TinyLlama-1.1B-intermediate-step-1431k-3T'
    # model_name_or_path = './T5-small'
    # evalmetric_name_or_path = './eval_accuracy/accuracy.py'

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    training_args = TrainingArguments(output_dir="tmp_trainer",
                                      evaluation_strategy="epoch",
                                      num_train_epochs=10,
                                      per_device_train_batch_size=2,
                                      # per_device_eval_batch_size=1,
                                      learning_rate=1e-03,
                                      )
    # training_args = TrainingArguments(output_dir="check_point",
    #                                   evaluation_strategy="epoch",
    #                                   num_train_epochs=1,
    #                                   )

    # 加载评估
    # metric = evaluate.load(evalmetric_name_or_path)
    # print(model.config)

    # if getattr(tokenizer, "pad_token_id") is None:
    #     print("\nlet pad_token = eos_token \n")
    #     tokenizer.pad_token = tokenizer.eos_token # eos 2 bos 1 unk 0
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    # data = dataset[10]
    # print(data)
    # inputs = tokenizer("i am a dog. you are a cat.", return_tensors="pt",
    #                    padding="max_length",
    #                    truncation=True,
    #                    max_length=400)
    # print('\n')
    # print(len(inputs['attention_mask'][0]))
    # print(inputs)

    # print("maping train dataset.....\n")
    # tokenized_datasets = dataset.map(tokenize_function_llama, batched=True)
    # print(tokenized_datasets["input_ids"][0])
    # print(tokenized_datasets["attention_mask"][0])
    # print(tokenizer.decode(tokenized_datasets['input_ids'][0]))

    #
    # tokenized_datasets = tokenized_datasets.remove_columns(['prompt', 'response'])
    # tokenized_datasets = tokenized_datasets.rename_column('target_ids', 'labels')
    # tokenized_datasets = tokenized_datasets.rename_column('target_attention_mask', 'decoder_attention_mask')

    prompt = "question: "+dataset["prompt"][0] + " answer: "+dataset["response"][0]
    # prompt = 'question '+dataset["prompt"][0]
    # target = "answer: " + dataset["response"][0]
    # prompt = prompt+" "+target
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'
    tokenized_datasets = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True,
                       max_length=950)
    print(tokenized_datasets)
    print(tokenizer.decode(tokenized_datasets['input_ids'][0]))
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'
    # tokenized_target = tokenizer(target, return_tensors="pt", padding="max_length", truncation=True,
    #                     max_length=10, add_special_tokens=False)
    # print(tokenized_target)
    # input = torch.cat((tokenized_datasets['input_ids'],tokenized_target['input_ids']), dim=1)
    # print(input)
    # print(tokenizer.decode(input[0]))

    # print(input)
    # print(tokenized_datasets[10]['input_ids'][-8:])
    # print(tokenized_datasets['attention_mask'][100:105])
    # print(tokenized_datasets.shape)
    # print(tokenized_datasets.data[50:60])
    # print(tokenized_datasets['labels'][50:60])
    # print(tokenized_datasets['attention_mask'][50:60])
    # print(tokenized_datasets['decoder_attention_mask'][50:60])

