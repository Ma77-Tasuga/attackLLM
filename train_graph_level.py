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
def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], return_tensors="pt", padding="max_length", truncation=True,
                       max_length=450)  #50:1100 30:600 20:400
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
    # smaller_ratio = 1  # 1 use full dataset
    eval_smaller_ratio = 0.15
    do_benign_sample = True

    datasets_train = ['trace', 'theia']
    datasets_eval = ['cadets']
    # datasets_eval = ["cadets", 'fivedirections', 'theia', 'trace']
    filename_prefix_attack = '_attack.json'
    filename_prefix_benign = '_benign.json'

    folder_train = './DARPA_T3/dataset_json/train'
    folder_test = './DARPA_T3/dataset_json/test'


    data = []
    data_eval = []
    for datasets_name in datasets_train:
        file_path_attack = os.path.join(folder_train, datasets_name+filename_prefix_attack)
        file_path_benign = os.path.join(folder_train, datasets_name+filename_prefix_benign)

        with open(file_path_attack, 'r', encoding='utf-8') as fa:
            data_attack = json.load(fa)
            # print(len(data_attack))

        with open(file_path_benign, 'r', encoding='utf-8') as fb:
            data_benign = json.load(fb)
            # print(len(data_benign))

        if do_benign_sample:
            print('Sampling benign dataset. \n')
            if len(data_attack) <= len(data_benign):
                data_benign_sampled = random.sample(data_benign, len(data_attack))
            else:
                print('data_attack > data_benign, using all benign data. \n')
                data_benign_sampled = data_benign
        else:
            print('No sampling comfirmed, using full data.\n')
            data_benign_sampled = data_benign
        print(f'num data of <{datasets_name}> in train folder is: {len(data_attack)} and {len(data_benign_sampled)}')
        data += data_attack
        data += data_benign_sampled

    for datasets_name in datasets_train:
        file_path_attack = os.path.join(folder_test, datasets_name+filename_prefix_attack)
        file_path_benign = os.path.join(folder_test, datasets_name+filename_prefix_benign)

        with open(file_path_attack, 'r', encoding='utf-8') as fa:
            data_attack = json.load(fa)
            # print(len(data_attack))

        with open(file_path_benign, 'r', encoding='utf-8') as fb:
            data_benign = json.load(fb)
            # print(len(data_benign))

        if do_benign_sample:
            print('Sampling benign dataset. \n')
            if len(data_attack) <= len(data_benign):
                data_benign_sampled = random.sample(data_benign, len(data_attack))
            else:
                print('data_attack > data_benign, using all benign data. \n')
                data_benign_sampled = data_benign
        else:
            print('No sampling comfirmed, using full data.\n')
            data_benign_sampled = data_benign
        print(f'num data of <{datasets_name}> in test folder is: {len(data_attack)} and {len(data_benign_sampled)}')
        data += data_attack
        data += data_benign_sampled

    for datasets_name in datasets_eval:
        file_path_attack = os.path.join(folder_test, datasets_name+filename_prefix_attack)
        file_path_benign = os.path.join(folder_test, datasets_name+filename_prefix_benign)

        with open(file_path_attack, 'r', encoding='utf-8') as fa:
            data_attack = json.load(fa)
            # print(len(data_attack))

        with open(file_path_benign, 'r', encoding='utf-8') as fb:
            data_benign = json.load(fb)
            # print(len(data_benign))

        if do_benign_sample:
            print('Sampling benign dataset. \n')
            if len(data_attack) <= len(data_benign):
                data_benign_sampled = random.sample(data_benign, len(data_attack))
            else:
                print('data_attack > data_benign, using all benign data. \n')
                data_benign_sampled = data_benign
        else:
            print('No sampling comfirmed, using full data.\n')
            data_benign_sampled = data_benign
        print(f'num data of <{datasets_name}> in test folder is: {len(data_attack)} and {len(data_benign_sampled)}')
        data_eval += data_attack
        data_eval += data_benign_sampled

    dataset = Dataset.from_list(data)
    dataset = dataset.shuffle(seed=42)
    # print(dataset[50:100])
    # data_size = int(dataset.shape[0] * smaller_ratio)
    # dataset = dataset.select(range())

    dataset_eval = Dataset.from_list(data_eval)
    dataset_eval = dataset_eval.shuffle(seed=42)
    data_size_eval = int(dataset_eval.shape[0] * eval_smaller_ratio)
    dataset_eval = dataset_eval.select(range(data_size_eval))
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
    training_args = TrainingArguments(output_dir="check_point",
                                      evaluation_strategy="epoch",
                                      num_train_epochs=10,
                                      per_device_train_batch_size=2,
                                      # per_device_eval_batch_size=1,
                                      learning_rate=1e-04,
                                      )


    print(training_args)
    # 加载评估
    # metric = evaluate.load(evalmetric_name_or_path)
    # print(model.config)

    # if getattr(tokenizer, "pad_token_id") is None:
    #     print("\nlet pad_token = eos_token \n")
    #     tokenizer.pad_token = tokenizer.eos_token

    print("maping train dataset.....\n")
    tokenized_datasets = dataset.map(tokenize_function_llama, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(['prompt', 'response'])
    tokenized_datasets = tokenized_datasets.rename_column('target_ids', 'labels')
    tokenized_datasets = tokenized_datasets.rename_column('target_attention_mask', 'decoder_attention_mask')

    print("mapping eval dataset.....\n")
    tokenized_datasets_eval = dataset_eval.map(tokenize_function_llama, batched=True)
    tokenized_datasets_eval = tokenized_datasets_eval.remove_columns(['prompt', 'response'])
    tokenized_datasets_eval = tokenized_datasets_eval.rename_column('target_ids', 'labels')
    tokenized_datasets_eval = tokenized_datasets_eval.rename_column('target_attention_mask', 'decoder_attention_mask')
    # print(tokenized_datasets.shape)
    # print(tokenized_datasets.data[50:60])
    # print(tokenized_datasets['labels'][50:60])
    # print(tokenized_datasets['attention_mask'][50:60])
    # print(tokenized_datasets['decoder_attention_mask'][50:60])

    # small_train_dataset = tokenized_datasets.select(range(data_size))
    # small_eval_dataset = tokenized_datasets.select(range(test_size))

    # print(small_train_dataset[50:100])
    print(f'train data shape is {tokenized_datasets.shape}')
    print(f'eval data shape is {tokenized_datasets_eval.shape}')

    # 训练器配置
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    # peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    """ define trainer """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets_eval,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained("output_lora_model")