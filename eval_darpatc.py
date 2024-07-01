import json
import os
from datasets import Dataset
from peft import PeftModel, PeftConfig
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer ,Text2TextGenerationPipeline, AutoModelForCausalLM
import torch


def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], return_tensors="pt", padding="max_length", truncation=True,
                       max_length=260)  #50:1100 30:600 20:400

    batch_size = len(examples["prompt"])

    # labels = [1 for _ in range(batch_size)]
    labels = [1] * batch_size


    outputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }

    return outputs

def compute_metrics(eval_pred):
    print('-----------------------\n')

    logits, labels = eval_pred
    print(logits)
    print(labels)
    # predictions = np.argmax(logits, axis=-1)
    # metric.compute(predictions=predictions, references=labels)
    return {
        'logits': logits,
        'labels': labels
    }

if __name__ == '__main__':
    set_seed(42)

    model = AutoModelForCausalLM.from_pretrained("./TinyLlama-1.1B-intermediate-step-1431k-3T")
    lora_config = PeftConfig.from_pretrained("./tmp_trainer")

    model = PeftModel.from_pretrained(model,"./tmp_trainer")
    tokenizer = AutoTokenizer.from_pretrained("./TinyLlama-1.1B-intermediate-step-1431k-3T")

    model = model.to("cuda")
    model.eval()

    datasets_list = ["cadets", 'fivedirections', 'theia', 'trace']

    chosed_dataset = 'cadets'
    folder_test = './DARPA_T3/dataset_json/test'

    assert chosed_dataset in datasets_list, 'unexpected dataset chosed \n'

    data_attack = []
    data_benign = []

    for filename in os.listdir(folder_test):
        if chosed_dataset not in filename:
            continue

        with open(os.path.join(folder_test, filename), 'r', encoding='utf-8') as f:
            if 'attack' in filename:
                data_attack = json.load(f)
            elif 'benign' in filename:
                data_benign = json.load(f)
            else:
                print("error: wrong filename \n")
    print("num of data_attack is: " + str(len(data_attack)))
    print("num of data_benign is: " + str(len(data_benign)))
    # print(data_attack[50:51])
    # print(data_benign[50:51])
    prompt = data_attack[53]['prompt']
    # prompt = prompt[:int(len(prompt)/2)]
    # data_size = int(dataset.shape[0] * smaller_retio)
    # test_size = int(data_size * split_ratio)
    #

    if getattr(tokenizer, "pad_token_id") is None:
        print("\nlet pad_token = eos_token \n")
        tokenizer.pad_token = tokenizer.eos_token
    prompt = "question: "+ prompt+' answer: '
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True,
                   max_length=260)
    # print(inputs)
    #
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_length = 300)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=False)[0])

    # trainer = Trainer(
    #     model=model,  # 要微调的模型
    #     # args=training_args,  # 训练参数
    #     # train_dataset=dataset['train'],  # 训练数据集
    #     eval_dataset=tokenized_attacks,  # 验证数据集
    #     compute_metrics=compute_metrics  # 计算评估指标的函数
    # )
    # trainer.evaluate()