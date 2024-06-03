import json
import os

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch



model = AutoModelForSeq2SeqLM.from_pretrained("./T5-small")
lora_config = PeftConfig.from_pretrained("./output_lora_model/T5_outofmem_cp")

model = PeftModel.from_pretrained(model,"./output_lora_model/T5_outofmem_cp")
tokenizer = AutoTokenizer.from_pretrained("./T5-small")

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
prompt = data_attack[15000]['prompt']
# prompt = prompt[:int(len(prompt)/2)]
# data_size = int(dataset.shape[0] * smaller_retio)
# test_size = int(data_size * split_ratio)
#

if getattr(tokenizer, "pad_token_id") is None:
    print("\nlet pad_token = eos_token \n")
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True,
                   max_length=500)
print(inputs)
#
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"))
print(outputs)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
