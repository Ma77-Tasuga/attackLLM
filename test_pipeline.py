import transformers
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./bert-base-uncased")
config = transformers.AutoConfig.from_pretrained("./bert-base-uncased")


batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences, padding=True, return_tensors='pt')
print(encoded_inputs)

# print(tokenizer.decode(encoded_input["input_ids"]))