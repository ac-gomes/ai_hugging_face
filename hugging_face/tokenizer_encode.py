from utils import output_formatter

from transformers import BertTokenizer

model = "bert-base-uncased"
sequence = "Wake up without being awake, wake up to live."


tokenizer = BertTokenizer.from_pretrained(model)

output = tokenizer(sequence)

output_formatter(output.data)
