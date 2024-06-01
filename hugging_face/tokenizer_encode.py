from utils import output_formatter

from transformers import BertTokenizer

model = "bert-base-uncased"

sequence = [
  "Hellow wold",
  "Hello data world",
]

tokenizer = BertTokenizer.from_pretrained(model)

output = tokenizer(sequence, padding=True)

output_formatter(output.data)
