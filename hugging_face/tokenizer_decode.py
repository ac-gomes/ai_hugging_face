from utils import edecoded_sequence

from transformers import BertTokenizer

model = "bert-base-uncased"
input_ids = [101, 5256, 2039, 2302, 2108, 8300, 1010, 5256, 2039, 2000, 2444, 1012, 102]

tokenizer = BertTokenizer.from_pretrained(model)

output = tokenizer.decode(input_ids, skip_special_tokens=True)

edecoded_sequence(output)
