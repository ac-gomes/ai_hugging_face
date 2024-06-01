from utils import edecoded_sequence

from transformers import BertTokenizer

model = "bert-base-uncased"
input_ids = [101, 2951, 7126, 2000, 3305, 2151, 3291, 1010, 2004, 2146, 2004, 2009, 2003, 2092, 8971, 1998, 10539, 102]

tokenizer = BertTokenizer.from_pretrained(model)

output = tokenizer.decode(input_ids, skip_special_tokens=True)

edecoded_sequence(output)
