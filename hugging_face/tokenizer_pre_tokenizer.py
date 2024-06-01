
from utils import pre_tokenizer_output_formatter

from tokenizers.pre_tokenizers import BertPreTokenizer

my_input = ("Data helps to understand any problem, as long as it is well \
             handled and reliable")

bert_pre_tokenizer = BertPreTokenizer()

result = bert_pre_tokenizer.pre_tokenize_str(my_input)

pre_tokenizer_output_formatter(result)
