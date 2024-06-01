from tokenizers.normalizers import BertNormalizer

my_input = "HêllÓ WõrlD"

bert_normalizer = BertNormalizer()

result = bert_normalizer.normalize_str(my_input)

print(f" \n >> Resultado: {result} \n")
