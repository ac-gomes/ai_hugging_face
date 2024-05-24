import pandas as pd
from IPython.display import display

from transformers import pipeline

# model = "Helsinki-NPL/opus-mt-en-es"

# en_to_es_translator = pipeline(
#     task="translation",
#     model=model,
# )

# output = en_to_es_translator(
#     "Hellow world!"
# )

# print(output)

# Using t5-small model to multiple languages

t5_small_model = pipeline(
    task="text2text-generation",
    model="t5-small",
    max_length=50,

)

output = t5_small_model(
    "translate English to Romanian: Hello world"
)

print(output)
