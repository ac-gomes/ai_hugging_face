import pandas as pd
from IPython.display import display

from datasets import load_dataset
from transformers import pipeline

# Documentatio: https://huggingface.co/docs/hub/datasets-pandas

model = "t5-small"

xsum_dataset = load_dataset(
    "xsum",
    version="1.2.0",
    trust_remote_code=True
)

xsum_sample = xsum_dataset["train"].select(range(10))

# display(
#     xsum_sample.to_pandas()
# )

summarizer = pipeline(
    task="summarization",
    model=model,
    min_length=20,
    max_length=40,
    truncation=True,
    # model_kwargs={cache_di},
)

output = summarizer(
    xsum_sample["document"][0]
)

print(output)
