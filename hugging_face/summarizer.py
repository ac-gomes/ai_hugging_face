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

summarizer = pipeline(
    task="summarization",
    model=model,
    min_length=20,
    max_length=40,
    truncation=True,

)

xsum_sample = xsum_dataset["train"].select(range(5))

output = summarizer(xsum_sample["document"])

display(
    pd.DataFrame.from_dict(output)
    .rename({"summary_text": "generated_summary"}, axis=1)
    .join(pd.DataFrame.from_dict(xsum_sample))[
        ["generated_summary", "summary", "document"]
    ]
)
