import pandas as pd
from IPython.display import display

from datasets import load_dataset
from transformers import pipeline

poem_dataset = load_dataset(
    "poem_sentiment",
    version="1.0.0",
    trust_remote_code=True
)

poem_sample = poem_dataset["train"].select(range(5))

# display(poem_sample.to_pandas())
model = "nickwong64/bert-base-uncased-poems-sentiment"

sentiment_classifier = pipeline(
    task="text-classification",
    model=model,
)

output = sentiment_classifier(poem_sample["verse_text"])

# join predictions with data
joined_data = (
    pd.DataFrame.from_dict(output)
    .rename({"label": "predicted_label"}, axis=1)
    .join(pd.DataFrame.from_dict(poem_sample).rename({"label": "true_label"}, axis=1))
)

# Change label indices to text labels
sentiment_lables = {0: "negative", 1: "positive", 2: "no_impact", 3: "mixed"}
joined_data = joined_data.replace({"true_label": sentiment_lables})

display(
    joined_data[["predicted_label", "true_label", "score", "verse_text"]]
)

