import os
import pandas
import pyarrow
import pyarrow.parquet as pq
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

#MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_NAME  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATA_FILE   = "/home/a130b319/a130b319/datasets/subsample_dataset.parquet"
OUTPUT_FILE = "/home/a130b319/a130b319/out.parquet"

def clean_tweets(series):
    return (
        series
        .fillna("").astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"http\S+",  "", regex=True)
        .str.replace(r"@\S+",     "", regex=True)
        .tolist()
    )

if __name__ == "__main__":
    device     = 0 if torch.cuda.is_available() else -1
    batch_size = 1024 if device == 0 else 32
    logging.info(f"Device: {'GPU' if device == 0 else 'CPU'}  |  Batch size: {batch_size}")

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

    table  = pandas.read_parquet(DATA_FILE, engine="pyarrow")
    tweets = clean_tweets(table["tweet"])
    logging.info(f"Loaded {len(tweets):,} tweets")

    results = []
    checkpoint_interval = 100000
    next_checkpoint = checkpoint_interval
 
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i : i + batch_size]
        results.extend(classifier(batch, truncation=True, max_length=512))
        if i % 10240 == 0:
            logging.info(f"  {i:,} / {len(tweets):,}")

        # write checkpoint
        if len(results) >= next_checkpoint:
            checkpoint_df = pandas.concat(
                [table.iloc[:len(results)].reset_index(drop=True), pandas.DataFrame(results)],
                axis=1
            )
            pq.write_table(pyarrow.Table.from_pandas(checkpoint_df), OUTPUT_FILE + ".checkpoint")

            next_checkpoint += checkpoint_interval

    output_df = pandas.concat(
        [table.reset_index(drop=True), pandas.DataFrame(results)],
        axis=1
    )
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pq.write_table(pyarrow.Table.from_pandas(output_df), OUTPUT_FILE)
    logging.info(f"Done. Written to {OUTPUT_FILE}")

