import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.csv as pv
import pandas as pd
import os
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import gzip
import torch
import json

dataset_file = "D:/Documents/EECS_767/subsample_dataset.parquet.gzip"
dataset = ds.dataset(dataset_file, format="parquet")
dataset_table = dataset.to_table()
df = dataset_table.to_pandas()
sample_df = df.groupby('year_month_pairs').sample(4, random_state=42)
sample_df['id'] = sample_df['id'].astype(str)
export_df = sample_df[['id', 'tweet']]
export_df.to_csv("subsample_tweets.tsv", sep="\t", index=False)
