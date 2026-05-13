import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd
import os
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import gzip

# input is the topic-labeled sampled dataset
dataset_file = "C:/Users/ninja/PycharmProjects/EECS767_topic_modeling/subsample_dataset_topic_label.parquet.gzip"

dataset = ds.dataset(dataset_file, format="parquet")
print(dataset.schema)
dataset_table = dataset.to_table()

labels = ["sports", "movies/tv shows", "art/design", "video games", "books/literature", "politics", "technology", "science", "business", "lifestyle", "music", "travel", "social/general/other"]

# extracting the tweet, month-year, and confidence score for each of the 13 labels
df = dataset_table.select(['tweet', 'year_month_pairs'] + labels).to_pandas()

# the primary topic label is the one with the highest confidence score
df['top_score'] = df[labels].max(axis=1)
df['label'] = df[labels].idxmax(axis=1)

null_count = df['label'].isna().sum()
print("Null labels:", null_count)

empty_count = (df['label'] == "").sum()
print("Empty labels:", empty_count)

# outputting some summary statistics for error checking
check_df = (df.groupby(['year_month_pairs', 'label']).size().reset_index(name='count').sort_values('year_month_pairs'))
print(check_df)
check_df.to_csv("label_distribution.csv", index=False)

movie_df = (df[df['label'] == 'movies/tv shows'].groupby(['year_month_pairs', 'label']).size().reset_index(name='count').sort_values('year_month_pairs'))
print(movie_df)
movie_df.to_csv("movies_label_distribution.csv", index=False)

# outputting the proportion of each topic label from month-to-month
counts = df.groupby(['year_month_pairs', 'label']).size().reset_index(name='count')
totals = df.groupby('year_month_pairs').size().reset_index(name='total')
prop_df = counts.merge(totals, on='year_month_pairs')
prop_df['proportion'] = prop_df['count'] / prop_df['total']
prop_df = prop_df.sort_values(['year_month_pairs', 'proportion'], ascending=[True, False])
print(prop_df)
prop_df.to_csv("label_proportions_by_month.csv", index=False)

# outputting the total proportions of each topic label over the entire sampled dataset
total_counts = df.groupby('label').size().reset_index(name='count')
total_tweets = len(df)
total_counts['proportion'] = total_counts['count'] / total_tweets
total_counts = total_counts.sort_values('proportion', ascending=False)
print(total_counts)
total_counts.to_csv('overall_label_proportions.csv', index=False)
