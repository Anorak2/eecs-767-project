import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.csv as pv
import pandas as pd
import matplotlib.pyplot as plt

dataset_label_file = "C:/Users/ninja/PycharmProjects/EECS767_results/subsample_dataset_topic_label.parquet.gzip"
dataset_sentiment_file = "C:/Users/ninja/PycharmProjects/EECS767_results/sentiment-full.parquet"
labels = ["sports", "movies/tv shows", "art/design", "video games", "books/literature", "politics", "technology", "science", "business", "lifestyle", "music", "travel", "social/general/other"]

label_dataset = ds.dataset(dataset_label_file, format="parquet")
print(label_dataset.schema)
sentiment_dataset = ds.dataset(dataset_sentiment_file, format="parquet")
print(sentiment_dataset.schema)

label_table = label_dataset.to_table()
sentiment_table = sentiment_dataset.to_table()
label_df = label_table.to_pandas()
sentiment_df = sentiment_table.to_pandas()
sentiment_df = sentiment_df.rename(columns={'label': 'sentiment'})
tweet_count_proportions = pd.read_csv("month_tweet_statistics.tsv", sep="\t")
tweet_count_proportions = tweet_count_proportions.rename(
    columns={"year_month": "year_month_pairs"}
)
print(tweet_count_proportions['tweet_count_proportion'].sum())

final_df = label_df.merge(sentiment_df[['id','sentiment']], on='id', how='left')
final_df['topic'] = final_df[labels].idxmax(axis=1)

sentiment_map = {
    "LABEL_2": "positive",
    "LABEL_1": "neutral",
    "LABEL_0": "negative"
}
encode_sentiment = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

final_df['sentiment'] = final_df['sentiment'].map(sentiment_map)
final_df['sentiment_value'] = final_df['sentiment'].map(encode_sentiment)

monthly_overall_sentiment = final_df.groupby('year_month_pairs')['sentiment_value'].mean().reset_index(name='overall_sentiment')
print(monthly_overall_sentiment.head(10).to_string())

monthly_per_topic_sentiment = final_df.groupby(['year_month_pairs', 'topic'])['sentiment_value'].mean().reset_index(name='avg_sentiment')
print(monthly_per_topic_sentiment.head(10).to_string())

overall_weighted_average_sentiment = monthly_overall_sentiment.merge(tweet_count_proportions[['year_month_pairs', 'tweet_count_proportion']], on='year_month_pairs', how='left')
# print(overall_weighted_average_sentiment.head(10).to_string())
overall_weighted_average_sentiment = (overall_weighted_average_sentiment['overall_sentiment'] * overall_weighted_average_sentiment['tweet_count_proportion']).sum()
# print(overall_weighted_average_sentiment)

per_topic_weighted_average_sentiment = monthly_per_topic_sentiment.merge(tweet_count_proportions[['year_month_pairs', 'tweet_count_proportion']], on='year_month_pairs', how='left')
# print(per_topic_weighted_average_sentiment.head(10).to_string())
per_topic_weighted_average_sentiment['weighted_avg_sentiment'] = per_topic_weighted_average_sentiment['avg_sentiment'] * per_topic_weighted_average_sentiment['tweet_count_proportion']
per_topic_weighted_average_sentiment = per_topic_weighted_average_sentiment.groupby('topic')['weighted_avg_sentiment'].sum().reset_index(name='avg_sentiment')
print(per_topic_weighted_average_sentiment.head(10).to_string())

monthly_per_topic_sentiment_wide = monthly_per_topic_sentiment.pivot(index='year_month_pairs', columns='topic', values='avg_sentiment').reset_index()
monthly_per_topic_sentiment_wide = monthly_per_topic_sentiment_wide.merge(monthly_overall_sentiment[['year_month_pairs', 'overall_sentiment']], on='year_month_pairs', how='left')

per_topic_weighted_average_sentiment.loc[len(per_topic_weighted_average_sentiment)] = {"topic":"overall", "avg_sentiment":overall_weighted_average_sentiment}

print(monthly_per_topic_sentiment_wide.head(10).to_string())
print(per_topic_weighted_average_sentiment.to_string())

monthly_per_topic_sentiment_wide.to_csv("monthly_average_sentiments.tsv", sep="\t", index=False)
per_topic_weighted_average_sentiment.to_csv("overall_weighted_average_sentiments.tsv", sep="\t", index=False)