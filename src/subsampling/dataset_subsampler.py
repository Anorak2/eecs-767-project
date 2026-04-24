import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import os
from zipfile import ZipFile
import pandas as pd

# specifying working directory
zip_folder = "D:/Documents/EECS_767/dataset_zips"
work_dir = "D:/Documents/EECS_767"
dataset_path = "D:/Documents/EECS_767/datasets"

#
# print(os.listdir(zip_folder))
# for zip_name in os.listdir(zip_folder):
#     zip_path = os.path.join(zip_folder, zip_name)
#     with ZipFile(zip_path, 'r') as zip:
#         zip.extractall(work_dir)
#

# unzipping file located in folder
dataset = ds.dataset(dataset_path, format='parquet')
print(dataset.schema)
nrows = sum(p.count_rows() for p in dataset.get_fragments())
print(nrows)

# loading dataset as table, extracting the month and year of each tweet and adding it as a new column for sampling by month-year pair(ex. 05-2021)
dataset_table = dataset.to_table()
dates_table = dataset.to_table(columns=['date'])
print(dates_table.slice())

year_month_pairs = pc.utf8_slice_codeunits(dates_table['date'], 0, 7)
dataset_table = dataset_table.append_column("year_month_pairs", year_month_pairs)

pair_counts = year_month_pairs.to_pandas()
print("type: ", type(pair_counts))
pair_counts = pair_counts.value_counts()

# output file for counts per each month-year pair
# pair_counts.index = pd.to_datetime(pair_counts.index, format="%Y-%m")
# pair_counts = pair_counts.sort_index()
# pair_counts.index = pair_counts.index.strftime("%Y-%m")
# pair_counts.to_csv("month_counts.tsv", header=["count"], sep="\t")

# looking for only month-year pairs that have more than 50000 tweets associated with them, outputting file for manual verification/deciding how many tweets to sample per month
filtered_pairs = pair_counts[pair_counts > 50000]
num_pairs = len(filtered_pairs)
print("list of pairs that have more than 50000 tweets:\n", filtered_pairs)
print("total number of months in the filtered time period: ", num_pairs)
# filtered_pairs.index = pd.to_datetime(filtered_pairs.index, format="%Y-%m")
# filtered_pairs = filtered_pairs.sort_index()
# filtered_pairs.index = filtered_pairs.index.strftime("%Y-%m")
# filtered_pairs.to_csv("filtered_month_counts.tsv", header=["count"], sep="\t")

# sample is set at 50000 per month-year pair
month_sample_size = 50000
subsample_size = num_pairs * month_sample_size
print("Size of Subsample taking 50,000 entries from each year-month pair: ", subsample_size)

# calculating proportion of total tweets from english-filtered dataset(will be used for weighted averages of the subsampled dataset later)
total_tweets = filtered_pairs.sum()
print("total number of tweets after filtering: ", total_tweets)
month_tweet_dataframe = pd.DataFrame({'year_month':filtered_pairs.index, 'tweet_count':filtered_pairs.values})
print(month_tweet_dataframe.head())
month_tweet_dataframe['tweet_count_proportion'] = month_tweet_dataframe['tweet_count'] / total_tweets
print(month_tweet_dataframe.head())
#month_tweet_dataframe.to_csv('month_tweet_statistics.tsv', sep='\t', index=False)

# filtering dataset for only months contained in our list of months that have more than 50000 tweets
filtered_months = filtered_pairs.index.tolist()
dataset_table_filtered = dataset_table.filter(pc.is_in(dataset_table['year_month_pairs'], pa.array(filtered_months)))
df = dataset_table_filtered.to_pandas()
df['year_month_pairs'] = df['year_month_pairs'].astype(str)

# uniform subsample going through each month-year and selecting 50000 tweets per month-year pair
month_subsample_list = []
for month in filtered_months:
    month_entries = df[df['year_month_pairs'] == month]
    month_sample = month_entries.sample(n=month_sample_size, random_state=42)
    month_subsample_list.append(month_sample)

# outputting subsampled dataset as parquet file
subsample_dataset = pd.concat(month_subsample_list, ignore_index=True)
sampled_month_counts = subsample_dataset['year_month_pairs'].value_counts()
print(sampled_month_counts)
subsample_dataset.to_parquet("subsample_dataset.parquet.gzip", compression="gzip")

