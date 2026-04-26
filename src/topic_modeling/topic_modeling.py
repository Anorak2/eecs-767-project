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

# debug statements checking if GPU is available for use
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# defining dataset file path, can be changed so path points to the file on a different system
dataset_file = "D:/Documents/EECS_767/subsample_dataset.parquet.gzip"

# loading dataset and creating table of tweets for topic modeling
dataset = ds.dataset(dataset_file, format="parquet")
print(dataset.schema)
nrows = sum(p.count_rows() for p in dataset.get_fragments())
print(nrows)

dataset_table = dataset.to_table()
tweets_table = dataset.to_table(columns=['tweet'])
# test_table_tweets = dataset_table.slice(0, 50000)
# test_table = test_table_tweets.select(['tweet'])

# initialization of GLiClass model and pipeline according to startup directions: https://github.com/Knowledgator/GLiClass
# classification is set to multi-label, meaning confidence scores for multiple labels per tweet will be computed
model = GLiClassModel.from_pretrained("./gliclass_finetuned_EECS767").to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("./gliclass_finetuned_EECS767")
pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device=device)

# defining the dictionary containing each label/topic and its description, extracting both labels and descriptions, and mapping the descriptions to the labels
label_dict = {
    "sports": "sports, athletes, teams, matches, olympics, scores",
    "movies/tv shows": "movies, films, TV shows, anime, actors, trailers, streaming, movie/film reviews",
    "art/design": "art, painting, drawing, illustration, photography, design, galleries, exhibitions",
    "video games": "video games, gameplay, game titles, consoles, video game streaming",
    "books/literature": "books, novels, literature, poetry, authors, literary criticism, book reviews",
    "politics": "politics, government, elections, politicians, policy, voting, legislation",
    "technology": "technology, software, AI, programming, gadgets, internet, hardware, data",
    "science": "science, biology, environmental science, chemistry, physics, astronomy, scientific research, genomics",
    "business": "business, finance, economy, markets, companies, stocks, cryptocurrency, promotions, advertisements, commercial services, customer service",
    "lifestyle": "health, fitness, food, wellness, fashion, DIY, home decor, interior design",
    "music": "music, songs, artists, albums, concerts, music festivals, music videos",
    "travel": "travel, tourism, trips, flights, hotels, destinations, vacations",
    "social/general/other": "casual conversation, replies, jokes, memes, reactions, random thoughts, short messages, emojis, informal chats"
}
label_descriptions = list(label_dict.values())
labels = list(label_dict.keys())
map_desc_to_label = {v: k for k, v in label_dict.items()}

# running the topic modeling in batches of tweets at a time
print("starting topic modeling: \n")
batch_size = 5000
tweets = tweets_table['tweet'].to_pylist()
tweet_label_scores = {label: [] for label in labels}

for i in range(0, len(tweets), batch_size):
    batch = tweets[i:i+batch_size]
    if i < 50000:
        print("topic modeling for tweets index: ", i, "to", i+batch_size)
    with torch.no_grad():
        results = pipeline(batch, label_descriptions)

    # for each tweet, extract confidence scores of all labels for a single tweet and add it as an entry to the array
    for result in results:
        label_scores = {label: 0.0 for label in labels}

        for label_result in result:
            description = label_result['label'].strip()
            label = map_desc_to_label[description]
            label_scores[label] = label_result['score']
        if i == 0 and len(tweet_label_scores[labels[0]]) < 10:
            print(label_scores)

        for label in labels:
            tweet_label_scores[label].append(label_scores[label])

# add each label as a column to the dataset and output as parquet file
dataset_table_topic_labeled = dataset_table
for label in labels:
    dataset_table_topic_labeled = dataset_table_topic_labeled.append_column(label, pa.array(tweet_label_scores[label]))
pq.write_table(dataset_table_topic_labeled, "subsample_dataset_topic_label2.parquet.gzip", compression="gzip")
print("\nfile written")