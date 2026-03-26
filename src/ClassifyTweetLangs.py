# Imports:
#   Pandas is needed as the base library for handling data
#   pyarrow is the engine that reads the parquet file
#   fasttext is used as the language classifier
import pandas
import pyarrow
import pyarrow.parquet as pq
import fasttext

# Useful Constants
schema = ["user", "id", "tweet", "replies", "likes", "quotes", "date"]
URL = "/home/adam/.cache/huggingface/hub/datasets--enryu43--twitter100m_tweets/snapshots/5d742bec6f777adf262006017cd3b67e985b0874/data"

def classify_pq_file(filename, model):
    """This function will classify every tweet in a given .parquet file and write it to a new file
    """
    # First load the dataset
    table = pandas.read_parquet(f"{URL}/{filename}", columns=["tweet"], engine="pyarrow")
    print(table)
    tweets = (
        table["tweet"]
        .fillna("")                         # remove NaN
        .astype(str)                        # ensure string
        .str.replace(r"[\r\n]+", " ", regex=True)  # remove newlines
    )

    batch_size = 1000
    results = []

    batch_n = 0
    for i in range(0, len(tweets), batch_size):
        if batch_n == 10:
            print(f"WORKING: {i} - {i+batch_size*batch_n}")
            batch_n = 0
        # Extract a set of 1000 tweets
        batch = tweets.iloc[i:i+batch_size].tolist()

        # Classify the tweets
        labels, _ = model.predict(batch, k=1)

        results.extend([lbl[0] for lbl in labels])
        batch_n += 1

    table["lang"] = results
    output = pyarrow.Table.from_pandas(table)
    pq.write_table(output, "dataset-1.parquet")
    print("Parquet file written successfully!")
    
if __name__ =="__main__":
    # Load the model, it makes sense to use the same one for each
    model_inst = fasttext.load_model("lid.176.bin")

    classify_pq_file("/train-00000-of-00041-3f49db2da17edd5a.parquet", model_inst)
