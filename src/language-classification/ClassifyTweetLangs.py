# Imports:
#   Pandas is needed as the base library for handling data
#   pyarrow is the engine that reads the parquet file
#   fasttext is used as the language classifier
import pandas
import pyarrow
import pyarrow.parquet as pq
import fasttext
import os

# Useful Constants
schema = ["user", "id", "tweet", "replies", "likes", "quotes", "date"]
URL = "/home/adam/.cache/huggingface/hub/datasets--enryu43--twitter100m_tweets/snapshots/5d742bec6f777adf262006017cd3b67e985b0874/data"
#URL = "/home/adam/programming/class/767_information_retrieval/course-project-materials/datasets"
OUTPUT_DIR = "/home/adam/programming/class/767_information_retrieval/course-project-materials/datasets"

def classify_pq_file(filename, output_location, file_num, model):
    """This function will classify every tweet in a given .parquet file and write it to a new file
    """
    # First load the dataset
    table = pandas.read_parquet(f"{URL}/{filename}", engine="pyarrow")
    print(table)
    tweets = (
        table["tweet"]
        .fillna("")                         # remove NaN
        .astype(str)                        # ensure string
        .str.replace(r"[\r\n]+", " ", regex=True)  # remove newlines
        .str.replace(r"http\S+", "", regex=True) # Remove Links
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

    # Filter down to english results
    table["lang"] = results
    filtered_table = table[table["lang"]=="__label__en"]
    filtered_table = filtered_table.drop(columns=["lang"])

    # output to parquet
    output = pyarrow.Table.from_pandas(filtered_table)
    pq.write_table(output, f"{output_location}/dataset-{file_num}.parquet")
    print("Parquet file written successfully!")


if __name__ =="__main__":
    # Load the model, it makes sense to use the same one for each
    model_inst = fasttext.load_model("../../lid.176.bin")

    files = os.listdir(URL)
    files = sorted(files)
    for num, file in enumerate(files):
        if num < 6:
            continue
        print(f"--- {file} ---")
        classify_pq_file(file, OUTPUT_DIR, num, model_inst)
