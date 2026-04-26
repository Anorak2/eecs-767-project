# Imports:
#   Pandas is needed as the base library for handling data
#   pyarrow is the engine that reads the parquet file
#   tranformers is a generic interface to work off
import os
import pandas
import pyarrow
import pyarrow.parquet as pq
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACE_API_TOKEN")


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# Useful Constants
URL ="/home/adam/programming/class/767_information_retrieval/course-project-materials/datasets"
OUTPUT_DIR = "/home/adam/programming/class/767_information_retrieval/course-project-materials/output"

def classify_pq_file(filename, output_location, Classifier):
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
        .str.replace(r"@\S+", "", regex=True) # Remove usernames
    )

    batch_size = 1000
    results = []

    batch_n = 0
    file_n = 0
    # with a batch size of 1k this means
    checkpointing = 68
    for i in range(0, len(tweets), batch_size):
        print(f"WORKING: {i} - {i+batch_size}")
        if batch_n == checkpointing:
            batch_n = 0
            file_n += 1
            sent_df = pandas.DataFrame(results)
            # Merge the two lists, the things on table.iloc(... are just for safeguarding
            output_df = pandas.concat(
                [
                    table.iloc[:len(sent_df)].reset_index(drop=True),
                    sent_df
                ],
                axis=1)

            # output to parquet
            output = pyarrow.Table.from_pandas(output_df)
            pq.write_table(output, f"{output_location}/dataset-classified-{file_n}.parquet")
            print("Parquet file written successfully!")
            results = []


        # Extract a set of 1000 tweets
        batch = tweets.iloc[i:i+batch_size].tolist()

        # Classify the tweets
        sentiment = Classifier(batch)

        results.extend(sentiment)
        batch_n += 1


    sent_df = pandas.DataFrame(results)
    # Merge the two lists, the things on table.iloc(... are just for safeguarding
    output_df = pandas.concat(
        [
            table.iloc[:len(sent_df)].reset_index(drop=True),
            sent_df
        ],
        axis=1)

    # output to parquet
    output = pyarrow.Table.from_pandas(output_df)
    pq.write_table(output, f"{output_location}/dataset-classified.parquet")
    print("Parquet file written successfully!")


if __name__ =="__main__":
    # Load the model, it makes sense to use the same one for each
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    files = os.listdir(URL)
    files = sorted(files)
    for num, file in enumerate(files):
        print(f"--- {file} ---")
        classify_pq_file(file, OUTPUT_DIR, classifier)
