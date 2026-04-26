import pandas
import pyarrow
import pyarrow.parquet as pq
import fasttext
import os

if __name__ =="__main__":
    # Load the model, it makes sense to use the same one for each
    #URL = input("folder to load from:")
    URL = "/home/adam/programming/class/767_information_retrieval/course-project-materials/datasets"

    files = os.listdir(URL)
    files = sorted(files)

    for num, file in enumerate(files):
        print(f"--- {file} ---")
        table = pandas.read_parquet(f"{URL}/{file}", engine="pyarrow")
        print(table["label"].value_counts(normalize=True) * 100)
        print(f"Columns = {table.columns}")
        print(table.head())
        print(table)

        print("\n\n")

        sample = table.groupby("label").apply(lambda x: x.sample(50, random_state=42)).reset_index(drop=True)
        # sample = table.sample(50, random_state=42)

        for _, row in sample.iterrows():
            tweet = row["tweet"]
            sentiment = row["score"]
            label = row["label"]
            print(f"\n-----------------------------\n{tweet}\n----\n{label}: {sentiment}")
