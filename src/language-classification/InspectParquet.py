import pandas
import pyarrow
import pyarrow.parquet as pq
import fasttext
import os

if __name__ =="__main__":
    # Load the model, it makes sense to use the same one for each
    URL = input("folder to load from:")

    files = os.listdir(URL)
    files = sorted(files)

    for num, file in enumerate(files):
        print(f"--- {file} ---")
        table = pandas.read_parquet(f"{URL}/{file}", engine="pyarrow")
        print(f"Columns = {table.columns}")
        print(table.head())
        print(table)

        print("\n\n")

        sample = table.sample(50, random_state=42)

        for _, row in sample.iterrows():
            print(row["tweet"])
