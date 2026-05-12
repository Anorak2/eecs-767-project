import pandas as pd
import matplotlib.pyplot as plt

labels = ["sports", "movies/tv shows", "art/design", "video games", "books/literature", "politics", "technology", "science", "business", "lifestyle", "music", "travel", "social/general/other"]

df = pd.read_csv("label_proportions_by_month.csv")
df["year_month_pairs"] = pd.to_datetime(df["year_month_pairs"])
pivot_df = df.pivot(index="year_month_pairs", columns="label", values="proportion")

pivot_df.plot(figsize=(15, 6))
plt.title("Topic Proportions Over Time")
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.legend(loc="upper right", bbox_to_anchor=(1.11, 1))
plt.savefig('tweet_label_proportions.png')
plt.show()

