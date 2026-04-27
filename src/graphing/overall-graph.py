# overall_weighted_average_sentiments.tsv


import csv
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

URL = "/home/adam/programming/class/767_information_retrieval/course-project-materials/"
FILE = "overall_weighted_average_sentiments.tsv"

counts = []
buckets = []
with open(URL+FILE, newline='') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    header = next(tsv_reader)

    for row in tsv_reader:
        buckets.append(row[0])
        counts.append(float(row[1]))

fig = plt.figure(facecolor="#efeee7")
ax = fig.add_subplot()

counts, buckets = zip(*sorted(zip(counts, buckets)))

colors = ['#d8a7a7', '#8FA3B5', '#C47C6B', '#3E3E3E']

plt.bar(buckets, counts, color=colors)

# Labels
plt.xlabel('Categories')
plt.ylabel('Weighted Sentiment')

# turn labels 45 degrees
plt.xticks(rotation=45)

# add x axis back
ax.axhline(y=0, color='k')

# Spacing
plt.subplots_adjust(top=.95, bottom=.23)

plt.show()
