import csv
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

URL = "/home/adam/programming/class/767_information_retrieval/course-project-materials/"
FILE = "monthly_average_sentiments.tsv"

class Mode(Enum):
    base = 1
    elon = 2

mode = Mode.base

text = []
time = []
with open(URL+FILE, newline='') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    header = next(tsv_reader)
    labels = header[1:]

    for row in tsv_reader:
        time.append(row[0])
        text.append([float(x) for x in row[1:]])


text = np.array(text)
fig = plt.figure(facecolor="#efeee7")
ax = fig.add_subplot()

for i in range(text.shape[1]):
    plt.plot(time, text[:, i], label=labels[i])

# Change to location of the legend.
leg = plt.legend(loc="lower right")
plt.draw()
bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
xOffset = .18
bb.x0 += xOffset
bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
plt.subplots_adjust(left=.05)
plt.subplots_adjust(right=.85)


# Add an x axis
#ax.set_aspect('equal')
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

# Show every N label
ax.set_xticks(ax.get_xticks()[::12])

# Background and Show

if mode == Mode.base:
    plt.title("Sentiment Per-Category Over Time")

elif mode == Mode.elon:
    plt.title("Elon Musk's Purchase")
    ax.axvline(x="2022-05", color='k', linestyle='-.')
    ax.axvline(x="2022-11", color='k', linestyle='--')

#plt.title("Elon Musk's Purchase")
plt.show()
