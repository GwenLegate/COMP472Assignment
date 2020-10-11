import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants as c
def graph_initial_values():

    latin_key = pd.read_csv(c.LATIN_KEY, sep=',',skiprows=0).to_numpy()
    greek_key = pd.read_csv(c.GREEK_KEY, sep=',',skiprows=0).to_numpy()
    latin = pd.read_csv(c.FILE_TRAIN_LATIN, sep=',',header=None).to_numpy()
    greek = pd.read_csv(c.FILE_TRAIN_GREEK, sep=',',header=None).to_numpy()

    latin_x, latin_y = latin[:, :-1], latin[:, -1]
    greek_x, greek_y = greek[:, :-1], greek[:, -1]

    total = 0
    latin_totals = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0}
    greek_totals = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    for y in latin_y:
        latin_totals[int(y)] += 1
        total += 1

    for y in greek_y:
        greek_totals[int(y)] += 1
        total += 1

    ly = latin_totals.values()
    lx = latin_key[:, -1]
    gy = greek_totals.values()
    gx = greek_key[:, -1]

    fig, axs = plt.subplots(2)
    fig.suptitle('Distribution of Instances in Greek and Latin Alphabet Datasets')
    axs[0].bar(lx, ly, color="orange")
    axs[1].bar(gx, gy, color="green")
    plt.xticks(fontsize=10, rotation=36)
    plt.savefig("InitialGraph.png")
    plt.show()

