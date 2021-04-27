#! /usr/bin/env python
# -*- coding: utf-8 -*-
    
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""
Initial input
    dir: path of a working folder
        Ex: C:\...
    data: input file
        Ex: \test.csv
    indicator: input index column name
    figname: output figure name
"""
dir = r""
data = r"\test.csv"
indicator = "test"
figname = r"\test.png"

list = pd.read_csv(dir + data).set_index(indicator)

from matplotlib.colors import LinearSegmentedColormap

cmapset = LinearSegmentedColormap('cset', {
    'green': [(0.0, 1.0, 1.0), (0.5, 0.2, 0.2), (1.0, 0.0, 0.0)],
    'red': [(0.0, 0.0, 0.0), (0.5, 0.2, 0.2), (1.0, 1.0, 1.0)],
    'blue': [(0.0, 0.0, 0.0), (0.5, 0.2, 0.2), (1.0, 0.0, 0.0)],
        })

def draw_heatmap(a, cmap=cmapset):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram

    plt.figure(figsize=(6, 10), dpi=100)
    main_axes = plt.gca()
    divider = make_axes_locatable(main_axes)

    plt.sca(divider.append_axes("left", 1.0, pad=0))
    ylinkage = linkage(pdist(a, metric='euclidean'), method='average', metric='euclidean')
    ydendro = dendrogram(ylinkage, 
                         orientation="left",
                         no_labels=True,
                         distance_sort=False,
                         color_threshold = 6)
    plt.gca().set_axis_off()
    a = a.loc[[a.index[i] for i in ydendro['leaves']]]
    
    plt.sca(main_axes)
    plt.imshow(a, aspect='auto', interpolation='none',
               cmap=cmap, vmin=-1, vmax=1)
    plt.gca().yaxis.tick_right()
    plt.xticks(range(a.shape[1]), a.columns, rotation=90, size='small')
    plt.yticks(range(a.shape[0]), a.index, size=6)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(pad=0.3)
    cbar.ax.tick_params(labelsize=10)

draw_heatmap(list)
plt.savefig(dir + figname, dpi=600)