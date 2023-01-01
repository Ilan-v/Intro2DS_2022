import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_clusters_2D(data,clusters, title,figsize=(8,5)):
    plt.figure(figsize=figsize)
    num_clusters = len(np.unique(clusters))
    sns.scatterplot(
        x=data[:,0], y=data[:,1],
        hue=clusters,
        palette=sns.color_palette("hls", num_clusters),
        legend="full",
        alpha=0.8
    )
    plt.title(title)
    plt.show()

def plot_clusters_3D(data,clusters, title,figsize=None):
    fig = go.Figure(data=[go.Scatter3d(
    x=data[:,0], y=data[:,1], z=data[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=clusters,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
    )])
    fig.update_layout(title=title)
    if figsize:
        fig.update_layout(
            width=figsize[0],
            height=figsize[1],
        )
    fig.show()