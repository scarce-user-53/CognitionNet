import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def generate_plot(data_name, loc):
    data = np.load(loc, allow_pickle=True)
    epochs = [i for i in range(0, len(data))]
    plt.plot(epochs, data, color="blue", marker="o", linestyle="dashed")
    plt.xlabel("epochs")
    plt.ylabel(data_name)
    plt.show()


def plot_tsne(tsne_results, cluster_allocation, file_name):
    df_subset = pd.DataFrame(tsne_results, columns=["tsne-2d-one", "tsne-2d-two"])
    df_subset["cluster"] = cluster_allocation

    plot_clusters = sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="cluster",
        palette=sns.color_palette("hls", 7),
        data=df_subset,
        legend=False,  # "full",
        alpha=0.3,
    )
    plot_clusters.set(xlabel=None)
    plot_clusters.set(ylabel=None)
    loc = "../images/tsne_ep_" + file_name + ".png"
    figure = plot_clusters.get_figure()
    figure.savefig(loc, dpi=100)


if __name__ == "__main__":
    # ls_space_loc = "<path to latent space representation>"
    # cluster_hist_loc = "<path to cluster mapping>"
    # rc_loc = "<path to RC loss history>"
    # cce_loc = "<path to CCE loss history>"
    # fcm_loc = "<path to bridge loss history>"
    # sil_loc = "<path to silhoutte score history>"
    # trace_loc = "<path to trace loss history>"

    ls_tsne = np.load(ls_space_loc, allow_pickle=True)
    history = pd.read_csv(cluster_hist_loc)
    for index in range(0, ls_tsne.shape[0],):
        ls_space = ls_tsne[index, :, :]
        start = index * ls_tsne.shape[1]
        epoch_history = history[start : start + ls_tsne.shape[1]]

        mode_cluster = epoch_history["cluster"].mode()[0]
        mask = epoch_history["cluster"] != mode_cluster

        epoch_history = epoch_history[mask]
        ls_space = ls_space[mask, :]

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(ls_space)
        pca_ls_space = pd.DataFrame()
        pca_ls_space["pca-one"] = pca_result[:, 0]
        pca_ls_space["pca-two"] = pca_result[:, 1]
        pca_ls_space["pca-three"] = pca_result[:, 2]
        print(
            "Explained variation per principal component: {}".format(pca.explained_variance_ratio_)
        )

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=0, perplexity=200, n_iter=1000)
        tsne_results = tsne.fit_transform(pca_ls_space)
        print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))
        plt.figure()
        plot_tsne(tsne_results, epoch_history["cluster"].values, str(index))

    generate_plot("Reconstruction Loss", rc_loc)
    generate_plot("CCE Loss", cce_loc)
    generate_plot("FCM Loss", fcm_loc)
    generate_plot("Silhoutte Score", sil_loc)
    generate_plot("Trace Loss", trace_loc)

