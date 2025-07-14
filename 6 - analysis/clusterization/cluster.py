import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# number of clusters
number_clusters = 4

# Create output folders
os.makedirs("labelled", exist_ok=True)
os.makedirs("non_labelled", exist_ok=True)

# Load dataset
df = pd.read_csv("mine.csv")
labels = df['X'].tolist()
features = df.drop(columns=['X'])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Marker style
marker_kwargs = dict(s=250, edgecolor='black', linewidth=2)

# ===== 1. KMeans clustering on multivariate data =====
kmeans = KMeans(n_clusters=number_clusters, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

for labelled in [True, False]:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df, x='PCA1', y='PCA2', hue='Cluster_KMeans',
        palette='Set1', ax=ax, **marker_kwargs
    )
    if labelled:
        for i, lab in enumerate(labels):
            ax.text(df['PCA1'][i] + 0.05, df['PCA2'][i] + 0.05, lab, fontsize=9)
    ax.set_title('Multivariate Clustering – KMeans + PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)
    ax.legend(title='Cluster')
    fig.tight_layout()
    
    suffix = "_lab" if labelled else "_nolabel"
    folder = "labelled" if labelled else "non_labelled"
    fig.savefig(os.path.join(folder, f"KMeans_PCA{suffix}.png"), dpi=300)
    plt.close(fig)

# ===== 2. Individual variable clustering with KMeans =====
for column in features.columns:
    values = features[[column]].values
    scaled = StandardScaler().fit_transform(values)
    clusters_ind = KMeans(n_clusters=number_clusters, random_state=42).fit_predict(scaled)

    for labelled in [True, False]:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            x=range(len(scaled)), y=scaled.flatten(), hue=clusters_ind,
            palette='Set2', ax=ax, **marker_kwargs
        )
        if labelled:
            for i, lab in enumerate(labels):
                ax.text(i + 0.1, scaled[i][0], lab, fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_title(f'Individual Clustering – {column}')
        ax.set_xlabel('Molecule Label')
        ax.set_ylabel(f'{column} (standardized)')
        ax.grid(True)
        ax.legend(title='Cluster')
        fig.tight_layout()

        safe_name = column.replace(" ", "_").replace("|", "abs").replace("(", "").replace(")", "")
        suffix = "_lab" if labelled else "_nolabel"
        folder = "labelled" if labelled else "non_labelled"
        fig.savefig(os.path.join(folder, f"individual_{safe_name}{suffix}.png"), dpi=300)
        plt.close(fig)

# ===== 3. Hierarchical Clustering (HCA) =====
Z = linkage(X_scaled, method='ward')  # You can try 'average', 'complete', etc.
df['Cluster_HCA'] = fcluster(Z, number_clusters, criterion='maxclust')

# --- HCA PCA scatter ---
for labelled in [True, False]:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df, x='PCA1', y='PCA2', hue='Cluster_HCA',
        palette='Set1', ax=ax, **marker_kwargs
    )
    if labelled:
        for i, lab in enumerate(labels):
            ax.text(df['PCA1'][i] + 0.05, df['PCA2'][i] + 0.05, lab, fontsize=9)
    ax.set_title('Hierarchical Clustering (HCA) – PCA projection')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)
    ax.legend(title='Cluster')
    fig.tight_layout()

    suffix = "_lab" if labelled else "_nolabel"
    folder = "labelled" if labelled else "non_labelled"
    fig.savefig(os.path.join(folder, f"HCA_PCA{suffix}.png"), dpi=300)
    plt.close(fig)

# --- HCA dendrogram ---
for labelled in [True, False]:
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        Z,
        labels=labels if labelled else None,
        leaf_rotation=90,
        leaf_font_size=10 if labelled else 0,
        color_threshold=None
    )
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Molecule" if labelled else "")
    ax.set_ylabel("Distance")
    fig.tight_layout()

    suffix = "_lab" if labelled else "_nolabel"
    folder = "labelled" if labelled else "non_labelled"
    fig.savefig(os.path.join(folder, f"HCA_dendrogram{suffix}.png"), dpi=300)
    plt.close(fig)
