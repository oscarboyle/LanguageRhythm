import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# General configuration
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load data
df = pd.read_csv("/Users/elenanieto/Documents/SMC/AMPLAB/env/features/notAnnotatedWiki_rhythmic_features.csv")

# List of rhythmic descriptors
rhythmic_features = [
    "rhythmic_density", "avg_ioi", "stdev_ioi", "nPVI",
    "lz_complexity", "norm_lz_complexity"
]

# Convert columns to numeric (just in case)
for feature in rhythmic_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

# ---------- DESCRIPTIVE STATISTICS ----------

# Boxplots / Violin plots by language
for feature in rhythmic_features:
    sns.violinplot(x="language", y=feature, data=df, inner="box")
    plt.title(f"{feature} by language")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Histograms / KDE plots by language
for feature in rhythmic_features:
    try:
        sns.displot(data=df, x=feature, hue="language", kind="kde", fill=True)
        plt.title(f"KDE of {feature} by language")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting {feature}: {e}")

# Correlation matrix
corr = df[rhythmic_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation matrix of rhythmic descriptors")
plt.show()

# ---------- COMPARATIVE ANALYSIS BY LANGUAGE ----------

# ANOVA or Kruskal-Wallis tests
for feature in rhythmic_features:
    groups = [g[feature].dropna() for _, g in df.groupby("language")]
    if all(len(g) >= 3 for g in groups):  # avoid errors with very small samples
        if all(stats.shapiro(g[:500])[1] > 0.05 for g in groups):  # normality
            stat, p = stats.f_oneway(*groups)
            test_type = "ANOVA"
        else:
            stat, p = stats.kruskal(*groups)
            test_type = "Kruskal-Wallis"
        print(f"{test_type} for {feature}: stat = {stat:.2f}, p = {p:.4f}")

        # Post-hoc test
        if p < 0.05:
            print("Post-hoc test (Tukey if ANOVA, Dunn if Kruskal)")
            if test_type == "ANOVA":
                tukey = pairwise_tukeyhsd(df[feature].dropna(), df["language"][df[feature].notna()])
                print(tukey.summary())
            else:
                dunn = sp.posthoc_dunn(df, val_col=feature, group_col="language", p_adjust="bonferroni")
                print(dunn)
    else:
        print(f"Skipping {feature}: not enough samples per language.")

# ---------- GENRE AND STYLE CONTROL ----------

# Boxplots by language and genre
for feature in rhythmic_features:
    g = sns.catplot(data=df, x="language", y=feature, hue="Genre", kind="box", height=6, aspect=2)
    g.fig.suptitle(f"{feature} by language and genre")
    g.set_xticklabels(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------- CLUSTERING & DIMENSIONALITY REDUCTION ----------

# Preprocessing
X = df[rhythmic_features].dropna()
X_scaled = StandardScaler().fit_transform(X)
languages = df.loc[X.index, "language"]

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=languages)
plt.title("PCA of rhythmic descriptors")
plt.show()

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=languages)
plt.title("t-SNE of rhythmic descriptors")
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10")
plt.title("K-Means Clustering in PCA space")
plt.show()
