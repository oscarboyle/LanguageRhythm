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
df = pd.read_csv("/Users/elenanieto/Documents/SMC/AMPLAB/env/all_features_cleaned_filtered.csv", encoding='ISO-8859-1')

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
    # Modified title for the feature
    title = feature.replace('_', ' ').title()
    # Ensure nPVI is displayed correctly
    title = title.replace("Npvi", "nPVI")
    plt.title(f"{title} by language")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Histograms / KDE plots by language
for feature in rhythmic_features:
    try:
        sns.displot(data=df, x=feature, hue="language", kind="kde", fill=True)
        # Modified title for the feature
        title = feature.replace('_', ' ').title()
        # Ensure nPVI is displayed correctly
        title = title.replace("Npvi", "nPVI")
        plt.title(f"KDE of {title} by language")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting {feature}: {e}")

# Correlation matrix
corr = df[rhythmic_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation matrix of rhythmic descriptors")
plt.show()

# ---------- COMPARATIVE ANALYSIS BY LANGUAGE (Save Results) ----------

anova_results = []
posthoc_results = {}

for feature in rhythmic_features:
    data = df[["language", feature]].dropna()
    groups = [g[feature] for _, g in data.groupby("language")]

    if all(len(g) >= 3 for g in groups):  # suficientes muestras
        # Test de normalidad
        normal = all(stats.shapiro(g[:500])[1] > 0.05 for g in groups)

        if normal:
            stat, p = stats.f_oneway(*groups)
            test_type = "ANOVA"
        else:
            stat, p = stats.kruskal(*groups)
            test_type = "Kruskal-Wallis"

        anova_results.append({
            "feature": feature,
            "test": test_type,
            "stat": stat,
            "p_value": p
        })

        # Post-hoc
        if p < 0.05:
            if test_type == "ANOVA":
                tukey = pairwise_tukeyhsd(data[feature], data["language"])
                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                posthoc_results[feature] = tukey_df
            else:
                dunn = sp.posthoc_dunn(data, val_col=feature, group_col="language", p_adjust="bonferroni")
                posthoc_results[feature] = dunn

    else:
        anova_results.append({
            "feature": feature,
            "test": "Skipped (insufficient data)",
            "stat": np.nan,
            "p_value": np.nan
        })

# Guardar resultados ANOVA/Kruskal-Wallis
anova_df = pd.DataFrame(anova_results)
anova_df.to_csv("anova_kruskal_results.csv", index=False)

# Guardar cada tabla post-hoc por separado
for feature, result in posthoc_results.items():
    filename = f"posthoc_{feature}.csv"
    if isinstance(result, pd.DataFrame):
        result.to_csv(filename)
    else:
        result.to_csv(filename)  # también guarda si es matriz de Dunn

print("Resultados guardados en archivos CSV.")


# ---------- Boxplots by genre (only English, only selected features) ----------

selected_features = ["rhythmic_density", "nPVI", "norm_lz_complexity"]
df_en = df[df["language"] == "English"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
axes = axes.flatten()

for i, feature in enumerate(selected_features):
    sns.violinplot(data=df_en, x="genre", y=feature, ax=axes[i], inner="box", cut=0)
    # Modified title for the feature
    title = feature.replace('_', ' ').title()
    # Ensure nPVI is displayed correctly
    title = title.replace("Npvi", "nPVI")
    axes[i].set_title(f"{title} by genre (English only)")
    axes[i].tick_params(axis="x", rotation=45)

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
