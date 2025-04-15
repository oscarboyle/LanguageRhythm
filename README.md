# LanguageRhythm

**LanguageRhythm** is a Python toolkit for analyzing rhythmic features in songs across different languages and musical genres. It works with symbolic music formats such as MusicXML and Kern to extract descriptors, perform statistical comparisons, visualize distributions, and explore clustering in feature space.

## Features

- **Rhythmic feature extraction**  
  Computes metrics such as:
  - `rhythmic_density`  
  - `avg_ioi`, `stdev_ioi` (inter-onset intervals)  
  - `nPVI` (normalized Pairwise Variability Index)  
  - `lz_complexity`, `norm_lz_complexity` (Lempel-Ziv complexity)

- **Statistical analysis**  
  Group comparisons by language using ANOVA or Kruskal-Wallis, with post-hoc tests (Tukey HSD or Dunn's test with Bonferroni correction)

- **Visualization tools**  
  Violin plots, KDE plots, and correlation matrices to explore the distribution and relationship between rhythmic features

- **Dimensionality reduction and clustering**  
  Unsupervised analysis using PCA, t-SNE, and K-Means to discover structure in the data

## Installation

Clone the repository and install the required dependencies listed in `requirements.txt`.
```bash
git clone https://github.com/oscarboyle/LanguageRhythm.git
cd LanguageRhythm
pip install -r requirements.txt
```

## Data structure

The system works with `.csv` files that reference symbolic music files (`.mxl` for MusicXML or `.krn` for Kern), and include metadata such as language, genre, and publication year.

### Datasets used

- **PDMX**: Public domain MusicXML corpus  
- **OpenEWLD**: Curated subset of EWLD dataset with metadata (language, genre, year, author...)  
- **Wikifonia**: Legacy leadsheets dataset (before platform shutdown)  
- **Essen Folk Collection**: Over 40,000 traditional folk melodies (in Kern format)

### Processing steps

1. Generate `.csv` index files with .mxl paths using scripts like:
   - `folkKernProcessor.py`
   - `notAnnotatedWikiProcessor.py`

   Examples of output files (data folder):
   - `OpenEWLD_data.csv`
   - `PDMX_data.csv`
   - `folkKern_data.csv`
   - `notAnnotatedWiki_data.csv`

2. Extract rhythmic features using `rhythmExtractor.py`. This adds columns such as:
   - `rhythmic_density`, `avg_ioi`, `stdev_ioi`, `nPVI`, `lz_complexity`, `norm_lz_complexity`

3. Merge into a unified dataset (features folder):  
   - `all_features_clean.csv` (raw output)  
   - `all_features_cleaned_filtered.csv` (excluded pop/rock genres in languages other than English)

## Usage

Prepare a `.csv` dataset that includes metadata (language, genre) and the extracted rhythmic descriptors.

Run the main analysis script (analysis.py) to generate:
- Descriptive statistics by language
- Visualizations of distributions
- Statistical tests and post-hoc comparisons
- PCA and t-SNE projections
- K-Means clustering on rhythmic features

The results will include:
- Visual plots
- Statistical output files:
  - `anova_kruskal_results.csv`
  - `posthoc_<feature>.csv` (per-feature post-hoc analysis)

## Dependencies

This project requires the following Python libraries:
- pandas, numpy
- matplotlib, seaborn
- scipy, statsmodels, scikit-posthocs
- scikit-learn

All dependencies are listed in the `requirements.txt` file.

## Citation & Acknowledgements

This toolkit was developed as part of a study on the relationship between musical rhythm and language, using symbolic music analysis. It supports reproducible workflows for computational musicology and cross-linguistic rhythm studies.

## License

MIT License. See `LICENSE` for details.
