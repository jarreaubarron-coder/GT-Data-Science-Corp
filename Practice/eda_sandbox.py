"""
Data Science Corps 2026 — Master Sandbox Lab
Topic: Universal Ingestion, Data Cleaning, and EDA Workflow
Datasets: Palmer Penguins & Iris
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting style for clear data visualization
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# PHASE 1: DATA INGESTION & PIPELINE SETUP
# ==========================================
print("🚀 Phase 1: Fetching raw datasets from remote repositories...")

# URL sources for raw data access
URL_PENGUINS = "https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/master/palmerpenguins/data/penguins.csv"
URL_IRIS = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# Load data into memory
penguins_raw = pd.read_csv(URL_PENGUINS)
iris_raw = pd.read_csv(URL_IRIS)

# ==========================================
# PHASE 2: PROGRAMMATIC INSPECTION
# ==========================================
print("\n🔍 Phase 2: Programmatic Inspection...")

def inspect_dataset(df, name):
    print(f"\n--- {name.upper()} DATA PROFILE ---")
    print(f"Shape (Rows x Columns): {df.shape}")
    print("\nData Types & Memory Allocation:")
    print(df.info())
    print("\nMissing Values Count per Column:")
    print(df.isnull().sum())

inspect_dataset(penguins_raw, "Raw Penguins")
inspect_dataset(iris_raw, "Raw Iris")

# ==========================================
# PHASE 3: THE CLEANING PIPELINE
# ==========================================
print("\n🧼 Phase 3: Executing Data Cleaning Operations...")

# 1. Processing Penguins (Has real-world structural missing data flaws)
penguins_clean = penguins_raw.copy()

# Drop rows where vital physical features or target classes are null
penguins_clean = penguins_clean.dropna(subset=['bill_length_mm', 'flipper_length_mm', 'sex'])

# Impute categorical missing values if any remain (demonstration placeholder)
penguins_clean['island'] = penguins_clean['island'].fillna('Unknown')

# 2. Processing Iris (Naturally clean, but good practice to standardize names)
iris_clean = iris_raw.copy()
# Ensuring all string values are completely lowercase to avoid grouping errors
iris_clean['species'] = iris_clean['species'].str.lower()

print(f"Penguins post-cleaning row check: {penguins_clean.shape[0]} rows remaining.")
print(f"Iris post-cleaning row check: {iris_clean.shape[0]} rows remaining.")

# ==========================================
# PHASE 4: STATISTICAL INVESTIGATION
# ==========================================
print("\n📊 Phase 4: Statistical Summaries and Aggregations...")

print("\nMean physical characteristics of Penguins grouped by Species:")
print(penguins_clean.groupby('species')[['bill_length_mm', 'flipper_length_mm', 'body_mass_g']].mean())

print("\nMedian measurements of Iris grouped by Species:")
print(iris_clean.groupby('species').median())

# ==========================================
# PHASE 5: EXPLORATORY DATA VISUALIZATIONS (5+ TYPES)
# ==========================================
# print("\n🎨 Phase 5: Generating Static Visualization Deliverables...")

# --- PLOT 1: SCATTER PLOT (Continuous vs. Continuous with Color Grouping) ---
plt.figure()
sns.scatterplot(
    data=penguins_clean, 
    x="bill_length_mm", 
    y="flipper_length_mm", 
    hue="species", 
    style="sex",
    palette="Dark2", 
    s=70
)
plt.title("Penguin Bill Length vs. Flipper Length by Species & Sex", fontsize=14, pad=15)
plt.xlabel("Bill Length (mm)")
plt.ylabel("Flipper Length (mm)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("01_penguin_scatter.png", dpi=150)
plt.show()

# --- PLOT 2: PAIR PLOT / MATRIX (Multivariate Feature Interaction) ---
pair_plot = sns.pairplot(data=iris_clean, hue="species", palette="muted", diag_kind="kde")
pair_plot.fig.suptitle("Iris Morphological Feature Correlations Matrix", y=1.02, fontsize=14)
pair_plot.savefig("02_iris_pairplot.png", dpi=150)
plt.show()

# --- PLOT 3: BOX PLOT (Categorical Class vs. Numeric Distribution) ---
plt.figure()
sns.boxplot(
    data=penguins_clean, 
    x="species", 
    y="body_mass_g", 
    hue="island",
    palette="Pastel1"
)
plt.title("Distribution of Penguin Body Mass Across Islands", fontsize=14, pad=15)
plt.xlabel("Species")
plt.ylabel("Body Mass (g)")
plt.tight_layout()
plt.savefig("03_penguin_boxplot.png", dpi=150)
plt.show()

# --- PLOT 4: HISTOGRAM WITH KDE OVERLAY (Univariate Distribution/Density) ---
plt.figure()
sns.histplot(
    data=iris_clean, 
    x="sepal_length", 
    hue="species", 
    kde=True, 
    element="step", 
    stat="density", 
    common_norm=False,
    alpha=0.4
)
plt.title("Density Distribution of Sepal Length Across Iris Species", fontsize=14, pad=15)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("04_iris_histogram.png", dpi=150)
plt.show()

# --- PLOT 5: VIOLIN PLOT (Alternative Distribution Showing Probability Density) ---
plt.figure()
sns.violinplot(
    data=penguins_clean, 
    x="species", 
    y="flipper_length_mm", 
    hue="sex", 
    split=True,
    inner="quart", 
    palette="muted")
plt.title("Flipper Length Probability Density by Species and Sex", fontsize=14, pad=15)
plt.xlabel("Species")
plt.ylabel("Flipper Length (mm)")
plt.tight_layout()
plt.savefig("05_penguin_violin.png", dpi=150)
plt.show()

# --- PLOT 6: HEATMAP (Linear Feature Correlation Matrix) ---
plt.figure()
# Select numeric columns only for mathematical processing
numeric_iris = iris_clean.select_dtypes(include=[np.number])
correlation_matrix = numeric_iris.corr()
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1, 
    linewidths=0.5)
plt.title("Iris Quantitative Metric Correlation Heatmap", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("06_iris_heatmap.png", dpi=150)
plt.show()

print("\n🎉 Pipeline execution complete! All 6 high-resolution visualization files saved to directory.")
