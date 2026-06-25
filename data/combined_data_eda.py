import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting style for clear data visualization
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv("data/combined_data.csv")

# --- PLOT 1: SCATTER PLOT (Continuous vs. Continuous with Color Grouping) ---
plt.figure()
sns.scatterplot(
    data=df, 
    x="Overall_postings", 
    y="tpi_average_usd_per_m_tokens",
    palette="Dark2", 
    s=70
)
plt.title("TPI vs. Overall Job Postings", fontsize=14, pad=15)
plt.xlabel("Overall Postings")
plt.ylabel("TPI Average (USD per Million Tokens)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("01_tpi_scatter.png", dpi=150)
plt.show()