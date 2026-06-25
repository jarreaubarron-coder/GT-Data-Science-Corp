import pandas as pd
import numpy as np

jobs = pd.read_csv("serpapi_raw_data.csv")
print("Initial DataFrame shape:", jobs.shape)
print(jobs)