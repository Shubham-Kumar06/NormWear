import pandas as pd
import pickle

# Load pickle file
with open("data/results/downstream_results/epoch480_test_results_all.pkl", "rb") as f:
    data = pickle.load(f)

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel("output.xlsx", index=False)

print("Conversion successful!")