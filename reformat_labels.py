import pandas as pd

# Load the original CSV
df = pd.read_csv("new_labells_raw.csv")  # Replace with your actual filename

# Format the 'id' column
df["id"] = df["id"].apply(lambda x: f"labelled_{int(x):05d}")

# Sort by the new formatted 'id'
df = df.sort_values(by="id").reset_index(drop=True)

# Save the result
df.to_csv("new_labelled.csv", index=False)

print("Saved to 'labels_formatted_sorted.csv'")
