import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load JSON ===
with open('bug-data/NL2SQL-Bugs.json', 'r') as f:
    data = json.load(f)

# === Step 2: Collect (error_type, sub_error_type) where label == False ===
records = []
for item in data:
    if item.get('label') is False:
        for err in item.get('error_types', []):
            records.append({
                'error_type': err['error_type'],
                'sub_error_type': err['sub_error_type']
            })

# === Step 3: Count combinations ===
df = pd.DataFrame(records)
count_df = df.value_counts().reset_index(name='count')

# === Step 4: Pivot table for stacked bar plot ===
pivot_df = count_df.pivot(index='error_type', columns='sub_error_type', values='count').fillna(0)

# === Step 5: Assign dark color palette (same color family per error_type) ===
error_types = pivot_df.index.tolist()
base_colors = sns.color_palette("Set2", n_colors=len(error_types))
error_type_to_color = dict(zip(error_types, base_colors))

# Build sub_error_type to color mapping
sub_error_to_color = {}
for error_type in error_types:
    sub_errors = [col for col in pivot_df.columns if df[df['sub_error_type'] == col]['error_type'].iloc[0] == error_type]
    shades = sns.dark_palette(error_type_to_color[error_type], n_colors=len(sub_errors), input="rgb", reverse=False)
    for i, sub in enumerate(sub_errors):
        sub_error_to_color[sub] = shades[i]

# === Step 6: Plot with improved layout ===
fig, ax = plt.subplots(figsize=(12, 7))

pivot_df.plot(
    kind='bar',
    stacked=True,
    color=[sub_error_to_color[col] for col in pivot_df.columns],
    ax=ax
)

# === Step 7: Labels and Title ===
ax.set_ylabel("Number of Occurrences", fontsize=12)
ax.set_xlabel("Error Type", fontsize=12)
ax.set_title("Distribution of Semantic Error Types and Subtypes", fontsize=14, pad=15)

# === Step 8: Adjust Legend (move below with multiple columns) ===
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title="Sub Error Type",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    fontsize=9,
    title_fontsize=10,
    frameon=False
)

# === Step 9: Final layout and save ===
plt.tight_layout()
plt.savefig("error_type_distribution_fixed.pdf", bbox_inches='tight')  # use .png if needed
plt.show()
