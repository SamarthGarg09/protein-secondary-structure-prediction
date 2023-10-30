import pandas as pd
import re

# Assuming df is the dataframe containing your data and 'dssp8' and 'dssp3' are the columns for Q8 and Q3 sequences respectively.

def load_dataset(path):
    df = pd.read_excel(path)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.shape)
    return df

df = load_dataset("/Data/deeksha/disha/ProtTrans/data/final_excel_files/df_final.xlsx")
# %%
df['input_x'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_x']]
X = df['input_x'].values
y = df['dssp8'].values

# Initialize counters for Q8 and Q3 labels
q8_counter = {'B': 0, 'C': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0, 'S': 0, 'T': 0}
q3_counter = {'C': 0, 'E': 0, 'H': 0}

# Iterate through Q8 sequences and count label occurrences
for seq in df['dssp8']:
    for label in seq.split():
        q8_counter[label] += 1

# Iterate through Q3 sequences and count label occurrences
for seq in df[' dssp3']:
    for label in seq.split():
        q3_counter[label] += 1

# Convert counters to dataframes
df_q8 = pd.DataFrame(list(q8_counter.items()), columns=['Label', 'Count']).set_index('Label')
df_q3 = pd.DataFrame(list(q3_counter.items()), columns=['Label', 'Count']).set_index('Label')

# Save dataframes to Excel
df_q8.to_excel("q8_counts.xlsx")
df_q3.to_excel("q3_counts.xlsx")
