import os
import pandas as pd

gcj_dir = 'dataset/GCJ'
extracted_dir = os.path.join(gcj_dir, 'googlejam4')

# Load clone pairs (headerless, two columns: file_a, file_b)
clone_df = pd.read_csv(os.path.join(gcj_dir, 'GCJ_clone.csv'), header=None, names=['path_a', 'path_b'])
clone_df['label'] = 1

# Load non-clone pairs
noclone_df = pd.read_csv(os.path.join(gcj_dir, 'GCJ_noclone_270000.csv'), header=None, names=['path_a', 'path_b'])
noclone_df['label'] = 0

# Combine
pairs_df = pd.concat([clone_df, noclone_df], ignore_index=True)

# Add full paths
pairs_df['path_a'] = pairs_df['path_a'].apply(lambda x: os.path.join(extracted_dir, x))
pairs_df['path_b'] = pairs_df['path_b'].apply(lambda x: os.path.join(extracted_dir, x))

# Verify files exist
pairs_df['exists_a'] = pairs_df['path_a'].apply(os.path.exists)
pairs_df['exists_b'] = pairs_df['path_b'].apply(os.path.exists)

print(f"Total pairs: {len(pairs_df)}")
print(f"Clone pairs: {(pairs_df['label']==1).sum()}")
print(f"Non-clone pairs: {(pairs_df['label']==0).sum()}")
print(f"Both files exist: {(pairs_df['exists_a'] & pairs_df['exists_b']).sum()}")

# Keep only valid pairs
valid_df = pairs_df[pairs_df['exists_a'] & pairs_df['exists_b']].copy()
valid_df = valid_df[['path_a', 'path_b', 'label']]

# Add an ID column (required by runner)
valid_df.insert(0, 'id', range(len(valid_df)))

# Save
output_path = os.path.join(gcj_dir, 'pairs_mapped.csv')
valid_df.to_csv(output_path, index=False)
print(f"\nSaved {len(valid_df)} valid pairs to {output_path}")
print("Sample:\n", valid_df.head())
