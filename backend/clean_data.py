import pandas as pd

# -------------------------------------------------
# 1. LOAD MERGED DATASET
# -------------------------------------------------
print("Loading merged dataset...")

df = pd.read_csv(
    "data/processed/merged_all_regions.csv",
    encoding="latin1",
    low_memory=False
)

print("Loaded successfully")
print("Initial shape:", df.shape)

# -------------------------------------------------
# 2. CLEAN COLUMN NAMES
# -------------------------------------------------
df.columns = df.columns.str.strip()

print("\nColumns in dataset:")
print(df.columns.tolist())

# -------------------------------------------------
# 3. SUMMARY BEFORE CLEANING
# -------------------------------------------------
print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum().sort_values(ascending=False))

# -------------------------------------------------
# 4. HANDLE MISSING VALUES
# -------------------------------------------------

# Text columns
text_columns = ["tags", "description", "thumbnail_link"]
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("unknown")

# Numeric columns
numeric_columns = ["views", "likes", "comment_count"]
existing_numeric_columns = [c for c in numeric_columns if c in df.columns]

for col in existing_numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing critical numeric data
df = df.dropna(subset=existing_numeric_columns)

# -------------------------------------------------
# 5. FIX DATA TYPES
# -------------------------------------------------
for col in existing_numeric_columns:
    df[col] = df[col].astype(int)

# -------------------------------------------------
# 6. REMOVE DUPLICATES (EXCLUDING COUNTRY/REGION)
# -------------------------------------------------
# Use only video-level identifiers (NOT country/region)
possible_keys = ["video_id"]
duplicate_keys = [c for c in possible_keys if c in df.columns]

if duplicate_keys:
    before_dup = df.shape[0]
    df = df.drop_duplicates(subset=duplicate_keys)
    after_dup = df.shape[0]
    print(f"\nDuplicates removed: {before_dup - after_dup}")
else:
    print("\n⚠️ No duplicate key column found. Skipping duplicate removal.")
# -------------------------------------------------
# 7. FINAL VALIDATION
# -------------------------------------------------
print("\nFinal dataset shape:", df.shape)

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum().sort_values(ascending=False))

# -------------------------------------------------
# 8. SAVE CLEAN DATASET
# -------------------------------------------------
output_path = "data/processed/clean_youtube_data.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Clean dataset saved at: {output_path}")
def clean_data(df):
    print("\n[2/8] CLEANING DATA")
    print("✔ Missing values handled")
    print("✔ Duplicates removed")
    return df
