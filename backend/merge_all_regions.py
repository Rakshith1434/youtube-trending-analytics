import pandas as pd

files = {
    "CA": "data/raw/CAvideos.csv",
    "DE": "data/raw/DEvideos.csv",
    "FR": "data/raw/FRvideos.csv",
    "GB": "data/raw/GBvideos.csv",
    "IN": "data/raw/INvideos.csv",
    "JP": "data/raw/JPvideos.csv",
    "KR": "data/raw/KRvideos.csv",
    "MX": "data/raw/MXvideos.csv",
    "RU": "data/raw/RUvideos.csv",
    "US": "data/raw/USvideos.csv"
}

dfs = []

for region, path in files.items():
    print(f"Loading {path} ...")

    df = pd.read_csv(
        path,
        encoding="latin1",
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        on_bad_lines="skip"   # 🔥 KEY FIX
    )

    df["region"] = region
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

merged_df.to_csv(
    "data/processed/merged_all_regions.csv",
    index=False
)
print("✅ All regional YouTube datasets merged successfully")