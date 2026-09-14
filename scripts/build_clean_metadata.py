"""Build a slim, analysis-ready version of the songs metadata.

Reads data/songs_metadata.csv (84 columns, full pipeline output with raw API
fields, intermediate fallback columns, and resolved "final" columns all mixed
together) and writes data/songs_metadata_clean.csv containing only:

- identifiers / core song stats (song_id, artist, title, n_verses,
  lyrics_len)
- resolved release info (release_date, release_year, decade, album)
- resolved artist demographics and genre (country, gender_or_group,
  genres_all, genre_main)
- the misogyny labels and train/test split flags

Raw MusicBrainz/Wikidata fields, `fill_*` fallback columns, `deep_*`
intermediate search results, `*_fuente` provenance columns, AcousticBrainz
audio features, and QA/debug columns are dropped -- they were already folded
into the resolved columns kept here, or aren't needed for downstream
analysis/modeling.
"""

import pandas as pd

SRC = "data/songs_metadata.csv"
DST = "data/songs_metadata_clean.csv"

KEEP_RENAME = {
    "song_id": "song_id",
    "artist": "artist",
    "title": "title",
    "n_verses": "n_verses",
    "lyrics_len": "lyrics_len",
    "release_date": "release_date",
    "release_year": "release_year",
    "decade": "decade",
    "album_final": "album",
    "country": "country",
    "gender_or_group": "gender_or_group",
    "genres_all": "genres_all",
    "genre_main": "genre_main",
    "is_misogynistic": "is_misogynistic",
    "type_sexualization": "type_sexualization",
    "type_violence": "type_violence",
    "type_hate": "type_hate",
    "has_gender_stereotype": "has_gender_stereotype",
    "in_train": "in_train",
    "in_test": "in_test",
}

df = pd.read_csv(SRC)

# release_year is the one column the ask explicitly wanted "filled" -- the
# pipeline already resolves it for every row, so this is a guard, not a fix.
assert df["release_year"].isna().sum() == 0, "release_year has unexpected gaps"
assert df["genre_main"].isna().sum() == 0, "genre_main has unexpected gaps"
assert df["genres_all"].isna().sum() == 0, "genres_all has unexpected gaps"

clean = df[list(KEEP_RENAME)].rename(columns=KEEP_RENAME)

# Whole-number columns come in as float64 purely from how they were written
# out upstream; cast them to a plain/nullable int so the clean CSV reads as
# int, not "2024.0".
clean["release_year"] = clean["release_year"].astype("int64")
clean["decade"] = clean["decade"].astype("int64")
clean["n_verses"] = clean["n_verses"].astype("int64")
clean["lyrics_len"] = clean["lyrics_len"].astype("int64")

# 0/1 flags that are NaN for train-set NM rows (no subtype was annotated) --
# nullable Int64 keeps that distinction instead of silently becoming float.
for col in ["type_sexualization", "type_violence", "type_hate"]:
    clean[col] = clean[col].astype("Int64")

clean.to_csv(DST, index=False)

print(f"Wrote {DST}: {clean.shape[0]} rows x {clean.shape[1]} columns")
print()
print("dtypes:")
print(clean.dtypes)
print()
print("missing values per column:")
print(clean.isna().sum())
