import sqlite3
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
TSV_PATH    = PROJECT_DIR / "CHV_concepts_terms_flatfile_20110204.tsv"
DB_PATH     = PROJECT_DIR / "chv.db"

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS chv (
    cui TEXT,
    term TEXT,
    term_explanation TEXT,
    display_name TEXT,
    extra_field TEXT,
    is_consumer_preferred INTEGER,
    is_umls_preferred INTEGER,
    is_disparaged INTEGER,
    term_score1 REAL,
    cui_score1 REAL,
    term_score2 REAL,
    cui_score2 REAL,
    combo_score REAL,
    chv_string_id INTEGER,
    chv_concept_id INTEGER
);
""")
conn.commit()

if not TSV_PATH.exists():
    raise FileNotFoundError(f"TSV not found: {TSV_PATH}")

with open(TSV_PATH, newline='', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    batch = []
    for row in reader:
        if len(row) != 15:
            continue

        row[5] = 1 if row[5].strip().lower() in ("yes","1","true") else 0
        row[6] = 1 if row[6].strip().lower() in ("yes","1","true") else 0
        row[7] = 1 if row[7].strip().lower() in ("yes","1","true") else 0

        for i in range(8, 13):
            try:
                row[i] = float(row[i])
            except:
                row[i] = None

        try:
            row[13] = int(row[13])
        except:
            row[13] = None
        try:
            row[14] = int(row[14])
        except:
            row[14] = None

        batch.append(row)
        if len(batch) >= 500:
            cur.executemany(
                "INSERT INTO chv VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            batch.clear()

    if batch:
        cur.executemany(
            "INSERT INTO chv VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            batch
        )
        conn.commit()

conn.close()
print(f"Database created and populated at: {DB_PATH}")
