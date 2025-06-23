import sqlite3
import re
from pathlib import Path
import stanza
import spacy
from keybert import KeyBERT

CHV_DB_PATH = Path("chv.db")

'Download models once (comment out after)'
# stanza.download('en')
# spacy.cli.download("en_core_web_sm")
'=========================================='

nlp_stanza = stanza.Pipeline(
    lang='en',
    processors='tokenize,pos,lemma,depparse',
    use_gpu=False
)

# spaCy for noun‐chunk extraction
nlp_spacy = spacy.load("en_core_web_sm")

# KeyBERT for fallback keyphrase extraction
kw_model = KeyBERT()

conn = sqlite3.connect(CHV_DB_PATH)
cur  = conn.cursor()

def normalize_and_tokenize(text):
    """
    1) Lowercase
    2) Strip out non-word (keep hyphens)
    3) Collapse whitespace
    4) Split on spaces
    """
    t = text.lower()
    t = re.sub(r"[^\w\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split()

def pos_and_lemmatize(text):
    """
    Use Stanza to get tokens + POS + lemma + dependencies.
    Returns:
      tokens_info: [{'word','pos','lemma','head','deprel'}, …]
      dependencies: [(gov, rel, dep), …]
    """
    doc = nlp_stanza(text)
    tokens_info = []
    dependencies = []
    for sent in doc.sentences:
        for w in sent.words:
            tokens_info.append({
                'word':   w.text,
                'pos':    w.upos,
                'lemma':  w.lemma,
                'head':   w.head,
                'deprel': w.deprel
            })
        for gov, rel, dep in sent.dependencies:
            dependencies.append((gov.text, rel, dep.text))
    return tokens_info, dependencies

def extract_candidates(tokens_info, dependencies, text):
    """
    Build a set of candidate terms from:
      - noun chunks (spaCy)
      - noun/verb/adjective/adverb lemmas (Stanza)
      - 1–2 word keyphrases (KeyBERT)
    """
    cands = set()

    # noun chunks
    doc_sp = nlp_spacy(text)
    for chunk in doc_sp.noun_chunks:
        cands.add(chunk.text)

    # POS-based lemmas
    for tok in tokens_info:
        pos = tok['pos']
        lem = tok['lemma']
        if pos.startswith('N') or pos.startswith('V') or pos in ('JJ','RB'):
            cands.add(lem)

    # fallback keyphrases
    for phrase, _ in kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=5):
        cands.add(phrase)

    return list(cands)

def map_to_chv(candidates):
    """
    Look up each candidate in the CHV table.
    Returns dict: term -> list of {'cui','preferred','score'}.
    """
    mapped = {}
    for term in candidates:
        cur.execute(
            "SELECT cui, display_name, combo_score FROM chv WHERE term = ?",
            (term.lower(),)
        )
        rows = cur.fetchall()
        mapped[term] = [
            {'cui': row[0], 'preferred': row[1], 'score': row[2]}
            for row in rows
        ]
    return mapped

def build_queries(mapped_terms):
    """
    Build a single Boolean AND–joined query from ALL mapped term strings.
    """
    terms = list(mapped_terms.keys())
    # wrap each in quotes to preserve multi-word phrases
    return " AND ".join(f'("{t}")' for t in terms)

def main():
    q = 'What type of extracolonic tumors does the PMS2 germline mutation cause?'
    tokens, _    = normalize_and_tokenize(q), None
    toks_info, deps = pos_and_lemmatize(q)
    cands        = extract_candidates(toks_info, deps, q)
    mapped       = map_to_chv(cands)
    query_string = build_queries(mapped)

    print("Candidate terms:", cands)
    print("Mapped CHV entries:", mapped)
    print("Initial Boolean Query:")
    print(query_string)

if __name__ == "__main__":
    main()
