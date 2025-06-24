import sqlite3
import re
from pathlib import Path
import stanza
import spacy
from keybert import KeyBERT
import requests

CHV_DB_PATH          = Path("chv.db")
UMLS_API_KEY         = ''
UMLS_SEARCH_ENDPOINT = "https://uts-ws.nlm.nih.gov/rest/search/current"

# stanza.download('en')                 # run once
# spacy.cli.download("en_core_web_sm")  # run once

nlp_stanza = stanza.Pipeline(
    lang='en', processors='tokenize,pos,lemma,depparse', use_gpu=False
)
nlp_spacy  = spacy.load("en_core_web_sm")
kw_model   = KeyBERT()

conn = sqlite3.connect(CHV_DB_PATH)
cur  = conn.cursor()

def query_umls(term, page_size=5):
    """
    Perform a general UMLS Search for `term` (no restrictions),
    print CUI, name, and score of each hit, and return the hit list.
    """
    params = {
        "string":   term,
        "pageSize": page_size,
        "apiKey":   UMLS_API_KEY
    }
    r = requests.get(UMLS_SEARCH_ENDPOINT, params=params)
    if not r.ok:
        print(f"UMLS search error {r.status_code}: {r.text}")
        return []

    hits = r.json().get("result", {}).get("results", [])
    if not hits:
        print(f"No UMLS hits found for '{term}'")
        return []

    print(f"Top {len(hits)} UMLS hits for '{term}':")
    for hit in hits:
        print(f"  CUI={hit.get('ui')}, name={hit.get('name')}, score={hit.get('score')}")
    return hits


def normalize_and_tokenize(text):
    t = text.lower()
    t = re.sub(r"[^\w\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split()

def pos_and_lemmatize(text):
    doc = nlp_stanza(text)
    tokens_info, dependencies = [], []
    for sent in doc.sentences:
        for w in sent.words:
            tokens_info.append({
                'word':   w.text,
                'lemma':  w.lemma,
                'pos':    w.upos,  # Universal POS
                'xpos':   w.xpos,  # Penn POS
                'head':   w.head,
                'deprel': w.deprel
            })
        for gov, rel, dep in sent.dependencies:
            dependencies.append((gov.text, rel, dep.text))
    return tokens_info, dependencies

ALLOWED_XPOS = {'JJ','JJR','JJS','RB','RBR','RBS'}

def passes_pos_filter(term, tokens_info):
    """
    Return True if every word in `term` has
    upos.startswith('N') or upos.startswith('V')
    OR xpos in ALLOWED_XPOS.
    """
    for word in term.split():
        matches = [
            tok for tok in tokens_info
            if tok['lemma'].lower() == word.lower()
            or tok['word'].lower()   == word.lower()
        ]
        if not matches:
            return False
        if not any(
            tok['pos'].startswith(('N','V')) or tok['xpos'] in ALLOWED_XPOS
            for tok in matches
        ):
            return False
    return True

def extract_candidates(tokens_info, dependencies, text):
    """
    1) spaCy noun chunks
    2) Stanza POS-lemma (single tokens)
    3) KeyBERT 1–2-grams
    THEN filter all by passes_pos_filter.
    """
    raw = set()

    # noun-chunks
    for chunk in nlp_spacy(text).noun_chunks:
        raw.add(chunk.text)

    # Stanza POS-based lemmas
    for tok in tokens_info:
        if tok['pos'].startswith(('N','V')) or tok['xpos'] in ALLOWED_XPOS:
            raw.add(tok['lemma'])

    # KeyBERT fallback
    for phrase, _ in kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,3),
        stop_words='english',
        top_n=5
    ):
        raw.add(phrase)

    # apply POS filter to every candidate
    return [term for term in raw if passes_pos_filter(term, tokens_info)]

def map_and_build_query(candidates):
    """
    For each candidate:
      • If CHV hits exist → include CHV display_name(s)
      • Else if UMLS hits exist → include original term + top 2 UMLS names
      • Else → drop term
    Returns a list.
    """
    final_terms = []

    for term in candidates:
        # CHV lookup
        cur.execute(
            "SELECT display_name FROM chv WHERE term = ?",
            (term.lower(),)
        )
        chv_rows = cur.fetchall()

        if chv_rows:
            # include each display_name
            for (disp,) in chv_rows:
                if disp not in final_terms:
                    final_terms.append(disp)
            continue

        # UMLS fallback
        hits = query_umls(term, page_size=2)
        if not hits:
            # drop term if no UMLS match
            continue

        # include original term itself
        if term not in final_terms:
            final_terms.append(term)
        # include up to top 2 UMLS match names
        for hit in hits[:2]:
            name = hit.get('name')
            if name and name not in final_terms:
                final_terms.append(name)

    return final_terms


def main():
    q = """I am 30, 6\"2, 215, white male, I have a hiatal hernia, barrett's esophagus, a PFO, and had high cholesterol but it was treated with diet. \n\nSo I am not anti-vaccine, and I know how damaging Covid can be, I am interested in getting the vaccine as is my mom, but I am naturally shy to take taking medications for fears of side effects, and in this case it doesnt help that 75% of my family has bought fully into every Covid vaccine conspiracy there is. I don't believe in a vast amount of garbage that is brought out by conspiracies but I do have some concerns. I do know have people I have associated with who claim to know people that have had either strokes or died after getting the vaccine, though while it would be odd I dont know if any of that could be confirmed to be from the vaccine. \n\nOne thing I am curious about, is, do we have any idea what the odds actually are long term that something could go wrong from the vaccine, im talking 1-5-10-15 years down the line is there any real possibility that something could go wrong and cause health issues that long from now or would it almost always be something in the first 2 weeks. \n\nAnother thing that I assume is just fully incorrect, but would like to know for sure. My dad keeps saying that the vaccine to rapidly reproduce spike protein, which then sticks to your blood vessels and causes clots, strokes and other issues. He also claims theres many people getting nerve damage from it, though im sure I've seen that around online and seen it debunked. \n\nI do know a lot of people of all age ranges that have gotten it, and I dont personally know anyone with a major reaction to the vaccine, so it does puzzle how some people I see online say they can know 2 to 3 people who died from the vaccine, I've never been able to explain it, and I cant really refute their stories because I dont know them personally, but thats stuck out to me too."""
    # q = 'Are long non coding RNAs spliced?'

    tokens_info, deps = pos_and_lemmatize(q)
    cands            = extract_candidates(tokens_info, deps, q)
    query_string     = map_and_build_query(cands)

    print("Candidate terms:", cands)
    print("Final Boolean Query:", query_string)


if __name__ == "__main__":
    main()
