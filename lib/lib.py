import spacy
import re

nlp = spacy.load("en_core_web_sm",disable=["ner", "parser"])
# 1. Test Preprossesing with Pands
def lower_replace(series):
    # 1. Lowercase
    output = series.str.lower()
    # 2. Remove text in brackets [like this]
    output = output.str.replace(r'\[.*?\]', '', regex=True)
    # 3. KEEP ONLY a-z, 0-9, and whitespace (removes  and all punctuation)
    output = output.str.replace(r'[^a-z0-9\s]', '', regex=True)
    # 4. Clean up double spaces
    output = output.str.replace(r'\s+', ' ', regex=True).str.strip()
    return output

#Tokenize : Lemmatiza : stop word
def token_lemm_nonstop(text):
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    output= ' '.join(output)
    return output

# parts of speech tagging
def filter_pos(text, pos_list=["NOUN","PROPN"]):
    doc = nlp(text)
    output = [token.text for token in doc if token.pos_ in pos_list]
    output = ' '.join(output)
    return output

# 3. NLP Pipeline
def nlp_pipeline(series):
    print("lower_replace")
    output = lower_replace(series)
    print("token_lemm_nonstop")
    output = output.apply(token_lemm_nonstop)
    print("filter_pos")
    output = output.apply(filter_pos)
    return output

def pre_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)       # remove [brackets]
    text = re.sub(r'[^a-z\s]', ' ', text)     # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

def fast_nlp_pipeline(texts, batch_size=1000):
    """
    texts: list or pandas Series of strings
    returns: list of cleaned strings
    """
    output_texts = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        # Keep only lemmas that are not stopwords and are NOUN or PROPN
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.pos_ in ("NOUN","PROPN")
        ]
        output_texts.append(" ".join(tokens))
    return output_texts