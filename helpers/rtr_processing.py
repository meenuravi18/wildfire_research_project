import requests
import re
import spacy
from llama_index.core import Document
import numpy as np
import fitz
import pickle
nlp = spacy.load("en_core_web_sm")
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import os
import us
GEO_CACHE_FILE = "geo_cache.json"

if os.path.exists(GEO_CACHE_FILE):
    with open(GEO_CACHE_FILE, "r") as f:
        geo_cache = json.load(f)
else:
    geo_cache = {}

def cached_hierarchy(place):
    if place in geo_cache:
        return geo_cache[place]
    result = hierarchy_pipeline(place, "above")
    geo_cache[place] = result
    return result
us_states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming"
]
US_STATE_QIDS = {
    "Alabama": "Q173", "Alaska": "Q797", "Arizona": "Q816", "Arkansas": "Q1612",
    "California": "Q99", "Colorado": "Q1261", "Connecticut": "Q779", "Delaware": "Q1393",
    "Florida": "Q812", "Georgia": "Q1428", "Hawaii": "Q782", "Idaho": "Q1212",
    "Illinois": "Q1204", "Indiana": "Q1415", "Iowa": "Q1546", "Kansas": "Q1558",
    "Kentucky": "Q1603", "Louisiana": "Q1588", "Maine": "Q724", "Maryland": "Q1391",
    "Massachusetts": "Q771", "Michigan": "Q1166", "Minnesota": "Q1527", "Mississippi": "Q1494",
    "Missouri": "Q1581", "Montana": "Q1212", "Nebraska": "Q1553", "Nevada": "Q1227",
    "New Hampshire": "Q759", "New Jersey": "Q1408", "New Mexico": "Q1522", "New York": "Q1384",
    "North Carolina": "Q1454", "North Dakota": "Q1408", "Ohio": "Q1397", "Oklahoma": "Q1649",
    "Oregon": "Q1481", "Pennsylvania": "Q1400", "Rhode Island": "Q1387", "South Carolina": "Q1456",
    "South Dakota": "Q1211", "Tennessee": "Q1509", "Texas": "Q1439", "Utah": "Q829",
    "Vermont": "Q1669", "Virginia": "Q1370", "Washington": "Q1223", "West Virginia": "Q1371",
    "Wisconsin": "Q1537", "Wyoming": "Q1214",
}

HEADERS = {
    "User-Agent": "WildfireKG/1.0 (ravim@vt.edu)"
}
def batch_hierarchy_lookup(place_names):
    uncached = [p for p in place_names if p not in geo_cache]
    if not uncached:
        return

    with ThreadPoolExecutor(max_workers=10) as executor:
        qid_results = list(tqdm(
            executor.map(lambda p: (p, resolve_place_to_qid(p)), uncached),
            total=len(uncached),
            desc="Resolving QIDs"
        ))
    
    qid_map = {p: qid for p, qid in qid_results if qid}
    if not qid_map:
        return

    values = " ".join(f"wd:{q}" for q in qid_map.values())
    query = f"""
    SELECT ?item ?parent ?parentLabel WHERE {{
        VALUES ?item {{ {values} }}
        ?item wdt:P131* ?parent .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    print(f"Running SPARQL for {len(qid_map)} places...")
    r = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query": query, "format": "json"},
        headers=HEADERS
    )
    if r.status_code != 200:
        print(f"SPARQL error: {r.status_code}")
        return

    from collections import defaultdict
    bindings = r.json()["results"]["bindings"]
    item_parents = defaultdict(list)
    for b in bindings:
        item_qid = b["item"]["value"].split("/")[-1]
        item_parents[item_qid].append(b["parentLabel"]["value"])

    qid_to_place = {v: k for k, v in qid_map.items()}
    for qid, parents in item_parents.items():
        place = qid_to_place.get(qid)
        if place:
            geo_cache[place] = {
                "detected_place": place,
                "wikidata_qid": qid,
                "administrative_hierarchy": parents
            }
    print(f"Cached {len(item_parents)} places")
            
def resolve_place_to_qid(place_name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": place_name,
        "type": "item",
        "limit": 5
    }
    r = requests.get(url, params=params, headers=HEADERS)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if "search" not in data or not data["search"]:
        return None

    qids = [item["id"] for item in data["search"]]
    
    # check all candidates in one SPARQL query
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
    SELECT ?item WHERE {{
        VALUES ?item {{ {values} }}
        ?item wdt:P625 ?coord .
    }}
    """
    try:
        r2 = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers=HEADERS
        )
        if r2.status_code == 200 and r2.text.strip():
            results = r2.json()["results"]["bindings"]
            if results:
                # return first one that has coordinates, preserving search rank order
                found_qids = {r["item"]["value"].split("/")[-1] for r in results}
                for qid in qids:
                    if qid in found_qids:
                        return qid
    except Exception:
        pass

    return qids[0]


def get_administrative_hierarchy_above(qid):

    query = f"""
    SELECT ?parent ?parentLabel
    WHERE {{
      wd:{qid} wdt:P131* ?parent .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

    r = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query": query, "format": "json"},
        headers=HEADERS
    )

    if r.status_code != 200:
        print("SPARQL error:", r.status_code)
        return []

    try:
        data = r.json()
    except Exception:
        print("Non-JSON SPARQL response:", r.text[:200])
        return []

    results = data["results"]["bindings"]

    return [item["parentLabel"]["value"] for item in results]

def get_administrative_hierarchy_below(qid):

    query = f"""
    SELECT ?county ?countyLabel
    WHERE {{
    ?county wdt:P131 wd:{qid} ;
            wdt:P31 wd:Q13212489 .  # instance of county
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

    r = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query": query, "format": "json"},
        headers=HEADERS
    )

    if r.status_code != 200:
        print("SPARQL error:", r.status_code)
        return []

    try:
        data = r.json()
    except Exception:
        print("Non-JSON SPARQL response:", r.text[:200])
        return []

    results = data["results"]["bindings"]

    return [item["countyLabel"]["value"] for item in results]



def extract_state_from_query(query):
    doc = nlp(query)

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text

    return None
def hierarchy_pipeline(place_name, direction):
    qid = resolve_place_to_qid(place_name)
    if not qid:
        return {"error": "Place not found in Wikidata."}
    if direction=="above":
        hierarchy = get_administrative_hierarchy_above(qid)
    elif direction=="below":
        hierarchy = get_administrative_hierarchy_below(qid)

    return {
        "detected_place": place_name,
        "wikidata_qid": qid,
        "administrative_hierarchy": hierarchy
    }
    
def get_wikidata_aliases(qid: str):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    entity = data["entities"][qid]
    
    aliases = []
    for alias in entity.get("aliases", {}).get("en", []):
        aliases.append(alias["value"])
    
    label = entity.get("labels", {}).get("en", {}).get("value", "")
    if label:
        aliases.append(label)
    
    return aliases
    
ALIAS_TO_STATE = {}
for state_name, qid in US_STATE_QIDS.items():
    aliases = get_wikidata_aliases(qid)
    for alias in aliases:
        ALIAS_TO_STATE[alias] = state_name
    ALIAS_TO_STATE[state_name] = state_name

PLACE_ALIASES = {}

def build_place_aliases(place_name):
    if place_name in PLACE_ALIASES:
        return PLACE_ALIASES[place_name]
    result = hierarchy_pipeline(place_name, "above")
    if "error" in result:
        return [place_name]
    qid = result["wikidata_qid"]
    aliases = get_wikidata_aliases(qid)
    PLACE_ALIASES[place_name] = aliases
    # also map every alias back to the same list
    for alias in aliases:
        PLACE_ALIASES[alias] = aliases
    return aliases

def get_scope_variants(scope: str) -> list:
    if not scope:
        return []
    normalized = normalize_geo_scope(scope)
    variants = set()
    variants.add(scope)
    variants.add(normalized)
    variants.add(f"The City Of {normalized}")
    variants.add(f"Greater {normalized}")
    variants.add(f"{normalized}'S")
    variants.add(f"The {normalized}")
    # add wikidata aliases
    aliases = build_place_aliases(normalized)
    variants.update(aliases)
    words = normalized.split()
    if len(words) >= 2:
        short = "".join(w[0] for w in words)  # "LA"
        variants.add(f"{short} County")        # "LA County"
        variants.add(f"{short}")               # "LA"
        # first letters with dots e.g. "L.A."
        abbrev = ".".join(w[0] for w in words) + "."
        variants.add(abbrev)                   # "L.A."
        variants.add(f"{abbrev} County")       # "L.A. County"
        # capitalize first letters joined e.g. "La County"
        short_title = "".join(w[0].upper() + w[1:2].lower() for w in words)
        variants.add(f"{short_title} County")  # "La County"

    return list(variants)
    
regions = ["Midwest", "Southwest", "Northeast", "Southeast", "Western United States"]
# cache for expensive hierarchy lookups


# faster state lookup
US_STATES_SET = {s.lower() for s in us_states}


def normalize_place(place):
    if not place:
        return place
    p = place.strip()
    if p in ALIAS_TO_STATE:
        return ALIAS_TO_STATE[p]
    state = us.states.lookup(p)
    if state:
        return state.name
    try:
        s = statestyle.get(p)
        if s:
            return s.name
    except Exception:
        pass
    result = hierarchy_pipeline(p, "above")
    if "error" not in result and result["administrative_hierarchy"]:
        canonical = result["administrative_hierarchy"][0]
        PLACE_ALIASES[p] = [canonical]
        return canonical
    return p.title()

def get_state_from_path(pdf_path):
    parts = str(pdf_path).split("/")
    for p in parts:
        if p in us_states:
            return p
    return None

def is_us_place(result: dict):
    return "United States" in result.get("administrative_hierarchy", [])

def infer_level(place_name):
    if not place_name:
        return 0
    p = place_name.lower()
    if "united states" in p or p == "us":
        return 0
    elif p in [s.lower() for s in us_states]:
        return 1
    elif place_name in regions or "region" in p or "southern united states" in p:
        return 2
    elif "county" in p:
        return 3
    elif "city" in p:
        return 4
    else:
        return 5

def read_pdf_text(pdf_path, max_chars=3000):
    doc = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= max_chars:
            break
    doc.close()
    return text[:max_chars]

def extract_geo_from_chunk(text, fallback_title="", fallback_scope="United States", fallback_level=0):
    doc_nlp = nlp(text)
    best_scope = None
    best_level = -1
    places = {ent.text for ent in doc_nlp.ents if ent.label_ == "GPE" or ent.text in regions}
    for place in places:
        place = normalize_place(place)
        if place not in geo_cache:
            geo_cache[place] = cached_hierarchy(place)
        result = geo_cache[place]
        if "error" not in result:
            if not is_us_place(result):
                continue
            detected = result["detected_place"]
            level = infer_level(detected)
            if level > best_level:
                best_scope = detected
                best_level = level

    if best_scope:
        return best_scope, best_level

    # fallback to title
    if fallback_title:
        doc_title = nlp(fallback_title)
        for ent in doc_title.ents:
            if ent.label_ == "GPE" or ent.text in regions:
                place = normalize_place(ent.text)
                if place not in geo_cache:
                    geo_cache[place] = cached_hierarchy(place)
                result = geo_cache[place]
                if "error" not in result and is_us_place(result):
                    return result["detected_place"], infer_level(result["detected_place"])

    return fallback_scope, fallback_level

def process_pdf(pdf_file):
    folder_name = pdf_file.parent.name
    geo_scope = folder_name
    geo_level = infer_level(folder_name)

    docs = []
    pdf_doc = fitz.open(str(pdf_file))
    state_from_path = get_state_from_path(pdf_file)
    for page_num, page in enumerate(pdf_doc):
        text = page.get_text()
        if not text:
            continue
        text = " ".join(text.split())
        text = text.encode("ascii", errors="ignore").decode("ascii")
        text = re.sub(r"\S{200,}", " ", text)
        if len(text.strip()) < 20:
            continue
        # tag each page individually
        page_scope, page_level = extract_geo_from_chunk(
            text,
            fallback_scope=geo_scope,
            fallback_level=geo_level
        )
        docs.append(Document(
            text=text,
            metadata={
                "source": pdf_file.name,
                "fileType": "guideline",
                "geo_scope": page_scope,
                "geo_level": page_level,
                "page": page_num,
            }
        ))
    pdf_doc.close()
    return docs

def load_guidelines(guidelines_path):
    pdf_files = list(Path(guidelines_path).rglob("*.pdf"))
    results = [process_pdf(p) for p in tqdm(pdf_files)]
    documents = [doc for result in results for doc in result]
    print(f"Done. {len(documents)} pages from {len(pdf_files)} PDFs")
    return documents

def process_article(article, doc_nlp):
    text = article.get("text", "")
    if not text or len(text.strip()) < 20:
        return None

    title = article.get("title", "")

    text = " ".join(text.split())
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\S{200,}", " ", text)

    geo_scope = "United States"
    geo_level = 0

    title_doc = nlp(title)

    best_scope = None
    best_level = -1

    places = {ent.text for ent in doc_nlp.ents if ent.label_ == "GPE" or ent.label_ == "PRODUCT"}
    title_places = [ent.text for ent in title_doc.ents if ent.label_ == "GPE"]

    for place in places:
        place = normalize_place(place)
        result = geo_cache.get(place)
        if result is None:
            result = cached_hierarchy(place)
            geo_cache[place] = result
        if "error" in result or not is_us_place(result):
            continue
        detected = result["detected_place"]
        level = infer_level(detected)
        if level > best_level:
            best_scope = detected
            best_level = level

    for place in title_places:
        place = normalize_place(place)
        result = geo_cache.get(place)
        if result is None:
            result = cached_hierarchy(place)
            geo_cache[place] = result
        if "error" in result or not is_us_place(result):
            continue
        detected = result["detected_place"]
        level = infer_level(detected)
        if level > best_level:
            best_scope = detected
            best_level = level

    if best_scope and best_level > geo_level:
        geo_scope = best_scope
        geo_level = best_level

    return Document(
        text=text,
        metadata={
            "source": article.get("decoded_url", article.get("title", "")),
            "title": article.get("title", ""),
            "published_date": article.get("published_date", ""),
            "fileType": "news",
            "geo_scope": geo_scope,
            "geo_level": geo_level,
        }
    )

def load_news_json():
    json_file = "documents/News/final_wildfire_articles.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data if isinstance(data, list) else [data]
    print(f"Found {len(articles)} articles")

    texts = [a.get("text", "")[:5000] for a in articles]
    print("Running NLP pipeline...")
    nlp_docs = list(tqdm(nlp.pipe(texts, batch_size=64), total=len(texts), desc="NLP"))
    
    # collect raw place names first WITHOUT normalizing (avoid wikidata calls here)
    all_raw_places = set()
    for doc in nlp_docs:
        for ent in doc.ents:
            if ent.label_ in ("GPE", "PRODUCT"):
                all_raw_places.add(ent.text)

    # normalize only against ALIAS_TO_STATE and us.states (no wikidata)
    all_places = set()
    for p in all_raw_places:
        normalized = p.strip()
        if normalized in ALIAS_TO_STATE:
            normalized = ALIAS_TO_STATE[normalized]
        else:
            state = us.states.lookup(normalized)
            if state:
                normalized = state.name
        all_places.add(normalized)

    # now batch lookup all via wikidata
    uncached = [p for p in all_places if p not in geo_cache]
    print(f"Looking up {len(uncached)} unique places...")
    chunk_size = 50
    for i in tqdm(range(0, len(uncached), chunk_size), desc="Batch hierarchy"):
        batch_hierarchy_lookup(uncached[i:i+chunk_size])

    print("Processing articles...")
    results = [
        process_article(article, doc)
        for article, doc in tqdm(zip(articles, nlp_docs), total=len(articles), desc="Articles")
    ]
    documents = [r for r in results if r is not None]
    print(f"Done. {len(documents)} news articles")
    return documents

def save_geo_cache():
    with open(GEO_CACHE_FILE, "w") as f:
        json.dump(geo_cache, f)