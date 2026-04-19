import os

# ─── Paths ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_DIR, "final dataset for FYP.csv")

# ─── Required columns ─────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "worker_id",
    "primary_skill",
    "worker_latitude",
    "worker_longitude",
    "working_rating_given_to_customer_avg",
    "account_status",
]

# ─── Global State (Kept Empty for Instant Boot) ───────────────────────────────
data = None
_model = None
provider_embeddings = None
bm25 = None
is_loaded = False

def load_engine_if_needed():
    global data, _model, provider_embeddings, bm25, is_loaded
    if is_loaded:
        return
        
    print("[AI] 🚀 First request received! Loading heavy AI libraries now...")
    
    # 🚨 LAZY IMPORTS: We hide these here so the server boots in 0.1 seconds
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    
    print("[AI] Loading dataset...")
    df = pd.read_csv(CSV_PATH, usecols=REQUIRED_COLS, low_memory=False)
    df = df[df["primary_skill"].notna()]
    df = df[df["primary_skill"].astype(str).str.strip() != ""]
    df["worker_latitude"] = pd.to_numeric(df["worker_latitude"], errors="coerce")
    df["worker_longitude"] = pd.to_numeric(df["worker_longitude"], errors="coerce")
    df["working_rating_given_to_customer_avg"] = pd.to_numeric(df["working_rating_given_to_customer_avg"], errors="coerce")
    df.dropna(subset=["worker_latitude", "worker_longitude", "working_rating_given_to_customer_avg"], inplace=True)
    df["working_rating_given_to_customer_avg"] = df["working_rating_given_to_customer_avg"].clip(0, 5)
    df.drop_duplicates(subset=["worker_id"], keep="first", inplace=True)
    df["account_status"] = df["account_status"].astype(str).str.strip()
    df.reset_index(drop=True, inplace=True)
    data = df
    
    print("[AI] Loading AI Models...")
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _descriptions = data["primary_skill"].astype(str).tolist()
    provider_embeddings = _model.encode(_descriptions, show_progress_bar=False, batch_size=64)
    
    _tokenized_corpus = [desc.lower().split() for desc in _descriptions]
    bm25 = BM25Okapi(_tokenized_corpus)
    
    is_loaded = True
    print("[AI] ✅ Model and indexes fully loaded into memory.")


# ─── STEP 3: Recommend function ───────────────────────────────────────────────
def recommend(
    user_query: str,
    user_lat: float,
    user_lng: float,
    radius_km: float = 50.0,
    top_n: int = 5,
) -> list[dict]:
    
    # 1. Trigger the lazy load if this is the first time!
    load_engine_if_needed()
    
    # Lazy import these math functions too
    from sklearn.metrics.pairwise import cosine_similarity
    from geopy.distance import geodesic
    
    user_location = (user_lat, user_lng)
    df = data.copy()

    # ── 1. Semantic similarity ────────────────────────────────────────────────
    query_embedding = _model.encode([user_query])
    semantic_scores = cosine_similarity(query_embedding, provider_embeddings)[0]
    df["semantic_score"] = semantic_scores

    # ── 2. BM25 keyword match ─────────────────────────────────────────────────
    tokenized_query = user_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    df["bm25_score"] = bm25_scores

    # ── 3. RRF hybrid fusion ──────────────────────────────────────────────────
    K = 60  
    df["semantic_rank"] = df["semantic_score"].rank(ascending=False, method="first")
    df["bm25_rank"] = df["bm25_score"].rank(ascending=False, method="first")
    df["hybrid_score"] = (1 / (K + df["semantic_rank"]) + 1 / (K + df["bm25_rank"]))

    # ── 4. Distance calculation + radius filter ───────────────────────────────
    coords = list(zip(df["worker_latitude"], df["worker_longitude"]))
    df["distance_km"] = [geodesic(user_location, wc).km for wc in coords]

    df = df[df["distance_km"] <= radius_km].copy()

    if df.empty:
        return [] 

    # ── 5. Normalised sub-scores ──────────────────────────────────────────────
    df["distance_score"] = 1 / (1 + df["distance_km"])
    df["rating_score"] = df["working_rating_given_to_customer_avg"] / 5.0
    df["availability_score"] = df["account_status"].apply(lambda x: 1.0 if x == "Active" else 0.0)

    # ── 6. Final weighted score ───────────────────────────────────────────────
    df["final_score"] = (
        0.5 * df["hybrid_score"] +
        0.2 * df["rating_score"] +
        0.2 * df["distance_score"] +
        0.1 * df["availability_score"]
    )

    # ── 7. Sort and return top_n ──────────────────────────────────────────────
    top = (
        df.sort_values("final_score", ascending=False)
        .head(top_n)[[
            "worker_id",
            "primary_skill",
            "final_score",
            "distance_km",
            "working_rating_given_to_customer_avg",
        ]]
        .rename(columns={"working_rating_given_to_customer_avg": "rating"})
    )

    top["final_score"] = top["final_score"].round(6)
    top["distance_km"] = top["distance_km"].round(3)
    top["rating"] = top["rating"].round(2)

    return top.to_dict(orient="records")
