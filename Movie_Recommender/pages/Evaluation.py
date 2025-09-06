# pages/02_Evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Homepage import load_navbar

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ---------- Constants ----------
DEFAULT_TOP_K = 12
DEFAULT_POS_THRESHOLD = 4.0
DEFAULT_MIN_OVERLAP = 15
DEFAULT_MAX_USERS = 500
DEFAULT_CF_W = 0.60

# ---------- Tiny CSS polish (cards, chips, KPI, sticky header) ----------
def _load_eval_css():
    st.markdown("""
    <style>
      .sticky { position: sticky; top: 0; z-index: 50; background: linear-gradient(135deg,#1e1e2f,#2e2e4e);
                padding: 10px 12px; margin: -10px -16px 8px; border-bottom: 1px solid rgba(255,255,255,.06); }
      .card { border:1px solid rgba(255,255,255,.10); border-radius:12px; padding:16px; margin-bottom:12px;
              background: rgba(255,255,255,.03); }
      .chip { display:inline-flex; align-items:center; gap:6px; background:rgba(0,212,255,.12); color:#00d4ff;
              padding:6px 10px; border-radius:999px; font-size:.9rem; border:1px solid rgba(0,212,255,.35); }
      .chips { display:flex; gap:8px; flex-wrap:wrap; }
      .metric-grid { display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
      .metric-card{ background: rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.08);
                    border-radius:12px; padding:12px; }
      .metric-card .label{ color:#9aa2b1; font-size:.8rem; margin-bottom:4px }
      .metric-card .value{ font-size:1.25rem; font-weight:700 }
      .bar{ width:100%; height:8px; border-radius:8px; background:rgba(255,255,255,.07); overflow:hidden; margin-top:6px }
      .bar>span{ display:block; height:100%; background:linear-gradient(90deg,#4ECDC4,#45B7D1) }
      .muted { color:#9aa2b1; font-size:.9rem }
    </style>
    """, unsafe_allow_html=True)

def _kpi_card(label:str, value:float):
    v = 0.0 if value is None else float(value)
    pct = max(0.0, min(1.0, v))
    return f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value">{v:.3f}</div>
  <div class="bar"><span style="width:{pct*100:.1f}%"></span></div>
</div>
""".strip()



# ---------- Data loaders ----------
@st.cache_data
def load_ml_small(base_dir: str = "dataset/ml-latest-small"):
    ratings = pd.read_csv(f"{base_dir}/ratings.csv")      # userId, movieId, rating, timestamp
    movies  = pd.read_csv(f"{base_dir}/movies.csv")       # movieId, title, genres
    links   = pd.read_csv(f"{base_dir}/links.csv")        # movieId, imdbId, tmdbId
    links = links[links["tmdbId"].notna() & (links["tmdbId"] > 0)]
    links["tmdbId"] = links["tmdbId"].astype(int)
    return ratings, movies, links

@st.cache_data
def load_tags_data():
    try:
        tags_df = pd.read_csv("dataset/ml-latest-small/tags.csv")
        tags_df['tag'] = tags_df['tag'].astype(str).str.lower().str.strip()
        tags_df = tags_df[tags_df['tag'].str.len() >= 2]
        movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.unique())).reset_index()
        return movie_tags
    except FileNotFoundError:
        return pd.DataFrame(columns=["movieId", "tag"])

# ---------- Content-based preparation ----------
@st.cache_data
def build_movies_for_tfidf(movies_df: pd.DataFrame, movie_tags_df: pd.DataFrame):
    df = movies_df.copy()
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').apply(pd.to_numeric, errors='coerce')
    df['clean_title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    df = df.merge(movie_tags_df, on='movieId', how='left')
    df['tag'] = df['tag'].fillna('')
    df['genres'] = df['genres'].fillna('')
    df['clean_title'] = df['clean_title'].fillna('')
    df['title'] = df['title'].fillna('')
    df['recommendation_blob'] = (
        df['title'] + ' ' + df['clean_title'] + ' ' +
        df['genres'].str.replace('|', ' ') + ' ' + df['tag']
    ).str.lower()
    df['recommendation_blob'] = df['recommendation_blob'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['recommendation_blob'] = df['recommendation_blob'].str.replace(r'\s+', ' ', regex=True)
    return df

@st.cache_resource
def fit_tfidf(reco_blob_series: pd.Series):
    vec = TfidfVectorizer(stop_words='english', max_features=10000, min_df=1, ngram_range=(1,2))
    X = vec.fit_transform(reco_blob_series)
    return vec, X

# ---------- CF helpers ----------
@st.cache_resource
def build_item_user_sparse(ratings: pd.DataFrame):
    user_ids = ratings["userId"].astype(int).unique()
    movie_ids = ratings["movieId"].astype(int).unique()
    user_ids.sort(); movie_ids.sort()
    uid_to_col = {u:i for i,u in enumerate(user_ids)}
    mlid_to_row = {m:i for i,m in enumerate(movie_ids)}
    row_to_mlid = np.array(movie_ids)

    user_means = ratings.groupby("userId")["rating"].mean()
    rows, cols, data = [], [], []
    b_rows, b_cols, b_data = [], [], []

    for _, r in ratings.iterrows():
        u, m, rt = int(r.userId), int(r.movieId), float(r.rating)
        if u in uid_to_col and m in mlid_to_row:
            rows.append(mlid_to_row[m]); cols.append(uid_to_col[u]); data.append(rt - float(user_means.loc[u]))
            b_rows.append(mlid_to_row[m]); b_cols.append(uid_to_col[u]); b_data.append(1.0)

    n_items = len(movie_ids); n_users = len(user_ids)
    R_csr = sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_users))
    B_csr = sp.csr_matrix((b_data, (b_rows, b_cols)), shape=(n_items, n_users))
    norms = np.sqrt(R_csr.multiply(R_csr).sum(axis=1)).A1
    norms[norms == 0.0] = 1e-12
    return R_csr, B_csr, mlid_to_row, row_to_mlid, norms

def recommend_content_seed(seed_ml_movie_id: int, movies_df: pd.DataFrame, tfidf_matrix, top_n: int):
    idx_list = movies_df.index[movies_df['movieId'] == seed_ml_movie_id].tolist()
    if not idx_list: return []
    seed_idx = idx_list[0]
    sims = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).ravel()
    sims[seed_idx] = -np.inf
    top_idx = np.argsort(-sims)[:top_n]
    return [int(movies_df.iloc[i]['movieId']) for i in top_idx]

def cf_recs_fast(seed_ml_movie_id: int, prebuilt, min_overlap: int, top_n: int):
    R_csr, B_csr, mlid_to_row, row_to_mlid, norms = prebuilt
    seed_row = mlid_to_row.get(int(seed_ml_movie_id))
    if seed_row is None: return []
    numer = (R_csr[seed_row] @ R_csr.T).toarray().ravel()
    denom = norms[seed_row] * norms
    sims = np.divide(numer, denom, out=np.zeros_like(numer), where=(denom != 0))
    overlaps = (B_csr[seed_row] @ B_csr.T).toarray().ravel()
    sims[seed_row] = -np.inf
    valid = np.where(overlaps >= min_overlap)[0]
    if valid.size == 0:
        valid = np.argsort(-sims)[:top_n * 5]
    top_idx = valid[np.argsort(-sims[valid])][:top_n]
    return [int(row_to_mlid[i]) for i in top_idx if i != seed_row][:top_n]

def hybrid_recs_fast(seed_ml_movie_id: int,
                     prebuilt,
                     movies_df: pd.DataFrame,
                     tfidf_matrix,
                     min_overlap: int,
                     top_n: int,
                     cf_w: float = 0.6,
                     content_w: float = 0.4):
    rec_c = recommend_content_seed(seed_ml_movie_id, movies_df, tfidf_matrix, top_n=top_n*2)
    rec_f = cf_recs_fast(seed_ml_movie_id, prebuilt, min_overlap=min_overlap, top_n=top_n*2)
    cand = list(set(rec_c) | set(rec_f))
    if not cand: return []

    seed_idx_list = movies_df.index[movies_df['movieId'] == seed_ml_movie_id].tolist()
    if not seed_idx_list: return []
    seed_idx = seed_idx_list[0]

    sims_content = {}
    for mid in cand:
        ridx = movies_df.index[movies_df['movieId'] == mid].tolist()
        if not ridx: continue
        sims_content[mid] = float(cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix[ridx[0]])[0][0])
    if sims_content:
        c_vals = np.array(list(sims_content.values())); c_min, c_max = float(c_vals.min()), float(c_vals.max())
    else:
        c_min = c_max = 0.0

    R_csr, _, mlid_to_row, _, norms = prebuilt
    seed_row = mlid_to_row.get(int(seed_ml_movie_id))
    sims_cf = {}
    if seed_row is not None:
        numer_all = (R_csr[seed_row] @ R_csr.T).toarray().ravel()
        denom_all = norms[seed_row] * norms
        base_cf = np.divide(numer_all, denom_all, out=np.zeros_like(numer_all), where=(denom_all != 0))
        for mid in cand:
            r = mlid_to_row.get(mid)
            if r is not None:
                sims_cf[mid] = float(base_cf[r])
    if sims_cf:
        f_vals = np.array(list(sims_cf.values())); f_min, f_max = float(f_vals.min()), float(f_vals.max())
    else:
        f_min = f_max = 0.0

    scores = []
    for mid in cand:
        c = sims_content.get(mid, 0.0)
        f = sims_cf.get(mid, 0.0)
        c_n = (c - c_min) / (c_max - c_min) if c_max > c_min else (1.0 if c > 0 else 0.0)
        f_n = (f - f_min) / (f_max - f_min) if f_max > f_min else (1.0 if f > 0 else 0.0)
        scores.append((mid, cf_w * f_n + content_w * c_n))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [m for m,_ in scores[:top_n]]

# ---------- Temporal LOO ----------
def temporal_leave_one_out(ratings: pd.DataFrame, pos_threshold: float):
    df = ratings.copy()
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    pos = df[df["rating"] >= pos_threshold].copy()
    if "timestamp" in pos.columns:
        pos = pos.sort_values(["userId", "timestamp"])
    else:
        pos = pos.sort_values(["userId"])
    test_idx = pos.groupby("userId").tail(1).index
    test = df.loc[test_idx, ["userId", "movieId"]]
    train = df.drop(index=test_idx)
    if "timestamp" in train.columns:
        tpos = train[train["rating"] >= pos_threshold].sort_values(["userId", "timestamp"])
    else:
        tpos = train[train["rating"] >= pos_threshold].sort_values(["userId"])
    latest_train = tpos.groupby("userId").tail(1)
    seed_for_user = {int(r.userId): int(r.movieId) for _, r in latest_train.iterrows()}
    test_pairs = [(int(u), int(m)) for u, m in test.itertuples(index=False, name=None)]
    return train, test_pairs, seed_for_user

def _metrics_from_hits(hits: int, users_evald: int, k: int):
    if users_evald == 0 or k <= 0:
        return {"Precision@K": 0.0, "Recall@K": 0.0, "F1@K": 0.0, "HitRate@K": 0.0}
    hit_rate = hits / users_evald
    recall_at_k = hit_rate
    precision_at_k = hits / (users_evald * k)
    f1 = (2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)) if (precision_at_k + recall_at_k) else 0.0
    return {"Precision@K": precision_at_k, "Recall@K": recall_at_k, "F1@K": f1, "HitRate@K": hit_rate}

def evaluate_overall_topN_temporal_LOO(
    ratings: pd.DataFrame,
    movies_df: pd.DataFrame,
    k: int,
    pos_threshold: float,
    min_overlap: int,
    algos: list[str],
    max_users: int,
    seed: int,
    cf_w: float,
    content_w: float,
    progress_cb=None
):
    train, test_pairs, seed_for_user = temporal_leave_one_out(ratings, pos_threshold)

    rng = np.random.default_rng(seed)
    if max_users and len(test_pairs) > max_users:
        sel_idx = rng.choice(len(test_pairs), size=max_users, replace=False)
        test_pairs = [test_pairs[i] for i in sel_idx]

    prebuilt = build_item_user_sparse(train)

    total_work = max(1, len(test_pairs) * max(1, len(algos)))
    done = 0

    def eval_algo(name: str):
        nonlocal done
        hits, users_evald = 0, 0
        for u, heldout_mid in test_pairs:
            seed_mid = seed_for_user.get(u)
            if seed_mid is None:
                done += 1
                if progress_cb: progress_cb(min(1.0, done / total_work))
                continue

            users_evald += 1
            if name == "Content-Based":
                recs = recommend_content_seed(seed_mid, movies_df, tfidf_matrix, top_n=k)
            elif name == "Collaborative":
                recs = cf_recs_fast(seed_mid, prebuilt, min_overlap=min_overlap, top_n=k)
            else:
                recs = hybrid_recs_fast(seed_mid, prebuilt, movies_df, tfidf_matrix,
                                        min_overlap=min_overlap, top_n=k, cf_w=cf_w, content_w=content_w)

            if heldout_mid in (recs or []):
                hits += 1

            done += 1
            if progress_cb: progress_cb(min(1.0, done / total_work))

        return _metrics_from_hits(hits, users_evald, k), {"users": users_evald, "hits": hits}

    out, meta = {}, {}
    for a in algos:
        out[a], meta[a] = eval_algo(a)
    return out, meta

def evaluate_k_sweep(
    ratings: pd.DataFrame,
    movies_df: pd.DataFrame,
    k_min: int, k_max: int, step: int,
    pos_threshold: float, min_overlap: int, algos: list[str],
    max_users: int, seed: int, cf_w: float, content_w: float,
    progress_cb=None
):
    train, test_pairs, seed_for_user = temporal_leave_one_out(ratings, pos_threshold)
    rng = np.random.default_rng(seed)
    if max_users and len(test_pairs) > max_users:
        sel_idx = rng.choice(len(test_pairs), size=max_users, replace=False)
        test_pairs = [test_pairs[i] for i in sel_idx]

    Ks = list(range(k_min, k_max + 1, step))
    Kmax = max(Ks)
    prebuilt = build_item_user_sparse(train)

    def get_recs_for_user(seed_mid, algo):
        if algo == "Content-Based":
            return recommend_content_seed(seed_mid, movies_df, tfidf_matrix, top_n=Kmax)
        elif algo == "Collaborative":
            return cf_recs_fast(seed_mid, prebuilt, min_overlap=min_overlap, top_n=Kmax)
        else:
            return hybrid_recs_fast(seed_mid, prebuilt, movies_df, tfidf_matrix,
                                    min_overlap=min_overlap, top_n=Kmax, cf_w=cf_w, content_w=content_w)

    alg_results = {a: {"hits_at_k": {k:0 for k in Ks}, "users": 0} for a in algos}
    total_work = len(test_pairs) * max(1, len(algos))
    work = 0

    for u, heldout_mid in test_pairs:
        seed_mid = seed_for_user.get(u)
        if seed_mid is None:
            continue
        for a in algos:
            recs = get_recs_for_user(seed_mid, a)
            alg_results[a]["users"] += 1
            if recs:
                for k in Ks:
                    if heldout_mid in recs[:k]:
                        alg_results[a]["hits_at_k"][k] += 1
            work += 1
            if progress_cb: progress_cb(min(1.0, work / total_work))

    out = {a: {} for a in algos}
    for a in algos:
        users = alg_results[a]["users"]
        for k in Ks:
            hits = alg_results[a]["hits_at_k"][k]
            out[a][k] = _metrics_from_hits(hits, users, k)
    return Ks, out

# ---------- RMSE/MAE baseline ----------
def evaluate_rmse_baseline(ratings_df: pd.DataFrame, test_size=0.2, random_state=42):
    df = ratings_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split = int(len(df) * (1 - test_size))
    train = df.iloc[:split]; test = df.iloc[split:]
    mu = float(train["rating"].mean())
    bu = train.groupby("userId")["rating"].mean() - mu
    bi = train.groupby("movieId")["rating"].mean() - mu
    pred = mu + test["userId"].map(bu).fillna(0.0).astype(float) + test["movieId"].map(bi).fillna(0.0).astype(float)
    pred = np.clip(pred.values, 0.5, 5.0)
    true = test["rating"].values.astype(float)
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae  = float(np.mean(np.abs(true - pred)))
    return {"RMSE": rmse, "MAE": mae, "model": "Baseline (μ + bᵤ + bᵢ)"}

# ================== App ==================
load_navbar()
_load_eval_css()

# ---------- Load data & models (cached) ----------
ratings_small, movies_small, links_small = load_ml_small()
movie_tags_df = load_tags_data()
movies_for_tfidf = build_movies_for_tfidf(movies_small, movie_tags_df)
vec, tfidf_matrix = fit_tfidf(movies_for_tfidf["recommendation_blob"])

# Title + sticky summary
st.title("📊 Evaluation")


with st.container():
    st.markdown("<div class='sticky'>", unsafe_allow_html=True)
    st.markdown("**Global settings**", unsafe_allow_html=True)

    # -------- Global settings (expander) --------
    with st.expander("Show/Hide controls", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            K = st.slider("Top-K (Overall)", 5, 30, DEFAULT_TOP_K, 1,
                          help="Used in Overall Top-N evaluation. K-sweep below sets its own K range.")
        with c2:
            pos_threshold = st.slider("Positive rating threshold (≥)", 3.0, 5.0, DEFAULT_POS_THRESHOLD, 0.5,
                                      help="A rating ≥ this is positive for held-out logic.")
        with c3:
            min_overlap = st.slider("CF min co-rater overlap", 5, 50, DEFAULT_MIN_OVERLAP, 1,
                                    help="Minimum co-raters for CF similarity.")
        with c4:
            max_users = st.slider("Max users", 50, 700, DEFAULT_MAX_USERS, 25,
                                  help="Subset users for speed. Larger = slower but more stable.")

        c5, c6 = st.columns([2, 2])
        with c5:
            algos_selected = st.multiselect(
                "Algorithms", ["Content-Based", "Collaborative", "Hybrid"],
                default=["Content-Based", "Collaborative", "Hybrid"]
            )
        with c6:
            seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)

        show_hybrid = "Hybrid" in algos_selected
        cf_w = st.slider("Hybrid: CF weight", 0.0, 1.0, DEFAULT_CF_W, 0.05,
                         help="Only for Hybrid (content weight = 1 - CF).",
                         disabled=not show_hybrid)
        content_w = 1.0 - (cf_w if show_hybrid else DEFAULT_CF_W)

        st.caption("These settings apply to **all** evaluations. K-sweep has its own K range but inherits the rest.")
    # summary chips
    st.markdown(
        f"""
        <div class="chips">
          <span class="chip">Top-K {K}</span>
          <span class="chip">Pos ≥ {pos_threshold:.1f}</span>
          <span class="chip">Overlap ≥ {min_overlap}</span>
          <span class="chip">Users ≤ {max_users}</span>
          <span class="chip">Seed {seed}</span>
          <span class="chip">Algorithms: {", ".join(algos_selected) if algos_selected else "—"}</span>
          {"<span class='chip'>Hybrid: CF "+str(cf_w)+" / Content "+str(round(1-cf_w,2))+"</span>" if show_hybrid else ""}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")

# ----------------- Overall Top-N -----------------
st.subheader("🎯 Overall Top-N Recommendation Quality")
with st.form("overall_form"):
    st.caption("Evaluates whether the user’s last **liked** item can be recovered within the top-K recommendations from the previous liked item.")
    run_overall = st.form_submit_button("Run Overall Evaluation")

if run_overall:
    if not algos_selected:
        st.warning("Select at least one algorithm in Global settings.")
    else:
        with st.spinner("Running Overall evaluation…"):
            prog = st.progress(0.0)
            def _prog(x): prog.progress(x)
            t0 = time.time()
            metrics, meta = evaluate_overall_topN_temporal_LOO(
                ratings=ratings_small,
                movies_df=movies_for_tfidf,
                k=int(K),
                pos_threshold=float(pos_threshold),
                min_overlap=int(min_overlap),
                algos=algos_selected,
                max_users=int(max_users),
                seed=int(seed),
                cf_w=float(cf_w if show_hybrid else DEFAULT_CF_W),
                content_w=float(content_w if show_hybrid else 1.0 - DEFAULT_CF_W),
                progress_cb=_prog
            )
            dur = time.time() - t0
            st.session_state["overall_metrics_eval"] = {
                "metrics": metrics,
                "meta": meta,
                "settings": dict(K=K, pos_threshold=pos_threshold, min_overlap=min_overlap,
                                 max_users=max_users, seed=seed, duration_sec=dur, cf_w=(cf_w if show_hybrid else DEFAULT_CF_W))
            }
            prog.empty()

# Results
if "overall_metrics_eval" in st.session_state and st.session_state["overall_metrics_eval"]:
    payload = st.session_state["overall_metrics_eval"]
    om = payload["metrics"]; meta = payload["meta"]; settings = payload["settings"]; algos = list(om.keys())

    tabs = st.tabs(["Overview", "Charts", "Details"])
    # ---- Overview
    with tabs[0]:
        cols_alg = st.columns(len(algos))
        for i, a in enumerate(algos):
            m = om[a]
            with cols_alg[i]:
                st.markdown(f"### {a}")
                html = "".join([
                    _kpi_card("HitRate@K",    m.get("HitRate@K", 0.0)),
                    _kpi_card("Precision@K",  m.get("Precision@K", 0.0)),
                    _kpi_card("Recall@K",     m.get("Recall@K", 0.0)),
                    _kpi_card("F1@K",         m.get("F1@K", 0.0)),
                ])
                st.markdown(f'<div class="metric-grid">{html}</div>', unsafe_allow_html=True)

        st.markdown("##### Metrics table")
        df = pd.DataFrame(om).T
        st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0), use_container_width=True)
        st.download_button("⬇️ Download CSV", data=df.to_csv(index=True).encode('utf-8'),
                           file_name="overall_topN_metrics.csv", mime="text/csv")

    # ---- Charts
    with tabs[1]:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        metrics_to_plot = ["Precision@K", "Recall@K", "F1@K", "HitRate@K"]
        for i, metric in enumerate(metrics_to_plot):
            vals = [om[a][metric] for a in algos]
            bars = axes[i].bar(algos, vals)
            ymax = max(vals + [0.01]); margin = max(0.01, ymax * 0.18)
            axes[i].set_ylim(0, ymax + margin)
            axes[i].set_title(metric)
            axes[i].tick_params(axis='x', rotation=30)
            for b, v in zip(bars, vals):
                x = b.get_x() + b.get_width() / 2
                axes[i].text(x, v + margin*0.2, f"{v:.3f}", ha="center", va="bottom", fontsize=9, clip_on=False)
        plt.tight_layout()
        st.pyplot(fig)

    # ---- Details
    with tabs[2]:
        meta_df = pd.DataFrame({a: meta[a] for a in algos}).T
        st.dataframe(meta_df, use_container_width=True)

    st.caption(f"⏱️ Completed in {settings['duration_sec']:.2f}s")



st.markdown("---")

# ----------------- K-sweep (container; number inputs) -----------------
st.subheader("📈 K-sweep")

with st.form("ksweep_form"):
    st.caption("Explore how metrics change as K varies. Uses the **Global settings** above for all other parameters.")

    col1, col2, col3 = st.columns(3)

    # K min with stable key
    with col1:
        k_min = st.number_input(
            "K min",
            min_value=3, max_value=30,
            value=st.session_state.get("k_min_val", 5),
            step=1,
            key="k_min_val"
        )

    # --- sanitize k_max in session_state BEFORE rendering its widget ---
    kmax_min_allowed = int(k_min) + 1
    if "k_max_val" not in st.session_state:
        st.session_state["k_max_val"] = 20
    # ensure stored value respects the new min
    if st.session_state["k_max_val"] < kmax_min_allowed:
        st.session_state["k_max_val"] = kmax_min_allowed

    with col2:
        k_max = st.number_input(
            "K max",
            min_value=kmax_min_allowed, max_value=50,
            value=st.session_state["k_max_val"],
            step=1,
            key="k_max_val"
        )

    with col3:
        k_step = st.number_input(
            "K step",
            min_value=1, max_value=10,
            value=st.session_state.get("k_step_val", 2),
            step=1,
            key="k_step_val"
        )

    invalid = (k_max <= k_min) or (k_step <= 0)
    if invalid:
        st.error("Check K range: require K max > K min and step ≥ 1.")

    run_k = st.form_submit_button("Run K-sweep", disabled=invalid)

if 'run_k' in locals() and run_k:
    if not algos_selected:
        st.warning("Pick algorithms in Global settings first.")
    else:
        with st.spinner("Running K-sweep…"):
            prog2 = st.progress(0.0)
            def _p2(x): prog2.progress(x)
            t0 = time.time()
            Ks, sweep = evaluate_k_sweep(
                ratings=ratings_small,
                movies_df=movies_for_tfidf,
                k_min=int(k_min), k_max=int(k_max), step=int(k_step),
                pos_threshold=float(pos_threshold),
                min_overlap=int(min_overlap),
                algos=algos_selected,
                max_users=int(max_users),
                seed=int(seed),
                cf_w=float(cf_w if "Hybrid" in algos_selected else DEFAULT_CF_W),
                content_w=float((1.0 - cf_w) if "Hybrid" in algos_selected else (1.0 - DEFAULT_CF_W)),
                progress_cb=_p2
            )
            dur = time.time() - t0
            st.session_state["k_sweep"] = {"Ks": Ks, "sweep": sweep, "duration_sec": dur}
            prog2.empty()

if "k_sweep" in st.session_state:
    Ks = st.session_state["k_sweep"]["Ks"]
    sweep = st.session_state["k_sweep"]["sweep"]
    dur = st.session_state["k_sweep"]["duration_sec"]

    with st.container():
        tabs_k = st.tabs(["Charts", "Table"])
        with tabs_k[0]:
            metrics_list = ["HitRate@K", "Precision@K", "Recall@K", "F1@K"]
            figc, axes = plt.subplots(2, 2, figsize=(14, 8)); axes = axes.ravel()
            for ax, metric in zip(axes, metrics_list):
                ymax = 0.01
                for a in sweep.keys():
                    ys = [sweep[a][k][metric] for k in Ks]
                    ax.plot(Ks, ys, marker="o", label=a)
                    ymax = max(ymax, max(ys))
                margin = max(0.01, ymax * 0.18)
                ax.set_ylim(0, ymax + margin)
                ax.set_title(metric); ax.set_xlabel("K"); ax.grid(alpha=0.2)
                ax.legend()
            plt.tight_layout()
            st.pyplot(figc)
            st.caption(f"⏱️ K-sweep finished in {dur:.2f}s")

        with tabs_k[1]:
            rows = []
            for a in sweep.keys():
                for k in Ks:
                    row = {"Algorithm": a, "K": k}; row.update(sweep[a][k]); rows.append(row)
            kdf = pd.DataFrame(rows)
            st.dataframe(
                kdf.style.format({"Precision@K": "{:.3f}", "Recall@K": "{:.3f}",
                                  "F1@K": "{:.3f}", "HitRate@K": "{:.3f}"}),
                use_container_width=True
            )
            st.download_button("⬇️ Download K-sweep CSV",
                               data=kdf.to_csv(index=False).encode("utf-8"),
                               file_name="k_sweep_metrics.csv",
                               mime="text/csv")


st.markdown("---")

# ----------------- RMSE / MAE -----------------
st.subheader("📏 Rating Prediction Error")
with st.form("rmse_form"):
    colL, _ = st.columns([1, 2])
    with colL:
        test_size = st.selectbox("Test size", [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50], index=2,
                                 help="Hold-out share for baseline error. Uses Global seed.")
        run_rmse = st.form_submit_button("Run RMSE / MAE")

if 'run_rmse' in locals() and run_rmse:
    with st.spinner("Computing baseline error…"):
        res = evaluate_rmse_baseline(ratings_small, test_size=float(test_size), random_state=int(seed))
        st.session_state["rmse_res_eval"] = res

if "rmse_res_eval" in st.session_state:
    r = st.session_state["rmse_res_eval"]
    with st.container():
        # KPI cards
        html = "".join([_kpi_card("RMSE", r["RMSE"]), _kpi_card("MAE", r["MAE"])])
        st.markdown(f'<div class="metric-grid">{html}</div>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        bars = ax2.bar(["RMSE", "MAE"], [r["RMSE"], r["MAE"]])
        ymax = max([r["RMSE"], r["MAE"]]); margin = max(0.01, ymax * 0.18)
        ax2.set_title("Rating Prediction Error"); ax2.set_ylim(0, ymax + margin)
        for b, v in zip(bars, [r["RMSE"], r["MAE"]]):
            ax2.text(b.get_x() + b.get_width()/2, v + margin*0.2, f"{v:.4f}",
                     ha="center", va="bottom", clip_on=False)
        plt.tight_layout()
        st.pyplot(fig2)

