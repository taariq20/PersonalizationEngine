# app.py
import streamlit as st
import joblib, sqlite3, random, pandas as pd, numpy as np, os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title='Personalization Engine', page_icon='🎬', layout='centered')

# ── PyTorch model classes — must be defined before any torch.load ─
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    # ── NCF: Wide & Deep ─────────────────────────────────────────
    class AttentionScorer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.W_q   = nn.Linear(dim, dim, bias=False)
            self.W_k   = nn.Linear(dim, dim, bias=False)
            self.scale = dim ** 0.5

        def forward(self, user_emb, movie_emb):
            q = self.W_q(user_emb)
            k = self.W_k(movie_emb)
            return (q * k).sum(dim=-1) / self.scale

    class WideAndDeepNCF(nn.Module):
        def __init__(self, n_users, n_movies, n_genres,
                     embed_dim=64, mlp_layers=(256, 128, 64), dropout=0.3):
            super().__init__()
            self.user_emb    = nn.Embedding(n_users,  embed_dim)
            self.movie_emb   = nn.Embedding(n_movies, embed_dim)
            self.user_bias   = nn.Embedding(n_users,  1)
            self.movie_bias  = nn.Embedding(n_movies, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))

            layers, in_dim = [], embed_dim * 2
            for out_dim in mlp_layers:
                layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
                in_dim = out_dim
            self.deep_mlp    = nn.Sequential(*layers)
            self.deep_out    = nn.Linear(mlp_layers[-1], 1)
            self.wide_linear = nn.Linear(n_genres, 1, bias=True)
            self.attention   = AttentionScorer(embed_dim)
            self.alpha       = nn.Parameter(torch.tensor(0.5))

        def forward(self, user, movie, genres, return_attention=False):
            u  = self.user_emb(user)
            m  = self.movie_emb(movie)
            ub = self.user_bias(user).squeeze(1)
            mb = self.movie_bias(movie).squeeze(1)
            deep_score = self.deep_out(
                self.deep_mlp(torch.cat([u, m], dim=1))
            ).squeeze(1)
            wide_score = self.wide_linear(genres).squeeze(1)
            attn       = self.attention(u, m)
            a      = torch.sigmoid(self.alpha)
            raw    = a * deep_score + (1 - a) * wide_score + ub + mb + self.global_bias
            rating = torch.sigmoid(raw) * 4 + 1
            cosine_target = (
                nn.functional.cosine_similarity(u.detach(), m.detach(), dim=1) + 1
            ) / 2
            if return_attention:
                return rating, attn
            return rating, cosine_target, attn

    # ── BERT4Rec: Transformer sequence model ─────────────────────
    class BERT4RecMax(nn.Module):
        """
        Pre-LayerNorm BERT4Rec with weight-tied output embeddings.
        Architecture: hidden=256, layers=2, heads=2, ffn=512, dropout=0.2
        Trained with Cloze task + last-item fine-tuning on MovieLens 1M.
        Val HR@10 ~0.60 | Test HR@10 ~0.55 | NDCG@10 ~0.34
        """
        def __init__(self, vocab_size, max_seq_len, hidden_dim,
                     n_layers, n_heads, ffn_dim, dropout, pad_token=0):
            super().__init__()
            self.pad_token  = pad_token
            self.hidden_dim = hidden_dim

            self.item_emb    = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token)
            self.pos_emb     = nn.Embedding(max_seq_len, hidden_dim)
            self.emb_norm    = nn.LayerNorm(hidden_dim)
            self.emb_dropout = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model         = hidden_dim,
                nhead           = n_heads,
                dim_feedforward = ffn_dim,
                dropout         = dropout,
                activation      = 'gelu',
                batch_first     = True,
                norm_first      = True,  # Pre-LN for stable deep-stack training
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.padding_idx is not None:
                        m.weight.data[m.padding_idx].zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, input_ids):
            B, L = input_ids.shape
            pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
            x    = self.item_emb(input_ids) + self.pos_emb(pos)
            x    = self.emb_dropout(self.emb_norm(x))
            pad_mask = (input_ids == self.pad_token)
            x    = self.transformer(x, src_key_padding_mask=pad_mask)
            # Weight-tied output: logits = LN(x) @ E^T + bias
            x_norm = self.output_norm(x)
            logits = x_norm @ self.item_emb.weight.T + self.output_bias
            return logits  # (B, L, vocab_size)

    TORCH_AVAILABLE = True
except ImportError:
    pass

# ── Load models & data ───────────────────────────────────────────
@st.cache_resource
def load_models():
    best_svd = joblib.load('models/best_svd.pkl')
    movies   = pd.read_csv('movies.csv')
    ratings  = pd.read_csv('ratings.csv')

    movies['genre_list'] = movies['genres'].str.split('|')
    mlb          = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genre_list'])
    genre_df     = pd.DataFrame(genre_matrix, index=movies['movieId'], columns=mlb.classes_)
    movie_sim_df = pd.DataFrame(
        cosine_similarity(genre_df),
        index=genre_df.index,
        columns=genre_df.index
    )
    return best_svd, movies, ratings, movie_sim_df

@st.cache_resource
def load_ncf():
    """
    Load the Wide & Deep NCF checkpoint saved from Colab.
    Returns (model, user_enc, movie_enc, genre_matrix_ncf, pop_array, all_genres, device)
    or None if the checkpoint is not found or PyTorch is unavailable.

    WHAT TO PUT IN models/:
        models/ncf_model_checkpoint_v4.pt   ← the file saved by CELL 16 in the notebook
    """
    if not TORCH_AVAILABLE:
        st.sidebar.info("💡 Install `torch` to enable NCF variant.")
        return None

    checkpoint_path = 'models/ncf_model_checkpoint_v4.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'models/ncf_model_checkpoint_v4.pkt'

    if not os.path.exists(checkpoint_path):
        # Only show error if torch is actually available
        st.sidebar.warning(f"NCF model file not found at {checkpoint_path}")
        return None

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg    = ckpt['config']

        ncf_model = WideAndDeepNCF(
            n_users    = cfg['n_users'],
            n_movies   = cfg['n_movies'],
            n_genres   = cfg['n_genres'],
            embed_dim  = cfg['embed_dim'],
            mlp_layers = cfg['mlp_layers'],
            dropout    = cfg['dropout'],
        ).to(device)

        ncf_model.load_state_dict(ckpt['model_state_dict'])
        ncf_model.eval()

        user_enc        = ckpt['encoders']['user_enc']
        movie_enc       = ckpt['encoders']['movie_enc']
        genre_matrix_ncf = ckpt['genre_matrix']   # shape (n_movies, n_genres)
        pop_array       = ckpt['pop_array']        # shape (n_movies,) normalised [0,1]
        all_genres      = ckpt['all_genres']

        return ncf_model, user_enc, movie_enc, genre_matrix_ncf, pop_array, all_genres, device
    except Exception as e:
        st.warning(f"NCF checkpoint could not be loaded ({e}). NCF variant will fall back to SVD.")
        return None
    
@st.cache_resource
def load_bert4rec():
    """
    Load BERT4Rec MAX checkpoint.
    File: models/bert4rec_max_checkpoint.pt  (saved by BERT4Rec notebook Cell 30)

    Checkpoint structure:
        model_state_dict, config, encoders, sequences,
        best_val_hr10, history, test_results, prec_at_10, rec_at_10

    config keys:
        vocab_size, max_seq_len, hidden_dim, n_layers, n_heads,
        ffn_dim, dropout, pad_token, mask_token, n_movies

    sequences keys:
        train_seqs    — dict: encoded_user_idx (int) -> list of 1-based movie tokens
        val_targets, test_targets

    NOTE: BERT4Rec tokens are 1-based (0=PAD, 1..N=movies, N+1=MASK).
          To get the original movieId: movie_enc.classes_[token - 1]
    """
    if not TORCH_AVAILABLE:
        return None
    checkpoint_path = 'models/bert4rec_max_checkpoint.pt'
    if not os.path.exists(checkpoint_path):
        return None
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg    = ckpt['config']
        bert_model = BERT4RecMax(
            vocab_size  = cfg['vocab_size'],
            max_seq_len = cfg['max_seq_len'],
            hidden_dim  = cfg['hidden_dim'],
            n_layers    = cfg['n_layers'],
            n_heads     = cfg['n_heads'],
            ffn_dim     = cfg['ffn_dim'],
            dropout     = cfg['dropout'],
            pad_token   = cfg['pad_token'],
        ).to(device)
        bert_model.load_state_dict(ckpt['model_state_dict'])
        bert_model.eval()
        return (
            bert_model,
            ckpt['encoders']['user_enc'],    # original userId  -> encoded int
            ckpt['encoders']['movie_enc'],   # original movieId -> 0-based int
                                             # BERT token = 0-based + 1
            ckpt['sequences']['train_seqs'], # encoded_uid -> [1-based token, ...]
            cfg['mask_token'],
            cfg['max_seq_len'],
            cfg['n_movies'],
            device,
        )
    except Exception as e:
        st.sidebar.warning(f"BERT4Rec load failed: {e}")
        return None


best_svd, movies, ratings, movie_sim_df = load_models()
ncf_bundle  = load_ncf()
bert_bundle = load_bert4rec()
NCF_LOADED  = ncf_bundle  is not None
BERT_LOADED = bert_bundle is not None

ALL_GENRES = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]

# Variants available in the A/B test
VARIANTS = ['collaborative', 'content']
if NCF_LOADED:   VARIANTS.append('ncf')
if BERT_LOADED:  VARIANTS.append('bert4rec')

VARIANT_LABELS = {
    'collaborative': 'SVD Collab',
    'content':       'Content',
    'ncf':           'NCF Neural',
    'bert4rec':      'BERT4Rec',
}

# ── Database setup ───────────────────────────────────────────────
def init_db():
    con = sqlite3.connect('logs.db')
    con.execute('''CREATE TABLE IF NOT EXISTS events (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   TEXT,
        variant   TEXT,
        movie_id  INTEGER,
        event     TEXT,
        ts        DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    con.commit()
    con.close()

init_db()

def log_event(user_id, variant, movie_id, event):
    con = sqlite3.connect('logs.db')
    con.execute(
        'INSERT INTO events (user_id, variant, movie_id, event) VALUES (?,?,?,?)',
        (user_id, variant, movie_id, event)
    )
    con.commit()
    con.close()

def get_seen_movies(user_id):
    con = sqlite3.connect('logs.db')
    try:
        df = pd.read_sql(
            "SELECT movie_id FROM events WHERE user_id=? AND event IN ('like','dislike')",
            con, params=(str(user_id),)
        )
    except:
        df = pd.DataFrame(columns=['movie_id'])
    con.close()
    return df['movie_id'].tolist()

def is_cold_start(user_id):
    con   = sqlite3.connect('logs.db')
    cur   = con.execute(
        "SELECT COUNT(*) FROM events WHERE user_id=? AND event IN ('like','dislike')",
        (str(user_id),)
    )
    count = cur.fetchone()[0]
    con.close()
    return count < 5

# ── Explanation helpers ──────────────────────────────────────────
def explain_svd(user_id, recommended_movie_id):
    uid = int(user_id)

    # Find users who rated this movie highly
    similar_raters = ratings[
        (ratings['movieId'] == recommended_movie_id) &
        (ratings['rating'] >= 4)
    ]['userId'].tolist()

    # Find overlap — users who also rated the same movies as this user highly
    user_liked = set(ratings[(ratings['userId'] == uid) &
                              (ratings['rating'] >= 4)]['movieId'].tolist())

    best_shared = 0
    for other_uid in similar_raters[:100]:
        other_liked = set(ratings[(ratings['userId'] == other_uid) &
                                   (ratings['rating'] >= 4)]['movieId'].tolist())
        shared = len(user_liked & other_liked)
        if shared > best_shared:
            best_shared = shared

    movie_row  = movies[movies['movieId'] == recommended_movie_id].iloc[0]
    avg_rating = ratings[ratings['movieId'] == recommended_movie_id]['rating'].mean()
    n_raters   = len(similar_raters)

    if best_shared > 0:
        return (f"Users with {best_shared} movies in common with you "
                f"rated this {avg_rating:.1f}⭐ ({n_raters:,} similar users liked this)")
    return f"Highly rated by users who share your taste ({avg_rating:.1f}⭐, {n_raters:,} ratings)"

def explain_content(user_id, recommended_movie_id):
    uid   = int(user_id)
    liked = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4)]['movieId'].tolist()
    if not liked:
        return "Matches your genre preferences"
    rec_genres   = set(movies[movies['movieId'] == recommended_movie_id].iloc[0]['genre_list'])
    best_match, best_overlap = None, 0
    for mid in liked:
        row = movies[movies['movieId'] == mid]
        if row.empty:
            continue
        overlap = len(rec_genres & set(row.iloc[0]['genre_list']))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match   = row.iloc[0]['title']
    return f"Because you liked {best_match}" if best_match else "Matches your genre preferences"

# ── NCF helpers ──────────────────────────────────────────────────

def _ncf_candidates(user_idx_encoded, n_movies_ncf, seen_movie_indices):
    """Return array of candidate movie indices (encoded) not yet seen."""
    mask = np.ones(n_movies_ncf, dtype=bool)
    for s in seen_movie_indices:
        if s < n_movies_ncf:
            mask[int(s)] = False
    return np.where(mask)[0]


def get_ncf_recs(user_original_id, n=10):
    """
    NCF recommendations for an existing user.
    user_original_id is the raw userId from ratings.csv (e.g. 1..6040).
    Returns list of dicts identical in shape to get_svd_recs / get_content_recs.
    Falls back to SVD if anything goes wrong.
    """
    if not NCF_LOADED:
        return get_svd_recs(user_original_id, n)

    ncf_model, user_enc, movie_enc, genre_matrix_ncf, pop_array, all_genres, device = ncf_bundle

    # Check if user exists in the training encoders
    if user_original_id not in user_enc.classes_:
        # Unknown user — fall back to content filtering
        return get_content_recs(user_original_id, n)

    try:
        user_idx = int(user_enc.transform([user_original_id])[0])
        n_movies_ncf = len(movie_enc.classes_)

        # Seen in training data + in-session interactions
        seen_encoded = set(
            ratings[ratings['userId'] == int(user_original_id)]['movieId']
            .map(lambda mid: int(movie_enc.transform([mid])[0])
                 if mid in movie_enc.classes_ else -1)
            .pipe(lambda s: s[s >= 0])
            .values
        )
        # Also exclude movies seen in this session
        seen_movie_ids = get_seen_movies(user_original_id)
        for mid in seen_movie_ids:
            if mid in movie_enc.classes_:
                seen_encoded.add(int(movie_enc.transform([mid])[0]))

        candidates = _ncf_candidates(user_idx, n_movies_ncf, seen_encoded)
        if len(candidates) == 0:
            return get_svd_recs(user_original_id, n)

        c_t = torch.LongTensor(candidates).to(device)
        u_t = torch.LongTensor([user_idx] * len(candidates)).to(device)
        g_t = torch.FloatTensor(genre_matrix_ncf[candidates]).to(device)

        with torch.no_grad():
            preds, attn_logits = ncf_model(u_t, c_t, g_t, return_attention=True)

        preds_np  = preds.cpu().numpy()
        attns_np  = torch.sigmoid(attn_logits).cpu().numpy()
        top_idx   = np.argsort(preds_np)[::-1][:n]
        wide_w    = ncf_model.wide_linear.weight.squeeze().detach().cpu().numpy()

        results = []
        for i in top_idx:
            midx      = candidates[i]
            orig_id   = movie_enc.classes_[midx]
            row       = movies[movies['movieId'] == orig_id]
            if row.empty:
                continue
            row = row.iloc[0]
            gv        = genre_matrix_ncf[midx]
            top_genre_idx = np.argsort(gv * wide_w)[::-1]
            top_genre = next(
                (all_genres[j] for j in top_genre_idx if gv[j] > 0), 'various genres'
            )
            results.append({
                'id':               int(orig_id),
                'title':            row['title'],
                'genres':           row['genres'],
                'predicted_rating': round(float(preds_np[i]), 2),
                'attention_score':  round(float(attns_np[i]), 3),
                'explanation':      f"NCF: aligns with your taste for {top_genre}",
            })
        return results

    except Exception as e:
        st.warning(f"NCF inference error: {e}. Falling back to SVD.")
        return get_svd_recs(user_original_id, n)


def get_ncf_cold_start_recs(preferred_genres, n=10):
    """
    Cold-start using NCF's genre_matrix + pop_array (cosine-sim method).
    This is the CORRECT method from the notebook — does NOT use wide_linear weights
    because they can be negative for some genres.
    """
    if not NCF_LOADED:
        return get_cold_start_recs(preferred_genres, n)

    _, user_enc, movie_enc, genre_matrix_ncf, pop_array, all_genres, _ = ncf_bundle

    POP_WEIGHT = 0.15

    pref_set = set(preferred_genres)
    pref_vec = np.array(
        [1.0 if g in pref_set else 0.0 for g in all_genres],
        dtype=np.float32
    )
    pref_norm = np.linalg.norm(pref_vec)
    if pref_norm == 0:
        return get_cold_start_recs(preferred_genres, n)

    scores = []
    n_movies_ncf = len(movie_enc.classes_)
    for midx in range(n_movies_ncf):
        gv      = genre_matrix_ncf[midx]
        gv_norm = np.linalg.norm(gv)
        if gv_norm == 0:
            continue
        cosine_sim  = float(np.dot(pref_vec, gv) / (pref_norm * gv_norm))
        if cosine_sim <= 0:
            continue
        final_score = cosine_sim + POP_WEIGHT * float(pop_array[midx])
        scores.append((midx, cosine_sim, final_score))

    if not scores:
        return get_cold_start_recs(preferred_genres, n)

    scores.sort(key=lambda x: x[2], reverse=True)

    results = []
    for midx, cosine_sim, final_score in scores[:n]:
        orig_id = movie_enc.classes_[midx]
        row     = movies[movies['movieId'] == orig_id]
        if row.empty:
            continue
        row = row.iloc[0]
        results.append({
            'id':          int(orig_id),
            'title':       row['title'],
            'genres':      row['genres'],
            'explanation': f"NCF genre match ({cosine_sim:.2f}) — {', '.join(preferred_genres)}",
        })
    return results

# ── BERT4Rec helpers ──────────────────────────────────────────────
def _bert_pad_or_truncate(seq, max_len, pad_val=0):
    """Keep the most-recent max_len items, right-pad with pad_val."""
    if len(seq) > max_len:
        return seq[-max_len:]
    return seq + [pad_val] * (max_len - len(seq))

def _bert_candidates(uid, train_seqs, n_movies):
    """
    Return array of unseen 1-based movie token indices for this user.
    BERT4Rec token space: 0=PAD, 1..n_movies=items, n_movies+1=MASK
    """
    all_items = np.arange(1, n_movies + 1, dtype=np.int64)
    seen      = set(train_seqs.get(uid, []))
    mask      = np.ones(n_movies, dtype=bool)
    for s in seen:
        if 1 <= s <= n_movies:
            mask[s - 1] = False
    return all_items[mask]

def get_bert4rec_recs(user_original_id, n=10):
    """
    BERT4Rec sequence-based next-item recommendations.

    How it works:
      1. Takes the user's chronological watch history (from train_seqs in the checkpoint)
      2. Appends [MASK] token at the end
      3. Runs through the Transformer — gets logits for the masked position
      4. Ranks all unseen movies by those logits and returns top-N

    Falls back to content filtering for users not in training data,
    and to SVD if inference fails.
    """
    if not BERT_LOADED:
        return get_svd_recs(user_original_id, n)

    bert_model, user_enc, movie_enc, train_seqs, mask_token, max_seq_len, n_movies, device = bert_bundle

    if user_original_id not in user_enc.classes_:
        return get_content_recs(user_original_id, n)

    try:
        uid = int(user_enc.transform([user_original_id])[0])
        if uid not in train_seqs:
            return get_svd_recs(user_original_id, n)

        # Build input: history (truncated to max_seq_len - 1) + [MASK]
        hist = _bert_pad_or_truncate(train_seqs[uid], max_seq_len - 1, 0)
        inp  = torch.LongTensor(hist + [mask_token]).unsqueeze(0).to(device)

        with torch.no_grad():
            # logits: (1, seq_len, vocab_size) — we only need the last position
            mask_logits = bert_model(inp)[0, -1, :].cpu().numpy()

        # Get unseen candidates (1-based tokens)
        candidates = _bert_candidates(uid, train_seqs, n_movies)

        # Also exclude in-session interactions
        session_tokens = set()
        for mid in get_seen_movies(user_original_id):
            if mid in movie_enc.classes_:
                session_tokens.add(int(movie_enc.transform([mid])[0]) + 1)
        if session_tokens:
            candidates = candidates[~np.isin(candidates, list(session_tokens))]

        if len(candidates) == 0:
            return get_svd_recs(user_original_id, n)

        cand_scores = mask_logits[candidates]
        top_idx     = np.argsort(cand_scores)[::-1][:n]

        results = []
        for i in top_idx:
            midx    = int(candidates[i])
            # BERT token is 1-based; movie_enc.classes_ is 0-based
            orig_id = movie_enc.classes_[midx - 1]
            row     = movies[movies['movieId'] == orig_id]
            if row.empty: continue
            row = row.iloc[0]
            results.append({
                'id':             int(orig_id),
                'title':          row['title'],
                'genres':         row['genres'],
                'bert4rec_score': round(float(cand_scores[i]), 4),
                'explanation':    'BERT4Rec: predicted next based on your watch sequence',
            })
        return results

    except Exception as e:
        st.warning(f"BERT4Rec inference error: {e}. Falling back to SVD.")
        return get_svd_recs(user_original_id, n)

def get_bert4rec_cold_start_recs(preferred_genres, n=10):
    """
    BERT4Rec has no sequence for a brand-new user — cold-start is not
    sequence-based. We reuse the NCF cosine-genre method (same genre data)
    and just re-label the explanation.
    """
    if NCF_LOADED:
        recs = get_ncf_cold_start_recs(preferred_genres, n)
        for r in recs:
            r['explanation'] = r['explanation'].replace('NCF genre match', 'BERT4Rec cold-start')
        return recs
    return get_cold_start_recs(preferred_genres, n)

# ── Recommendation functions ─────────────────────────────────────
def get_svd_recs(user_id, n=10):
    rated   = ratings[ratings['userId'] == int(user_id)]['movieId'].tolist()
    seen    = get_seen_movies(user_id)
    unrated = [m for m in movies['movieId'].tolist() if m not in rated and m not in seen]
    preds   = [best_svd.predict(int(user_id), mid) for mid in unrated]
    preds.sort(key=lambda x: x.est, reverse=True)
    results = []
    for pred in preds[:n]:
        row = movies[movies['movieId'] == pred.iid].iloc[0]
        results.append({'id': int(pred.iid), 'title': row['title'], 'genres': row['genres']})
    return results

def get_content_recs(user_id, n=10):
    uid   = int(user_id)
    liked = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4)]['movieId'].tolist()
    if not liked:
        liked = ratings[ratings['userId'] == uid]['movieId'].tolist()
    rated      = ratings[ratings['userId'] == uid]['movieId'].tolist()
    seen       = get_seen_movies(user_id)
    sim_scores = movie_sim_df[liked].mean(axis=1)
    sim_scores = sim_scores.drop(index=[m for m in rated + seen if m in sim_scores.index], errors='ignore')
    top_movies = sim_scores.nlargest(n).index.tolist()
    results = []
    for mid in top_movies:
        row = movies[movies['movieId'] == mid].iloc[0]
        results.append({'id': int(mid), 'title': row['title'], 'genres': row['genres']})
    return results

def get_cold_start_recs(preferred_genres, n=10):
    popularity = ratings.groupby('movieId').size().reset_index(name='count')
    popular    = popularity.sort_values('count', ascending=False).merge(movies, on='movieId')
    mask       = popular['genres'].apply(
        lambda g: any(genre in g.split('|') for genre in preferred_genres)
    )
    results = popular[mask].head(n)
    return [
        {
            'id':          int(row['movieId']),
            'title':       row['title'],
            'genres':      row['genres'],
            'explanation': f"Popular in {', '.join(preferred_genres)}"
        }
        for _, row in results.iterrows()
    ]

# ── Session state init ───────────────────────────────────────────
if 'user_id'  not in st.session_state: st.session_state.user_id  = None
if 'variant'  not in st.session_state: st.session_state.variant  = None
if 'user_type' not in st.session_state: st.session_state.user_type = None
if 'preferred_genres' not in st.session_state: st.session_state.preferred_genres = []
if 'page'     not in st.session_state: st.session_state.page     = 'landing'
if 'liked_movies' not in st.session_state: st.session_state.liked_movies = []

# ── Pages ────────────────────────────────────────────────────────

def landing_page():
    st.title('🎬 Personalization Engine')
    active_models = []
    if NCF_LOADED:   active_models.append('NCF')
    if BERT_LOADED:  active_models.append('BERT4Rec')
    if active_models:
        st.caption(f"✅ {' + '.join(active_models)} loaded — {len(VARIANTS)}-way A/B test active.")
    else:
        st.caption(
            '⚠️ No neural checkpoints found in `models/`. '
            'Running SVD vs Content only. '
            'Add `ncf_model_checkpoint_v4.pt` and/or `bert4rec_max_checkpoint.pt` to enable them.'
        )
    st.write('How do you want to start?')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('👤 New User\n\nStart from scratch — pick your favourite genres',
                     use_container_width=True):
            st.session_state.user_id   = str(random.randint(10000, 99999))
            st.session_state.variant   = random.choice(VARIANTS)
            st.session_state.user_type = 'new'
            st.session_state.page      = 'survey'
            st.rerun()
    with col2:
        if st.button('🎬 Existing User\n\nUse a MovieLens user\'s history',
                     use_container_width=True):
            st.session_state.user_id   = str(random.randint(1, 6040))
            st.session_state.variant   = random.choice(VARIANTS)
            st.session_state.user_type = 'existing'
            st.session_state.page      = 'home'
            st.rerun()


def survey_page():
    st.title('👋 Welcome!')
    st.write('Pick your favourite genres to get started:')
    selected = []
    cols = st.columns(3)
    for i, genre in enumerate(ALL_GENRES):
        if cols[i % 3].checkbox(genre):
            selected.append(genre)
    if st.button('Get Recommendations →', type='primary'):
        if not selected:
            st.error('Please select at least one genre!')
        else:
            st.session_state.preferred_genres = selected
            st.session_state.page             = 'home'
            st.rerun()

def _next_variant(current):
    """Rotate through available variants on like/dislike."""
    idx = VARIANTS.index(current) if current in VARIANTS else 0
    return VARIANTS[(idx + 1) % len(VARIANTS)]

def home_page():
    user_id   = st.session_state.user_id
    variant   = st.session_state.variant
    user_type = st.session_state.user_type

    st.title('🎬 Movie Recommendations')
    col1, col2, col3 = st.columns([2, 2, 1])
    col1.metric('User ID', user_id)
    col2.metric('Variant', VARIANT_LABELS.get(variant, variant))    
    if col3.button('Reset'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Cold start check
    if user_type == 'new' and is_cold_start(user_id):
        preferred = st.session_state.preferred_genres
        if not preferred:
            st.session_state.page = 'survey'
            st.rerun()
        st.info('🌱 Based on your genre preferences — like/dislike movies to get personalised recommendations after 5 interactions!')
        
        if variant == 'bert4rec' and BERT_LOADED:
            recs = get_bert4rec_cold_start_recs(preferred)
        elif variant == 'ncf' and NCF_LOADED:
            recs = get_ncf_cold_start_recs(preferred)
        else:
            recs = get_cold_start_recs(preferred)
        for rec in recs:
            log_event(user_id, variant, rec['id'], 'impression')

    # ── Warm-start path ──────────────────────────────────────────
    else:
        if variant == 'collaborative':
            recs = get_svd_recs(user_id)
            for rec in recs:
                rec['explanation'] = explain_svd(user_id, rec['id'])
        
        elif variant == 'content':
            recs = get_content_recs(user_id)
            for rec in recs:
                rec['explanation'] = explain_content(user_id, rec['id'])

        elif variant == 'ncf':
            # get_ncf_recs already populates 'explanation' and handles fallback
            recs = get_ncf_recs(int(user_id) if user_id.isdigit() else user_id)
            if not recs:
                # Final safety net
                recs = get_svd_recs(user_id)
                for rec in recs:
                    rec['explanation'] = explain_svd(user_id, rec['id'])
        
        elif variant == 'bert4rec':
            recs = get_bert4rec_recs(int(user_id) if user_id.isdigit() else user_id)
            if not recs:
                recs = get_svd_recs(user_id)
                for rec in recs: rec['explanation'] = explain_svd(user_id, rec['id'])

        for rec in recs:
            log_event(user_id, variant, rec['id'], 'impression')

    # Render cards
    for rec in recs:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{rec['title']}**")
                st.caption(rec['genres'])
                if rec.get('explanation'):
                    st.caption(f"💡 {rec['explanation']}")
                # Show NCF-specific scores when available
                if variant == 'ncf' and rec.get('predicted_rating'):
                    st.caption(
                        f"🧠 Predicted rating: {rec['predicted_rating']:.2f} · "
                        f"Attention: {rec.get('attention_score', 0):.3f}"
                    )
                if variant == 'bert4rec' and rec.get('bert4rec_score') is not None:
                    st.caption(f"🤖 BERT4Rec score: {rec['bert4rec_score']:.4f}")
            with col2:
                col_a, col_b = st.columns(2)
                if col_a.button('👍', key=f"like_{rec['id']}"):
                    log_event(user_id, variant, rec['id'], 'like')
                    st.session_state.liked_movies.append({
                        'title':   rec['title'],
                        'genres':  rec['genres'],
                        'variant': variant
                    })
                    st.session_state.variant = _next_variant(variant)
                    st.toast(f"Liked! Switching to {st.session_state.variant}")
                    st.rerun()
                if col_b.button('👎', key=f"dislike_{rec['id']}"):
                    log_event(user_id, variant, rec['id'], 'dislike')
                    st.session_state.variant = _next_variant(variant)
                    st.toast(f"Disliked! Switching to {st.session_state.variant}")
                    st.rerun()               

def results_page():
    st.title('📊 Interleaved Test Results')
    con = sqlite3.connect('logs.db')
    try:
        df = pd.read_sql('SELECT * FROM events', con)
    except:
        df = pd.DataFrame()
    con.close()

    if df.empty:
        st.warning('No data yet — go interact with some recommendations first!')
        return

    summary = {}
    for variant in VARIANTS:
        v           = df[df['variant'] == variant]
        impressions = len(v[v['event'] == 'impression'])
        likes       = len(v[v['event'] == 'like'])
        dislikes    = len(v[v['event'] == 'dislike'])
        summary[variant] = {
            'impressions': impressions,
            'likes':       likes,
            'dislikes':    dislikes,
            'like_rate':   round(likes / impressions, 4) if impressions > 0 else 0
        }

    cols = st.columns(len(VARIANTS))
    # label = {'collaborative': 'SVD Collab', 'content': 'Content', 'ncf': 'NCF Neural'}
    
    for col, variant in zip(cols, VARIANTS):
        with col:
            s = summary[variant]
            st.subheader(VARIANT_LABELS.get(variant, variant))
            st.metric('Like Rate',   f"{s['like_rate']:.2%}")
            st.metric('Impressions', s['impressions'])
            st.metric('Likes',       s['likes'])
            st.metric('Dislikes',    s['dislikes'])

    # Bar chart
    chart_df = pd.DataFrame({
        'Variant':   [VARIANT_LABELS.get(v, v) for v in summary.keys()],
        'Like Rate': [v['like_rate'] for v in summary.values()]
    })
    st.bar_chart(chart_df.set_index('Variant'))

    # ── Statistical significance test ────────────────────────────
    st.subheader('📈 Model Comparison')
    from scipy import stats

    # Fix: Changed cactive to active
    active = [v for v in VARIANTS if summary[v]['impressions'] > 0]
    if len(active) < 2:
        st.info('Need data for at least two variants to run comparisons.')
        return

    # Chi-square across all active variants (omnibus test)
    contingency = [
        [summary[v]['likes'], summary[v]['impressions'] - summary[v]['likes']]
        for v in active
    ]
    
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    col1, col2 = st.columns(2)
    col1.metric('Chi-square (omnibus)', f"{chi2:.3f}")
    col2.metric('p-value', f"{p:.4f}")
    
    if p < 0.05:
        st.success(f'✅ Statistically significant difference detected overall (p={p:.4f}).')
    else:
        st.warning(f'⚠️ No significant overall difference yet (p={p:.4f}) — collect more data.')

    st.divider()

    # Pairwise Bayesian comparison vs best
    best_variant  = max(active, key=lambda v: summary[v]['like_rate'])
    best_s        = summary[best_variant]
    best_name     = VARIANT_LABELS.get(best_variant, best_variant.capitalize())

    st.markdown(f"### **Leading Variant: `{best_name}`** ({best_s['like_rate']:.2%} like rate)")
    
    samples = 100_000
    best_beta = stats.beta(best_s['likes'] + 1, best_s['impressions'] - best_s['likes'] + 1)
    best_samp = best_beta.rvs(samples)

    # Loop through active challengers to compare against the leader
    for v in active:
        if v == best_variant:
            continue
            
        s = summary[v]
        b = stats.beta(s['likes'] + 1, s['impressions'] - s['likes'] + 1)
        samp = b.rvs(samples)
        
        win_prob = (best_samp > samp).mean()
        current_name = VARIANT_LABELS.get(v, v.capitalize())

        st.markdown(f"**{best_name} vs {current_name}**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric('Current Leader', best_name)
        c2.metric('Challenger', current_name)
        c3.metric('Win Probability', f"{win_prob:.1%}")

        if win_prob >= 0.95:
            st.success(f'✅ **{best_name}** is better than {current_name} with {win_prob:.1%} probability.')
        elif win_prob >= 0.80:
            st.info(f'📊 **{best_name}** is likely better ({win_prob:.1%} probability) — keep collecting data.')
        else:
            st.warning(f'⚠️ Too close to call — {best_name} leads {current_name} with only {win_prob:.1%} probability.')
        
        st.divider()

    # Model info box
    info_lines = []
    if NCF_LOADED:
        info_lines.append("🧠 **NCF**: Wide & Deep, GELU MLP + attention. RMSE ~0.949 on ML-1M.")
    if BERT_LOADED:
        info_lines.append(
            "🤖 **BERT4Rec**: Pre-LN Transformer, Cloze + last-item fine-tuning. "
            "Val HR@10 ~0.60 | Test HR@10 ~0.55 | NDCG@10 ~0.34 on ML-1M. "
            "Sequence-based next-item prediction."
        )
    if info_lines:
        st.info('\n\n'.join(info_lines))

with st.sidebar:
    st.title('Navigation')
    if st.button('🏠 Home'):
        st.session_state.page = 'home' if st.session_state.user_id else 'landing'
        st.rerun()
    if st.button('📊 Results'):
        st.session_state.page = 'results'
        st.rerun()
    if st.button('🔄 Reset Session'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # NCF status badge
    st.divider()
    if NCF_LOADED:
        st.success('🧠 NCF model active')
    else:
        st.warning('🧠 NCF not loaded\nPlace `.pt` file in `models/`')
    if BERT_LOADED:  st.success('🤖 BERT4Rec active')
    else:            st.warning('🤖 BERT4Rec not loaded')

    # Liked movies in sidebar
    if st.session_state.liked_movies:
        st.divider()
        st.subheader(f'❤️ Liked Movies ({len(st.session_state.liked_movies)})')
        
        search = st.text_input('🔍 Search liked movies', placeholder='Type to search...')
        
        filtered = [
            m for m in st.session_state.liked_movies
            if search.lower() in m['title'].lower()
        ] if search else st.session_state.liked_movies
        
        # Scrollable container limited to 5 visible
        with st.container(height=300):
            if not filtered:
                st.caption('No matches found.')
            for movie in filtered[-20:]:  # show last 20 max
                st.markdown(f"**{movie['title']}**")
                st.caption(f"{movie['genres']} · via {movie['variant']}")
                st.divider()
                
# ── Router ───────────────────────────────────────────────────────
page = st.session_state.get('page', 'landing')
if page == 'landing':
    landing_page()
elif page == 'survey':
    survey_page()
elif page == 'home':
    if st.session_state.user_id is None:
        st.session_state.page = 'landing'
        st.rerun()
    home_page()
elif page == 'results':
    results_page()