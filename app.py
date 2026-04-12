from flask import Flask, request, jsonify, render_template, session, redirect
import joblib, sqlite3, random, pandas as pd, numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# ── Load models & data ──────────────────────────────────────────
best_svd = joblib.load('best_svd.pkl')

movies  = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Build content similarity matrix
movies['genre_list'] = movies['genres'].str.split('|')
mlb          = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genre_list'])
genre_df     = pd.DataFrame(genre_matrix, index=movies['movieId'], columns=mlb.classes_)
movie_sim_df = pd.DataFrame(
    cosine_similarity(genre_df),
    index=genre_df.index,
    columns=genre_df.index
)

ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]

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

# ── Cold start helpers ───────────────────────────────────────────
def is_cold_start(user_id):
    con = sqlite3.connect('logs.db')
    cur = con.execute(
        "SELECT COUNT(*) FROM events WHERE user_id=? AND event IN ('like','dislike')",
        (str(user_id),)
    )
    count = cur.fetchone()[0]
    con.close()
    return count < 5

def get_cold_start_recs(preferred_genres, n=10):
    popularity = ratings.groupby('movieId').size().reset_index(name='count')
    popular    = popularity.sort_values('count', ascending=False)
    popular    = popular.merge(movies, on='movieId')
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

# ── Explanation helpers ──────────────────────────────────────────
def explain_svd(user_id, recommended_movie_id):
    uid        = int(user_id)
    top_rated  = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4)]
    top_rated  = top_rated.merge(movies, on='movieId').sort_values('rating', ascending=False)

    if top_rated.empty:
        return "Popular with users like you"

    rec_movie  = movies[movies['movieId'] == recommended_movie_id].iloc[0]
    rec_genres = set(rec_movie['genres'].split('|'))

    for _, row in top_rated.head(10).iterrows():
        user_genres = set(row['genres'].split('|'))
        overlap     = rec_genres & user_genres
        if overlap:
            return f"Because you liked {row['title']} ({', '.join(overlap)})"

    return "Popular with users who share your taste"

def explain_content(user_id, recommended_movie_id):
    uid  = int(user_id)
    liked = ratings[(ratings['userId'] == uid) &
                    (ratings['rating'] >= 4)]['movieId'].tolist()

    if not liked:
        return "Matches your genre preferences"

    rec_movie    = movies[movies['movieId'] == recommended_movie_id].iloc[0]
    rec_genres   = set(rec_movie['genres'].split('|'))
    best_match   = None
    best_overlap = 0

    for mid in liked:
        row = movies[movies['movieId'] == mid]
        if row.empty:
            continue
        user_genres = set(row.iloc[0]['genres'].split('|'))
        overlap     = len(rec_genres & user_genres)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match   = row.iloc[0]['title']

    if best_match:
        return f"Because you liked {best_match}"
    return "Matches your genre preferences"

# ── Recommendation functions ─────────────────────────────────────
def get_svd_recs(user_id, n=10):
    rated = ratings[ratings['userId'] == int(user_id)]['movieId'].tolist()
    seen  = get_seen_movies(user_id)
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
    liked = ratings[(ratings['userId'] == uid) &
                    (ratings['rating'] >= 4)]['movieId'].tolist()
    if not liked:
        liked = ratings[ratings['userId'] == uid]['movieId'].tolist()

    rated = ratings[ratings['userId'] == uid]['movieId'].tolist()
    seen  = get_seen_movies(user_id)

    sim_scores = movie_sim_df[liked].mean(axis=1)
    sim_scores = sim_scores.drop(index=[m for m in rated + seen if m in sim_scores.index], errors='ignore')
    top_movies = sim_scores.nlargest(n).index.tolist()

    results = []
    for mid in top_movies:
        row = movies[movies['movieId'] == mid].iloc[0]
        results.append({'id': int(mid), 'title': row['title'], 'genres': row['genres']})
    return results

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/start', methods=['POST'])
def start():
    user_type = request.json.get('user_type')

    session['user_type'] = user_type

    if user_type == 'new':
        session['user_id'] = str(random.randint(10000, 99999))
    else:
        session['user_id'] = str(random.randint(1, 6040))

    session['variant'] = 'collaborative' if random.random() < 0.5 else 'content'

    return jsonify({'status': 'ok'})

@app.route('/home')
def index():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('index.html',
                           user_id=session['user_id'],
                           variant=session['variant'])

@app.route('/survey', methods=['POST'])
def survey():
    data   = request.json
    genres = data.get('genres', [])
    session['preferred_genres'] = genres
    session['onboarded']        = True
    return jsonify({'status': 'ok'})

@app.route('/recommendations')
def recommendations():
    user_id = session.get('user_id')
    variant = session.get('variant')

    user_type = session.get('user_type')

    # Only apply cold start to NEW users
    if user_type == 'new' and is_cold_start(user_id):
        preferred = session.get('preferred_genres', [])
        if not preferred:
            return jsonify({'show_survey': True})
        recs = get_cold_start_recs(preferred)
        for rec in recs:
            log_event(user_id, variant, rec['id'], 'impression')
        return jsonify({'variant': variant, 'recommendations': recs})

    if variant == 'collaborative':
        recs = get_svd_recs(user_id)
        for rec in recs:
            rec['explanation'] = explain_svd(user_id, rec['id'])
    else:
        recs = get_content_recs(user_id)
        for rec in recs:
            rec['explanation'] = explain_content(user_id, rec['id'])

    for rec in recs:
        log_event(user_id, variant, rec['id'], 'impression')

    return jsonify({'variant': variant, 'recommendations': recs})

@app.route('/feedback', methods=['POST'])
def feedback():
    data     = request.json
    user_id  = session.get('user_id')
    variant  = session.get('variant')
    movie_id = data.get('movie_id')
    action   = data.get('action')  # 'like' or 'dislike'

    log_event(user_id, variant, movie_id, action)

    return jsonify({'status': 'logged'})

@app.route('/results')
def results():
    con = sqlite3.connect('logs.db')
    df  = pd.read_sql('SELECT * FROM events', con)
    con.close()

    if df.empty:
        return jsonify({'message': 'No data yet'})

    summary = {}
    for variant in ['collaborative', 'content']:
        v           = df[df['variant'] == variant]
        impressions = len(v[v['event'] == 'impression'])
        likes    = len(v[v['event'] == 'like'])
        dislikes = len(v[v['event'] == 'dislike'])

        summary[variant] = {
            'impressions': impressions,
            'likes': likes,
            'dislikes': dislikes,
            'like_rate': round(likes / impressions, 4) if impressions > 0 else 0
        }

    return jsonify(summary)

def get_seen_movies(user_id):
    con = sqlite3.connect('logs.db')
    df = pd.read_sql(
        "SELECT movie_id FROM events WHERE user_id=? AND event IN ('like','dislike')",
        con,
        params=(str(user_id),)
    )
    con.close()
    return df['movie_id'].tolist()

@app.route('/reset')
def reset():
    session.clear()
    return 'Session cleared. <a href="/">Go back</a>'

if __name__ == '__main__':
    app.run(debug=True)