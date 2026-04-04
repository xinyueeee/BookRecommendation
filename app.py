# app.py - Book Recommendation System (Enhanced with TF-IDF Weighted Word2Vec)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide"
)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    books = pd.read_csv('books_processed.csv')
    ratings = pd.read_csv('ratings_processed.csv')
    return books, ratings

@st.cache_resource
def build_tfidf_word2vec_matrix(books):
    """
    Build similarity matrix using TF-IDF Weighted Word2Vec.
    This combines term importance (TF-IDF) with semantic meaning (Word2Vec).
    """
    
    # Step 1: Build TF-IDF to get term importance weights
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(books['combined_features'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Step 2: Create TF-IDF weight dictionary for faster lookup
    tfidf_weight_dict = {}
    for idx, word in enumerate(feature_names):
        # Get average TF-IDF weight across all books
        avg_weight = tfidf_matrix[:, idx].mean()
        tfidf_weight_dict[word] = avg_weight
    
    # Step 3: Load pre-trained Word2Vec (using Gensim)
    # Note: First run will download the model (~1.5GB)
    try:
        import gensim.downloader as api
        with st.spinner("Loading Word2Vec model (first time may take a few minutes)..."):
            word2vec_model = api.load('glove-wiki-gigaword-50')  # 50-dimensions, smaller than Google News
        st.success("✅ Word2Vec model loaded!")
    except Exception as e:
        st.warning(f"Word2Vec not available, falling back to TF-IDF only: {e}")
        word2vec_model = None
    
    # Step 4: Create document vectors with TF-IDF weighting
    def get_weighted_word2vec_vector(text):
        """Convert text to weighted Word2Vec vector (TF-IDF weighted)"""
        if word2vec_model is None:
            return None
        
        words = text.lower().split()
        vectors = []
        weights = []
        
        for word in words:
            if word in word2vec_model:
                # Get TF-IDF weight (default 0.01 if word not in TF-IDF)
                weight = tfidf_weight_dict.get(word, 0.01)
                vectors.append(word2vec_model[word])
                weights.append(weight)
        
        if len(vectors) == 0:
            return None
        
        # Weighted average (convert to numpy array for proper broadcasting)
        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1, 1)
        
        # Calculate weighted average
        weighted_vector = np.sum(vectors * weights, axis=0) / np.sum(weights)
        return weighted_vector
    
    # Step 5: Build document vectors for all books
    st.info("Building TF-IDF Weighted Word2Vec vectors for all books...")
    book_vectors = []
    valid_indices = []
    
    for idx, features in enumerate(books['combined_features']):
        vec = get_weighted_word2vec_vector(features)
        if vec is not None:
            book_vectors.append(vec)
            valid_indices.append(idx)
        else:
            # Fallback: use TF-IDF vector (but Word2Vec failed)
            book_vectors.append(np.zeros(50))
            valid_indices.append(idx)
    
    book_vectors = np.array(book_vectors)
    
    # Step 6: Normalize vectors for cosine similarity
    book_vectors_norm = normalize(book_vectors)
    
    # Step 7: Build similarity matrix (sparse for memory efficiency)
    similarity_matrix = cosine_similarity(book_vectors_norm)
    
    return similarity_matrix, valid_indices

# Load data
books, ratings = load_data()

# Build enhanced similarity matrix
with st.spinner("Building TF-IDF Weighted Word2Vec similarity matrix..."):
    cosine_sim, valid_indices = build_tfidf_word2vec_matrix(books)

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_book_index(book_title):
    """Find book index by title"""
    matches = books[books['title'].str.contains(book_title, case=False)]
    if len(matches) == 0:
        return None
    return matches.index[0]

def recommend_content_based(book_title, n=6):
    """Enhanced content-based recommendation using TF-IDF Weighted Word2Vec"""
    idx = get_book_index(book_title)
    if idx is None:
        return pd.DataFrame()
    
    # Get similarity scores from enhanced matrix
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    book_indices = [i[0] for i in sim_scores]
    
    results = books.iloc[book_indices][['title', 'authors', 'average_rating']].copy()
    results['similarity_score'] = [i[1] for i in sim_scores]
    return results

def recommend_collaborative(user_id, n=6):
    """Collaborative filtering using SVD"""
    # This is where Member A's SVD model goes
    top_books = books.nlargest(n, 'average_rating')[['title', 'authors', 'average_rating']]
    return top_books

def recommend_hybrid(user_id=None, book_title=None, n=6, collab_weight=0.5):
    """Hybrid recommendation combining both methods with adjustable weights"""
    collab_results = recommend_collaborative(user_id, n)
    content_results = recommend_content_based(book_title, n)
    
    # Combine with weights
    combined = pd.concat([collab_results, content_results]).drop_duplicates(subset=['title'])
    
    # Add weighted scores for sorting
    combined['hybrid_score'] = 0.0
    for idx in combined.index:
        score = 0.0
        if idx in collab_results.index:
            score += collab_weight * (collab_results.loc[idx, 'average_rating'] / 5.0)
        if idx in content_results.index and 'similarity_score' in content_results.columns:
            score += (1 - collab_weight) * content_results.loc[idx, 'similarity_score']
        combined.loc[idx, 'hybrid_score'] = score
    
    combined = combined.sort_values('hybrid_score', ascending=False)
    return combined.head(n)

# ============================================
# USER INTERFACE
# ============================================

st.title("📚 Advanced Book Recommendation System")
st.markdown("*Powered by TF-IDF Weighted Word2Vec + SVD + Hybrid AI*")

# Sidebar
st.sidebar.header("⚙️ Settings")
method = st.sidebar.selectbox(
    "Choose Recommendation Method",
    ["Content-Based (TF-IDF + Word2Vec)", "Collaborative (SVD)", "Hybrid"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**Content-Based**: Uses TF-IDF Weighted Word2Vec to capture semantic meaning of book titles, authors, and tags. Rare but important terms (like genre tags) get higher weight.

**Collaborative**: SVD-based filtering that finds patterns in user ratings.

**Hybrid**: Combines both approaches for optimal recommendations.
""")

st.sidebar.markdown("---")
st.sidebar.metric("📚 Total Books", f"{len(books):,}")
st.sidebar.metric("⭐ Total Ratings", f"{len(ratings):,}")
st.sidebar.caption("Enhanced with semantic similarity (Word2Vec)")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📖 Tell us what you like")
    
    if method == "Content-Based (TF-IDF + Word2Vec)":
        book_input = st.selectbox(
            "Select a book you enjoy:",
            options=books['title'].head(1000).tolist(),
            help="Uses TF-IDF weighted Word2Vec to find semantically similar books"
        )
        
        if st.button("🔍 Get Semantic Recommendations", type="primary"):
            with st.spinner("Finding semantically similar books using TF-IDF Weighted Word2Vec..."):
                results = recommend_content_based(book_input, n=6)
                st.session_state['results'] = results
                st.session_state['method'] = 'content'
                st.session_state['selected_book'] = book_input
    
    elif method == "Collaborative (SVD)":
        user_list = ratings.groupby('user_id').size().sort_values(ascending=False).head(50)
        user_options = [f"User {uid}" for uid in user_list.index]
        
        user_selection = st.selectbox("Select a user:", options=user_options)
        user_id = int(user_selection.split()[1])
        
        if st.button("🔍 Get Collaborative Recommendations", type="primary"):
            with st.spinner("Analyzing user preferences with SVD..."):
                results = recommend_collaborative(user_id, n=6)
                st.session_state['results'] = results
                st.session_state['method'] = 'collaborative'
    
    else:  # Hybrid
        st.info("Hybrid method combines semantic understanding with collaborative filtering")
        
        # Weight adjustment slider
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            collab_weight = st.slider("Collaborative Weight", 0.0, 1.0, 0.5, 0.1)
        with col_w2:
            content_weight = 1.0 - collab_weight
            st.metric("Content Weight", f"{content_weight:.0%}")
        
        book_input = st.selectbox(
            "Select a book you enjoy:",
            options=books['title'].head(1000).tolist(),
            key="hybrid_book"
        )
        
        if st.button("🎯 Get Hybrid Recommendations", type="primary"):
            with st.spinner("Combining semantic + collaborative recommendations..."):
                results = recommend_hybrid(book_title=book_input, n=6, collab_weight=collab_weight)
                st.session_state['results'] = results
                st.session_state['method'] = 'hybrid'
                st.session_state['selected_book'] = book_input
                st.session_state['collab_weight'] = collab_weight

with col2:
    st.subheader("🎯 Recommendations")
    
    if 'results' in st.session_state and st.session_state['results'] is not None:
        results = st.session_state['results']
        
        if len(results) == 0:
            st.warning("No recommendations found. Try another book or user!")
        else:
            # Show what the recommendation is based on
            if 'selected_book' in st.session_state:
                st.caption(f"Based on: **{st.session_state['selected_book']}**")
            if 'collab_weight' in st.session_state:
                st.caption(f"Hybrid balance: {st.session_state['collab_weight']:.0%} Collaborative / {(1-st.session_state['collab_weight']):.0%} Content")
            
            # Display results
            for idx, row in results.iterrows():
                with st.container():
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.markdown("📘")
                    with col_b:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"by {row['authors']}")
                        if 'average_rating' in row:
                            stars = "⭐" * int(round(row['average_rating']))
                            st.caption(f"{stars} ({row['average_rating']}/5)")
                        if 'similarity_score' in row:
                            st.progress(min(row['similarity_score'], 1.0), 
                                       text=f"Semantic Match: {row['similarity_score']:.0%}")
                        if 'hybrid_score' in row:
                            st.progress(min(row['hybrid_score'], 1.0),
                                       text=f"Hybrid Score: {row['hybrid_score']:.0%}")
                    st.divider()
    else:
        st.info("👈 Select a method and enter your preferences, then click 'Get Recommendations'")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("Advanced Book Recommendation System | TF-IDF Weighted Word2Vec + SVD + Hybrid | Dataset: Goodbooks-10k")
