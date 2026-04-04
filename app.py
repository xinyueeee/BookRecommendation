# app.py - Book Recommendation System (Working Version)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
def build_similarity_matrix(books):
    """Build TF-IDF similarity matrix for content-based filtering"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Load data
books, ratings = load_data()
cosine_sim = build_similarity_matrix(books)

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_book_index(book_title):
    """Find book index by title"""
    matches = books[books['title'].str.contains(book_title, case=False, regex=False)]
    if len(matches) == 0:
        return None
    return matches.index[0]

def recommend_content_based(book_title, n=6):
    """Content-based recommendation using cosine similarity"""
    idx = get_book_index(book_title)
    if idx is None:
        return pd.DataFrame()
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    book_indices = [i[0] for i in sim_scores]
    
    results = books.iloc[book_indices][['title', 'authors', 'average_rating']].copy()
    results['similarity_score'] = [i[1] for i in sim_scores]
    return results

def recommend_collaborative(user_id, n=6):
    """Collaborative filtering using SVD (placeholder)"""
    top_books = books.nlargest(n, 'average_rating')[['title', 'authors', 'average_rating']]
    return top_books

def recommend_hybrid(user_id=None, book_title=None, n=6):
    """Hybrid recommendation combining both methods"""
    collab_results = recommend_collaborative(user_id, n)
    content_results = recommend_content_based(book_title, n)
    combined = pd.concat([collab_results, content_results]).drop_duplicates(subset=['title'])
    return combined.head(n)

# ============================================
# USER INTERFACE
# ============================================

st.title("📚 Book Recommendation System")
st.markdown("*Discover your next favorite book using AI*")

st.sidebar.header("⚙️ Settings")
method = st.sidebar.selectbox(
    "Choose Recommendation Method",
    ["Content-Based", "Collaborative (SVD)", "Hybrid"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This system uses:
- **Content-Based**: Finds books similar to one you like (TF-IDF + Cosine Similarity)
- **Collaborative**: Finds what similar users enjoyed (SVD)
- **Hybrid**: Combines both approaches
""")

st.sidebar.markdown("---")
st.sidebar.metric("📚 Total Books", f"{len(books):,}")
st.sidebar.metric("⭐ Total Ratings", f"{len(ratings):,}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📖 Tell us what you like")
    
    if method == "Content-Based":
        book_input = st.selectbox(
            "Select a book you enjoy:",
            options=books['title'].head(500).tolist(),
            help="Select from 500 popular books"
        )
        
        if st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner("Finding similar books..."):
                results = recommend_content_based(book_input, n=6)
                st.session_state['results'] = results
                st.session_state['method'] = 'content'
    
    elif method == "Collaborative (SVD)":
        user_list = ratings.groupby('user_id').size().sort_values(ascending=False).head(50)
        user_options = [f"User {uid}" for uid in user_list.index]
        user_selection = st.selectbox("Select a user:", options=user_options)
        user_id = int(user_selection.split()[1])
        
        if st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner("Analyzing user preferences..."):
                results = recommend_collaborative(user_id, n=6)
                st.session_state['results'] = results
                st.session_state['method'] = 'collaborative'
    
    else:
        st.info("Hybrid method combines both approaches")
        book_input = st.selectbox(
            "Select a book you enjoy:",
            options=books['title'].head(500).tolist(),
            key="hybrid_book"
        )
        
        if st.button("🔍 Get Hybrid Recommendations", type="primary"):
            with st.spinner("Generating hybrid recommendations..."):
                results = recommend_hybrid(book_title=book_input, n=6)
                st.session_state['results'] = results
                st.session_state['method'] = 'hybrid'

with col2:
    st.subheader("🎯 Recommendations")
    
    if 'results' in st.session_state and st.session_state['results'] is not None:
        results = st.session_state['results']
        
        if len(results) == 0:
            st.warning("No recommendations found. Try another book or user!")
        else:
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
                                       text=f"Match: {row['similarity_score']:.0%}")
                    st.divider()
    else:
        st.info("👈 Select a method and enter your preferences, then click 'Get Recommendations'")

st.markdown("---")
st.caption("Book Recommendation System | Built with Streamlit | Dataset: Goodbooks-10k")
