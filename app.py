# app.py - Book Recommendation System (Full 10,000 Books with Search)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
def build_similarity_matrix(books):
    """Build TF-IDF similarity matrix for content-based filtering"""
    with st.spinner("Building similarity matrix for 10,000 books..."):
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

def search_books(search_term):
    """Search for books by title (case-insensitive)"""
    if not search_term:
        return pd.DataFrame()
    return books[books['title'].str.contains(search_term, case=False, regex=False)]

def recommend_content_based(book_title, n=6):
    """Content-based recommendation using cosine similarity"""
    # Find the book index
    matches = books[books['title'].str.contains(book_title, case=False, regex=False)]
    if len(matches) == 0:
        return pd.DataFrame()
    
    idx = matches.index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    book_indices = [i[0] for i in sim_scores]
    
    # Prepare results
    results = books.iloc[book_indices][['title', 'authors', 'average_rating']].copy()
    results['similarity_score'] = [i[1] for i in sim_scores]
    return results

def recommend_collaborative(user_id, n=6):
    """Collaborative filtering using SVD (Placeholder for Member A)"""
    # This is a placeholder. Member A will replace with actual SVD model.
    top_books = books.nlargest(n, 'average_rating')[['title', 'authors', 'average_rating']]
    top_books['predicted_rating'] = top_books['average_rating'] / 5.0
    return top_books

def recommend_hybrid(book_title, n=6, collab_weight=0.5):
    """Hybrid recommendation combining both methods"""
    content_results = recommend_content_based(book_title, n*2)
    
    # For now, just return content-based results with hybrid score
    if len(content_results) > 0:
        content_results['hybrid_score'] = content_results['similarity_score']
        return content_results.head(n)
    return pd.DataFrame()

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("⚙️ Settings")

method = st.sidebar.selectbox(
    "Choose Recommendation Method",
    ["🔍 Content-Based (TF-IDF)", "👥 Collaborative (SVD)", "🎯 Hybrid"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Statistics")
st.sidebar.metric("📚 Total Books", f"{len(books):,}")
st.sidebar.metric("⭐ Total Ratings", f"{len(ratings):,}")
st.sidebar.metric("👤 Total Users", f"{ratings['user_id'].nunique():,}")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
**Content-Based (TF-IDF)**: Finds books similar to your search using text similarity on titles, authors, and tags.

**Collaborative (SVD)**: Finds books liked by users with similar taste (requires user ID).

**Hybrid**: Combines both approaches for optimal recommendations.
""")

# ============================================
# MAIN CONTENT
# ============================================
st.title("📚 Book Recommendation System")
st.markdown("*Discover your next favorite book using AI | Search across 10,000 books*")

# Display current method
if method == "🔍 Content-Based (TF-IDF)":
    st.info("🔍 **Content-Based Mode**: Search for any book to find similar titles based on genre, author, and tags.")
elif method == "👥 Collaborative (SVD)":
    st.info("👥 **Collaborative Mode**: Select a user to get personalized recommendations based on their reading history.")
else:
    st.info("🎯 **Hybrid Mode**: Combines content similarity with collaborative filtering for better recommendations.")

st.markdown("---")

# Create two columns
col1, col2 = st.columns([1, 1])

# ============================================
# COLUMN 1: USER INPUT
# ============================================
with col1:
    if method == "🔍 Content-Based (TF-IDF)":
        st.subheader("🔍 Search for a Book")
        
        # Search box for ALL 10,000 books
        search_term = st.text_input(
            "Type a book title:",
            placeholder="e.g., Harry Potter, Hunger Games, Pride and Prejudice, Dune",
            help="Search across all 10,000 books in the database"
        )
        
        if search_term:
            # Search across ALL books
            search_results = search_books(search_term)
            
            if len(search_results) > 0:
                st.success(f"✅ Found {len(search_results)} book(s) matching '{search_term}'")
                
                # Show results in a select box (max 50 for performance)
                book_options = search_results['title'].head(50).tolist()
                selected_book = st.selectbox("Select a book:", options=book_options)
                
                if st.button("🔍 Get Similar Books", type="primary", use_container_width=True):
                    with st.spinner("Finding similar books..."):
                        results = recommend_content_based(selected_book, n=6)
                        st.session_state['results'] = results
                        st.session_state['selected_book'] = selected_book
                        st.session_state['method'] = 'content'
            else:
                st.warning(f"❌ No books found matching '{search_term}'. Try a different search.")
        else:
            st.info("👆 Enter a book title above to search across 10,000 books")
            st.caption("**Example searches:** Harry, Hunger, Pride, Tolkien, Rowling, Orwell, Austen, Stephen King")
    
    elif method == "👥 Collaborative (SVD)":
        st.subheader("👤 Select a User")
        
        # Get top users with most ratings
        user_list = ratings.groupby('user_id').size().sort_values(ascending=False).head(50)
        user_options = [f"User {uid} ({count} ratings)" for uid, count in user_list.items()]
        
        selected_user = st.selectbox("Select a user:", options=user_options)
        user_id = int(selected_user.split()[1])
        
        st.caption(f"User {user_id} has rated {user_list[user_id]:,} books")
        
        if st.button("👥 Get Personalized Recommendations", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing preferences for User {user_id}..."):
                results = recommend_collaborative(user_id, n=6)
                st.session_state['results'] = results
                st.session_state['selected_user'] = user_id
                st.session_state['method'] = 'collaborative'
    
    else:  # Hybrid
        st.subheader("🎯 Hybrid Recommendations")
        
        # Search box for books
        search_term = st.text_input(
            "Search for a book you like:",
            placeholder="e.g., Harry Potter",
            key="hybrid_search"
        )
        
        if search_term:
            search_results = search_books(search_term)
            if len(search_results) > 0:
                selected_book = st.selectbox(
                    "Select a book:",
                    options=search_results['title'].head(50).tolist(),
                    key="hybrid_book"
                )
                
                # Optional: User selection for collaborative part
                st.markdown("---")
                st.caption("Optional: Select a user to include collaborative filtering")
                user_list = ratings.groupby('user_id').size().sort_values(ascending=False).head(20)
                user_options = ["(None)"] + [f"User {uid}" for uid in user_list.index]
                selected_user_option = st.selectbox("Or select a user:", options=user_options, key="hybrid_user")
                
                # Weight adjustment
                st.markdown("---")
                st.caption("Adjust the balance between methods:")
                collab_weight = st.slider("Collaborative Weight", 0.0, 1.0, 0.5, 0.1)
                st.caption(f"Content Weight: {1.0 - collab_weight:.1f}")
                
                if st.button("🎯 Get Hybrid Recommendations", type="primary", use_container_width=True):
                    with st.spinner("Combining content-based + collaborative recommendations..."):
                        results = recommend_hybrid(selected_book, n=6, collab_weight=collab_weight)
                        st.session_state['results'] = results
                        st.session_state['selected_book'] = selected_book
                        st.session_state['method'] = 'hybrid'
            else:
                st.warning(f"No books found matching '{search_term}'")
        else:
            st.info("👆 Search for a book above to get hybrid recommendations")

# ============================================
# COLUMN 2: RESULTS
# ============================================
with col2:
    st.subheader("🎯 Recommendations")
    
    if 'results' in st.session_state and st.session_state['results'] is not None:
        results = st.session_state['results']
        
        if len(results) == 0:
            st.warning("No recommendations found. Try a different book or user!")
        else:
            # Show what the recommendation is based on
            if 'selected_book' in st.session_state:
                st.caption(f"Based on: **{st.session_state['selected_book']}**")
            if 'selected_user' in st.session_state:
                st.caption(f"Based on: **User {st.session_state['selected_user']}**")
            
            st.markdown("---")
            
            # Display results in a clean grid
            for idx, row in results.iterrows():
                with st.container():
                    col_a, col_b = st.columns([1, 4])
                    
                    with col_a:
                        st.markdown("📘")
                    
                    with col_b:
                        # Book title
                        st.markdown(f"**{row['title']}**")
                        
                        # Author
                        st.caption(f"by {row['authors']}")
                        
                        # Rating stars
                        if 'average_rating' in row:
                            rating = row['average_rating']
                            full_stars = int(rating)
                            half_star = 1 if (rating - full_stars) >= 0.5 else 0
                            empty_stars = 5 - full_stars - half_star
                            stars = "⭐" * full_stars + "½" * half_star + "☆" * empty_stars
                            st.caption(f"{stars} ({rating}/5)")
                        
                        # Similarity score progress bar
                        if 'similarity_score' in row:
                            score = min(row['similarity_score'], 1.0)
                            st.progress(score, text=f"Similarity: {score:.0%}")
                        
                        if 'hybrid_score' in row:
                            score = min(row['hybrid_score'], 1.0)
                            st.progress(score, text=f"Hybrid Score: {score:.0%}")
                    
                    st.divider()
    else:
        st.info("👈 Enter your preferences and click 'Get Recommendations'")
        
        # Show example books
        st.markdown("### 📖 Popular Books You Can Search")
        popular_books = [
            "The Hunger Games", "Harry Potter and the Sorcerer's Stone",
            "Pride and Prejudice", "The Great Gatsby", "To Kill a Mockingbird",
            "1984", "The Hobbit", "Fahrenheit 451", "The Catcher in the Rye"
        ]
        for book in popular_books[:5]:
            st.caption(f"• {book}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption(f"📚 Book Recommendation System | Powered by TF-IDF + Cosine Similarity | Dataset: Goodbooks-10k ({len(books):,} books, {len(ratings):,} ratings)")
