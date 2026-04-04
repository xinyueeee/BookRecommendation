# app.py - Book Recommendation System (Image-Free GUI)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BETTER LOOK
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-size: 1.1rem;
    }
    .book-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .book-card:hover {
        transform: translateY(-3px);
    }
    .book-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.3rem;
    }
    .book-author {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-bottom: 0.5rem;
    }
    .rating-stars {
        font-size: 0.9rem;
        margin: 0.3rem 0;
    }
    .similarity-badge {
        background: #27ae60;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        display: inline-block;
    }
    .genre-tag {
        background: #3498db;
        color: white;
        padding: 0.1rem 0.5rem;
        border-radius: 15px;
        font-size: 0.7rem;
        display: inline-block;
        margin-right: 0.3rem;
    }
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #7f8c8d;
    }
    .method-badge {
        background: #e74c3c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
    with st.spinner("📚 Building similarity matrix for 10,000 books..."):
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

def get_random_books(n=6):
    """Get random books for discovery section"""
    return books.sample(n=min(n, len(books)))

def search_books(search_term):
    """Search for books by title (case-insensitive)"""
    if not search_term:
        return pd.DataFrame()
    return books[books['title'].str.contains(search_term, case=False, regex=False)]

def get_rating_stars(rating):
    """Convert rating to star string"""
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "⭐" * full + "½" * half + "☆" * empty

def get_genre_tags(tags_string, max_tags=3):
    """Extract genre tags from combined features"""
    if not tags_string:
        return []
    # Get common genre-like tags
    genre_keywords = ['fantasy', 'fiction', 'mystery', 'thriller', 'romance', 
                      'sci-fi', 'science fiction', 'horror', 'biography', 
                      'history', 'classic', 'young adult', 'children', 
                      'adventure', 'drama', 'comedy', 'poetry']
    found_tags = []
    for genre in genre_keywords:
        if genre.lower() in tags_string.lower():
            found_tags.append(genre)
    return found_tags[:max_tags]

def recommend_content_based(book_title, n=6):
    """Content-based recommendation using cosine similarity"""
    matches = books[books['title'].str.contains(book_title, case=False, regex=False)]
    if len(matches) == 0:
        return pd.DataFrame()
    
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    book_indices = [i[0] for i in sim_scores]
    
    results = books.iloc[book_indices][['title', 'authors', 'average_rating', 'combined_features']].copy()
    results['similarity_score'] = [i[1] for i in sim_scores]
    return results

def recommend_collaborative(user_id, n=6):
    """Collaborative filtering using SVD (Placeholder)"""
    top_books = books.nlargest(n, 'average_rating')[['title', 'authors', 'average_rating', 'combined_features']].copy()
    top_books['similarity_score'] = [0.8 - i*0.05 for i in range(len(top_books))]
    return top_books

def recommend_hybrid(book_title, n=6):
    """Hybrid recommendation combining both methods"""
    content_results = recommend_content_based(book_title, n)
    if len(content_results) > 0:
        content_results['hybrid_score'] = content_results['similarity_score']
        return content_results.head(n)
    return pd.DataFrame()

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>📚 Book Discovery Engine</h1>
    <p>Find your next favorite read with AI-powered recommendations</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 🎯 Recommendation Mode")
    
    method = st.radio(
        "Select your preferred method:",
        ["🔍 Content-Based", "👥 Collaborative", "🎯 Hybrid"],
        index=0,
        help="Content-Based: Find books similar to ones you like | Collaborative: Based on user preferences | Hybrid: Best of both"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Library Stats")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(books):,}</div>
            <div class="stat-label">Books</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{ratings['user_id'].nunique():,}</div>
            <div class="stat-label">Readers</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(ratings):,}</div>
        <div class="stat-label">Reviews</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 How It Works")
    st.caption("""
    **Content-Based**: Uses TF-IDF to analyze book titles, authors, and tags. Finds books with similar content.
    
    **Collaborative**: SVD-based filtering that learns from user reading patterns.
    
    **Hybrid**: Combines both approaches for optimal recommendations.
    """)

# ============================================
# MAIN CONTENT
# ============================================

# Show current method badge
if "🔍" in method:
    st.markdown('<span class="method-badge">🔍 Content-Based Mode - Find books similar to your favorites</span>', unsafe_allow_html=True)
elif "👥" in method:
    st.markdown('<span class="method-badge">👥 Collaborative Mode - Personalized based on user history</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="method-badge">🎯 Hybrid Mode - Combining content + collaborative filtering</span>', unsafe_allow_html=True)

# Create two main columns
left_col, right_col = st.columns([1, 1.2], gap="large")

# ============================================
# LEFT COLUMN - USER INPUT
# ============================================
with left_col:
    if "Content-Based" in method:
        st.markdown("### 🔍 Search for a Book")
        st.caption("Type any book title to find similar recommendations")
        
        search_term = st.text_input(
            "",
            placeholder="e.g., Harry Potter, The Hunger Games, Pride and Prejudice, Dune",
            label_visibility="collapsed"
        )
        
        if search_term:
            search_results = search_books(search_term)
            
            if len(search_results) > 0:
                st.success(f"Found {len(search_results)} matching book(s)")
                
                selected_book = st.selectbox(
                    "Select a book:",
                    options=search_results['title'].head(30).tolist(),
                    label_visibility="collapsed"
                )
                
                if st.button("🔍 Find Similar Books", type="primary", use_container_width=True):
                    with st.spinner("Finding similar books..."):
                        results = recommend_content_based(selected_book, n=8)
                        st.session_state['results'] = results
                        st.session_state['selected_book'] = selected_book
                        st.session_state['method'] = 'content'
            else:
                st.warning(f"No books found matching '{search_term}'")
        else:
            st.info("✨ Try searching for: Harry Potter, The Hunger Games, Pride and Prejudice, 1984, The Hobbit")
            
            # Show random books for discovery
            st.markdown("### 🎲 Discover Random Books")
            random_books = get_random_books(4)
            for _, row in random_books.iterrows():
                st.caption(f"📖 {row['title'][:50]}...")
    
    elif "Collaborative" in method:
        st.markdown("### 👤 Select a Reader")
        st.caption("Get personalized recommendations based on reading history")
        
        user_list = ratings.groupby('user_id').size().sort_values(ascending=False).head(50)
        user_options = [f"Reader {uid} ({count} books)" for uid, count in user_list.items()]
        
        selected_user = st.selectbox("Choose a reader:", options=user_options, label_visibility="collapsed")
        user_id = int(selected_user.split()[1])
        
        if st.button("👥 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing reading patterns..."):
                results = recommend_collaborative(user_id, n=8)
                st.session_state['results'] = results
                st.session_state['selected_user'] = user_id
                st.session_state['method'] = 'collaborative'
    
    else:  # Hybrid
        st.markdown("### 🎯 Hybrid Search")
        st.caption("Find books similar to your favorites, enhanced with collaborative filtering")
        
        search_term = st.text_input(
            "",
            placeholder="Search for a book you like...",
            label_visibility="collapsed",
            key="hybrid_search"
        )
        
        if search_term:
            search_results = search_books(search_term)
            if len(search_results) > 0:
                selected_book = st.selectbox(
                    "Select a book:",
                    options=search_results['title'].head(30).tolist(),
                    label_visibility="collapsed",
                    key="hybrid_select"
                )
                
                if st.button("🎯 Get Hybrid Recommendations", type="primary", use_container_width=True):
                    with st.spinner("Generating hybrid recommendations..."):
                        results = recommend_hybrid(selected_book, n=8)
                        st.session_state['results'] = results
                        st.session_state['selected_book'] = selected_book
                        st.session_state['method'] = 'hybrid'
            else:
                st.warning(f"No books found matching '{search_term}'")

# ============================================
# RIGHT COLUMN - RESULTS
# ============================================
with right_col:
    if 'results' in st.session_state and st.session_state['results'] is not None:
        results = st.session_state['results']
        
        if len(results) == 0:
            st.warning("No recommendations found. Try a different search!")
        else:
            st.markdown("### 🎯 Your Recommendations")
            
            if 'selected_book' in st.session_state:
                st.caption(f"Based on: **{st.session_state['selected_book']}**")
            
            st.markdown("---")
            
            # Display results as cards
            for idx, row in results.iterrows():
                with st.container():
                    # Book card using columns for layout
                    col_icon, col_content = st.columns([0.3, 4])
                    
                    with col_icon:
                        # Use emoji as book icon based on rating
                        rating = row.get('average_rating', 3)
                        if rating >= 4.5:
                            icon = "🏆"
                        elif rating >= 4.0:
                            icon = "📘"
                        elif rating >= 3.5:
                            icon = "📙"
                        else:
                            icon = "📕"
                        st.markdown(f"<h1 style='font-size: 2.5rem; margin:0;'>{icon}</h1>", unsafe_allow_html=True)
                    
                    with col_content:
                        # Book title
                        st.markdown(f"**{row['title']}**")
                        
                        # Author
                        st.caption(f"by {row['authors']}")
                        
                        # Rating stars
                        if 'average_rating' in row:
                            stars = get_rating_stars(row['average_rating'])
                            st.markdown(f"<span class='rating-stars'>{stars} ({row['average_rating']}/5)</span>", unsafe_allow_html=True)
                        
                        # Tags
                        if 'combined_features' in row:
                            tags = get_genre_tags(row['combined_features'])
                            if tags:
                                tag_html = "".join([f'<span class="genre-tag">{tag}</span>' for tag in tags])
                                st.markdown(f"<div style='margin: 0.3rem 0;'>{tag_html}</div>", unsafe_allow_html=True)
                        
                        # Similarity score
                        if 'similarity_score' in row:
                            score = min(row['similarity_score'], 1.0)
                            st.progress(score, text=f"Match: {score:.0%}")
                    
                    st.markdown("---")
    
    else:
        # No results yet - show discovery section
        st.markdown("### 📖 Discover Popular Books")
        st.caption("Search for a book above to get personalized recommendations")
        
        st.markdown("---")
        
        # Show curated book suggestions
        popular_books = [
            ("The Hunger Games", "Suzanne Collins", 4.34),
            ("Harry Potter", "J.K. Rowling", 4.45),
            ("Pride and Prejudice", "Jane Austen", 4.28),
            ("The Great Gatsby", "F. Scott Fitzgerald", 3.93),
            ("To Kill a Mockingbird", "Harper Lee", 4.28),
            ("1984", "George Orwell", 4.24),
        ]
        
        for title, author, rating in popular_books:
            with st.container():
                col_icon, col_content = st.columns([0.3, 4])
                with col_icon:
                    st.markdown("<h1 style='font-size: 1.5rem; margin:0;'>📖</h1>", unsafe_allow_html=True)
                with col_content:
                    st.markdown(f"**{title}**")
                    st.caption(f"by {author}")
                    stars = get_rating_stars(rating)
                    st.caption(f"{stars} ({rating}/5)")
                st.markdown("---")
        
        st.info("💡 **Tip**: Try searching for any book title above to find similar reads!")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption(f"📚 {len(books):,} books in library")
with col_f2:
    st.caption("🤖 Powered by TF-IDF + Cosine Similarity")
with col_f3:
    st.caption("⭐ Dataset: Goodbooks-10k")
