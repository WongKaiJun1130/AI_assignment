# ============================================================
# BOOK RECOMMENDER SYSTEM
# ============================================================

# Methods:
# 1. Popularity-Based Recommendation
# 2. Content-Based Filtering
# 3. Collaborative Filtering
# 4. Hybrid Recommendation


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ============================================================
# 2. LOAD DATASET
# ============================================================

books = pd.read_csv("data/Books.csv", low_memory=False)
ratings_data = pd.read_csv("data/Ratings.csv", low_memory=False)
users = pd.read_csv("data/Users.csv", low_memory=False)

print("Books:", books.shape)
print("Ratings:", ratings_data.shape)
print("Users:", users.shape)


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# Remove missing important values
books = books.dropna(subset=["ISBN", "Book-Title"]).copy()
ratings_data = ratings_data.dropna(subset=["User-ID", "ISBN", "Book-Rating"]).copy()

# Fill missing book information
books["Book-Author"] = books["Book-Author"].fillna("Unknown")
books["Publisher"] = books["Publisher"].fillna("Unknown")

# Convert rating to numeric
ratings_data["Book-Rating"] = pd.to_numeric(ratings_data["Book-Rating"], errors="coerce")

# Remove invalid ratings
ratings_data = ratings_data.dropna(subset=["Book-Rating"])

# Keep explicit ratings from 1 to 10
ratings_data = ratings_data[(ratings_data["Book-Rating"] >= 1) & (ratings_data["Book-Rating"] <= 10)]

# Remove duplicate books
books = books.drop_duplicates(subset=["ISBN"]).reset_index(drop=True)

# Remove duplicate user-book ratings
ratings_data = ratings_data.drop_duplicates(subset=["User-ID", "ISBN"]).reset_index(drop=True)

# Merge Books and Ratings
data = pd.merge(ratings_data, books, on="ISBN", how="inner")

print("\nCleaned Books:", len(books))
print("Cleaned Ratings:", len(ratings_data))
print("Merged Records:", len(data))


# ============================================================
# 4. DATA ANALYSIS
# ============================================================

number_of_users = ratings_data["User-ID"].nunique()
number_of_books = ratings_data["ISBN"].nunique()
number_of_ratings = len(ratings_data)

possible_ratings = number_of_users * number_of_books
matrix_sparsity = 1 - (number_of_ratings / possible_ratings)

print("\n============================================================")
print("DATASET ANALYSIS")
print("============================================================")

print("Number of Users:", number_of_users)
print("Number of Books:", number_of_books)
print("Number of Ratings:", number_of_ratings)
print("Average Rating:", round(ratings_data["Book-Rating"].mean(), 2))
print("Minimum Rating:", ratings_data["Book-Rating"].min())
print("Maximum Rating:", ratings_data["Book-Rating"].max())
print("User-Item Matrix Sparsity:", f"{matrix_sparsity:.2%}")


# ============================================================
# 5. RATING DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
ratings_data["Book-Rating"].hist(bins=10, edgecolor="black")
plt.title("Distribution of User Book Ratings")
plt.xlabel("Book Rating")
plt.ylabel("Number of Ratings")
plt.xticks(range(1, 11))
plt.tight_layout()
plt.show()


# ============================================================
# 6. TOP 10 MOST-RATED BOOKS
# ============================================================

top_books = data.groupby("Book-Title")["Book-Rating"].count().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_books.sort_values().plot(kind="barh")
plt.title("Top 10 Most-Rated Books")
plt.xlabel("Number of Ratings")
plt.ylabel("Book Title")
plt.tight_layout()
plt.show()


# ============================================================
# 7. METHOD 1: POPULARITY-BASED RECOMMENDATION
# ============================================================

# Calculate average rating and number of ratings
book_rating = data.groupby(["ISBN", "Book-Title", "Book-Author"])["Book-Rating"].agg(["mean", "count"]).reset_index()

book_rating.columns = ["ISBN", "Book-Title", "Book-Author", "rating_average", "rating_count"]

# Overall average rating
C = book_rating["rating_average"].mean()

# Require at least 50 ratings
m = max(50, book_rating["rating_count"].quantile(0.90))

print("\nOverall Mean Rating:", round(C, 4))
print("Minimum Required Ratings:", int(m))


# Weighted Rating Function
def weighted_rating(x, m=m, C=C):

    v = x["rating_count"]
    R = x["rating_average"]

    return (v / (v + m) * R) + (m / (m + v) * C)


# Calculate weighted rating for all books
book_rating["weighted_rating"] = book_rating.apply(weighted_rating, axis=1)

# Keep qualified books for Popularity-Based recommendation
q_books = book_rating[book_rating["rating_count"] >= m].copy()

# Sort by weighted rating
q_books = q_books.sort_values("weighted_rating", ascending=False)


def recommend_popular(top_n=10):

    result = q_books[["Book-Title", "Book-Author", "rating_count", "rating_average", "weighted_rating"]].head(top_n).copy()

    result["rating_average"] = result["rating_average"].round(3)
    result["weighted_rating"] = result["weighted_rating"].round(3)

    result.index = range(1, len(result) + 1)

    return result


# ============================================================
# 8. METHOD 2: CONTENT-BASED FILTERING
# ============================================================

# Select useful book information
metadata = books[["ISBN", "Book-Title", "Book-Author", "Publisher"]].copy()

# Remove duplicate titles
metadata = metadata.drop_duplicates(subset=["Book-Title"]).reset_index(drop=True)

# Combine title, author and publisher
metadata["content"] = (
    metadata["Book-Title"].fillna("") + " " +
    metadata["Book-Author"].fillna("") + " " +
    metadata["Publisher"].fillna("")
)

# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_features=15000)

# Convert book information into vectors
tfidf_matrix = tfidf.fit_transform(metadata["content"])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Create Book Title to Index mapping
indices = pd.Series(metadata.index, index=metadata["Book-Title"]).drop_duplicates()


def find_book(title):

    # Exact match
    exact_match = metadata[metadata["Book-Title"].str.lower() == title.lower()]

    if not exact_match.empty:
        return exact_match.iloc[0]["Book-Title"]

    # Partial match
    partial_match = metadata[metadata["Book-Title"].str.contains(title, case=False, regex=False)]

    if not partial_match.empty:
        return partial_match.iloc[0]["Book-Title"]

    return None


def get_recommendations(title, top_n=10):

    if title not in indices:
        return pd.DataFrame()

    # Get selected book index
    idx = indices[title]

    # Calculate similarity between selected book and all books
    cosine_scores = linear_kernel(tfidf_matrix[idx:idx + 1], tfidf_matrix).flatten()

    # Sort similarity from highest to lowest
    similar_indices = cosine_scores.argsort()[::-1]

    # Remove selected book itself
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    # Get recommendations
    result = metadata[["Book-Title", "Book-Author"]].iloc[similar_indices].copy()

    # Add similarity score
    result["Similarity"] = cosine_scores[similar_indices]

    result["Similarity"] = result["Similarity"].round(3)

    result.index = range(1, len(result) + 1)

    return result


# ============================================================
# 9. PREPARE COLLABORATIVE FILTERING DATA
# ============================================================

# Select ISBN and Book Title
book_titles = books[["ISBN", "Book-Title"]]

# Merge rating data and book titles
collaborative_data = pd.merge(ratings_data, book_titles, on="ISBN")

# Calculate average rating of each book
ratings = pd.DataFrame(collaborative_data.groupby("Book-Title")["Book-Rating"].mean())

# Rename column
ratings.columns = ["rating"]

# Calculate number of ratings
ratings["num of ratings"] = collaborative_data.groupby("Book-Title")["Book-Rating"].count()

print("\n============================================================")
print("BOOK RATING STATISTICS")
print("============================================================")

print(ratings.sort_values("num of ratings", ascending=False).head(10))


# ============================================================
# 10. COLLABORATIVE FILTERING DATA ANALYSIS
# ============================================================

plt.figure(figsize=(10, 4))
ratings["num of ratings"].hist(bins=60)
plt.title("Distribution of Number of Book Ratings")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
ratings["rating"].hist(bins=20)
plt.title("Distribution of Average Book Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.show()


# ============================================================
# 11. METHOD 3: COLLABORATIVE FILTERING
# ============================================================

# Count ratings given by each user
user_counts = collaborative_data["User-ID"].value_counts()

# Keep active users with at least 5 ratings
active_users = user_counts[user_counts >= 5].index

# Count ratings received by every book
book_counts = collaborative_data["Book-Title"].value_counts()

# Keep books with at least 20 ratings
popular_titles = book_counts[book_counts >= 20].index

# Filter data
filtered_data = collaborative_data[
    collaborative_data["User-ID"].isin(active_users) &
    collaborative_data["Book-Title"].isin(popular_titles)
].copy()

print("\nFiltered Collaborative Data:", filtered_data.shape)

# Build User-Item Matrix
bookmat = filtered_data.pivot_table(index="User-ID", columns="Book-Title", values="Book-Rating")

print("User-Item Matrix:", bookmat.shape)


# Minimum common users required
MIN_COMMON_USERS = 10


def recommend_collaborative(title, top_n=10):

    # Check book exists
    if title not in bookmat.columns:
        return pd.DataFrame()

    # Ratings of selected book
    selected_book = bookmat[title]

    recommendation_list = []

    # Compare with every other book
    for other_book in bookmat.columns:

        if other_book == title:
            continue

        # Keep users who rated both books
        comparison = pd.concat([selected_book, bookmat[other_book]], axis=1).dropna()

        # Require at least 10 common users
        if len(comparison) < MIN_COMMON_USERS:
            continue

        # Calculate correlation
        correlation = comparison.iloc[:, 0].corr(comparison.iloc[:, 1])

        # Skip invalid correlation
        if pd.isna(correlation):
            continue

        recommendation_list.append({
            "Book-Title": other_book,
            "Correlation": correlation,
            "Common Users": len(comparison),
            "Number of Ratings": ratings.loc[other_book, "num of ratings"]
        })

    result = pd.DataFrame(recommendation_list)

    if result.empty:
        return result

    # Keep books with enough ratings
    result = result[result["Number of Ratings"] >= 20]

    # Sort by correlation and common users
    result = result.sort_values(["Correlation", "Common Users"], ascending=[False, False])

    # Get Top N
    result = result.head(top_n).copy()

    # Round correlation
    result["Correlation"] = result["Correlation"].round(3)

    result.index = range(1, len(result) + 1)

    return result


# ============================================================
# 12. SELECT EXAMPLE BOOK
# ============================================================

preferred_books = [
    "Harry Potter and the Order of the Phoenix (Book 5)",
    "Harry Potter and the Chamber of Secrets (Book 2)",
    "The Da Vinci Code",
    "The Lovely Bones: A Novel"
]

example_book = None

# Find preferred book that exists in both models
for title in preferred_books:

    if title in indices.index and title in bookmat.columns:
        example_book = title
        break

# Use another common book if preferred books are not found
if example_book is None:

    common_books = [title for title in bookmat.columns if title in indices.index]

    if len(common_books) > 0:
        example_book = common_books[0]
    else:
        raise ValueError("No common book found for recommendation.")

print("\nSelected Example Book:", example_book)


# ============================================================
# 13. METHOD 4: HYBRID RECOMMENDATION
# ============================================================

# Hybrid Weights
CONTENT_WEIGHT = 0.40
COLLABORATIVE_WEIGHT = 0.45
POPULARITY_WEIGHT = 0.15


def recommend_hybrid(title, top_n=10):

    # Get more candidates from each method
    content_result = get_recommendations(title, top_n=30)
    collaborative_result = recommend_collaborative(title, top_n=30)

    # Popularity candidates
    popularity_result = q_books[["Book-Title", "weighted_rating"]].head(30).copy()

    # Content-Based scores
    content_scores = {}

    if not content_result.empty:
        content_scores = content_result.set_index("Book-Title")["Similarity"].to_dict()

    # Collaborative Filtering scores
    collaborative_scores = {}

    if not collaborative_result.empty:
        collaborative_scores = collaborative_result.set_index("Book-Title")["Correlation"].to_dict()

    # Popularity scores
    popularity_scores = book_rating.set_index("Book-Title")["weighted_rating"].to_dict()

    # Combine candidate books
    candidate_titles = set(content_scores.keys())
    candidate_titles.update(collaborative_scores.keys())
    candidate_titles.update(popularity_result["Book-Title"].tolist())

    hybrid = pd.DataFrame({"Book-Title": list(candidate_titles)})

    # Add Content-Based score
    hybrid["Content"] = hybrid["Book-Title"].map(content_scores).fillna(0)

    # Add Collaborative score
    hybrid["Collaborative"] = hybrid["Book-Title"].map(collaborative_scores)

    # Convert correlation from -1 to 1 into 0 to 1
    hybrid["Collaborative"] = (hybrid["Collaborative"] + 1) / 2

    # Use neutral 0.5 when collaborative score is unavailable
    hybrid["Collaborative"] = hybrid["Collaborative"].fillna(0.5)

    # Add Popularity score
    hybrid["Popularity"] = hybrid["Book-Title"].map(popularity_scores).fillna(C)

    # Convert popularity score from 1-10 into 0-1
    hybrid["Popularity"] = hybrid["Popularity"] / 10

    # Calculate final Hybrid Score
    hybrid["Hybrid Score"] = (
        CONTENT_WEIGHT * hybrid["Content"] +
        COLLABORATIVE_WEIGHT * hybrid["Collaborative"] +
        POPULARITY_WEIGHT * hybrid["Popularity"]
    )

    # Add author information
    author_data = books[["Book-Title", "Book-Author"]].drop_duplicates(subset=["Book-Title"])

    hybrid = hybrid.merge(author_data, on="Book-Title", how="left")

    # Remove selected book
    hybrid = hybrid[hybrid["Book-Title"] != title]

    # Sort by final Hybrid Score
    hybrid = hybrid.sort_values("Hybrid Score", ascending=False)

    # Round scores
    hybrid["Content"] = hybrid["Content"].round(3)
    hybrid["Collaborative"] = hybrid["Collaborative"].round(3)
    hybrid["Popularity"] = hybrid["Popularity"].round(3)
    hybrid["Hybrid Score"] = hybrid["Hybrid Score"].round(3)

    # Select columns
    result = hybrid[
        [
            "Book-Title",
            "Book-Author",
            "Content",
            "Collaborative",
            "Popularity",
            "Hybrid Score"
        ]
    ].head(top_n).copy()

    result.index = range(1, len(result) + 1)

    return result



# ============================================================
# 14. STREAMLIT USER INTERFACE
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #666;
        font-size: 17px;
        margin-bottom: 25px;
    }
    .method-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📚 Book Recommender System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Explore books using Popularity-Based, Content-Based, '
    'Collaborative, and Hybrid Recommendation methods.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Controls")

common_books = sorted(set(indices.index).intersection(set(bookmat.columns)))

if not common_books:
    st.error("No common books are available between the content and collaborative models.")
    st.stop()

default_book = example_book if example_book in common_books else common_books[0]

selected_book = st.sidebar.selectbox(
    "Select a book",
    common_books,
    index=common_books.index(default_book)
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Weights")
st.sidebar.write("Content-Based: **40%**")
st.sidebar.write("Collaborative: **45%**")
st.sidebar.write("Popularity-Based: **15%**")

# -----------------------------
# Dashboard metrics
# -----------------------------
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Users", f"{number_of_users:,}")
col2.metric("Books", f"{number_of_books:,}")
col3.metric("Ratings", f"{number_of_ratings:,}")
col4.metric("Average Rating", f"{ratings_data['Book-Rating'].mean():.2f}/10")

st.info(
    f"**Selected book:** {selected_book}  \n"
    f"**User-Item Matrix Sparsity:** {matrix_sparsity:.2%}"
)

# -----------------------------
# Charts
# -----------------------------
st.header("📈 Dataset Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Rating Distribution")
    rating_counts = ratings_data["Book-Rating"].value_counts().sort_index()
    st.bar_chart(rating_counts)

with chart_col2:
    st.subheader("Top 10 Most-Rated Books")
    st.bar_chart(top_books.sort_values(ascending=True))

# -----------------------------
# Recommendation methods
# -----------------------------
st.header("🔎 Recommendations")

tab1, tab2, tab3, tab4 = st.tabs([
    "⭐ Popularity-Based",
    "📖 Content-Based",
    "👥 Collaborative",
    "🔥 Hybrid"
])

with tab1:
    st.subheader("Popularity-Based Recommendation")
    st.caption("Ranks books using weighted rating, considering both rating quality and number of ratings.")

    popular_result = recommend_popular(top_n)

    if popular_result.empty:
        st.warning("No popularity-based recommendations found.")
    else:
        display_popular = popular_result.rename(columns={
            "Book-Title": "Book Title",
            "Book-Author": "Author",
            "rating_count": "Ratings",
            "rating_average": "Average Rating",
            "weighted_rating": "Weighted Rating"
        })
        st.dataframe(display_popular, use_container_width=True)

with tab2:
    st.subheader("Content-Based Filtering")
    st.caption("Finds books with similar title, author, and publisher information using TF-IDF and cosine similarity.")

    content_result = get_recommendations(selected_book, top_n)

    if content_result.empty:
        st.warning("No content-based recommendations found for this book.")
    else:
        display_content = content_result.rename(columns={
            "Book-Title": "Book Title",
            "Book-Author": "Author",
            "Similarity": "Similarity Score"
        })
        st.dataframe(display_content, use_container_width=True)

with tab3:
    st.subheader("Collaborative Filtering")
    st.caption("Recommends books based on rating patterns from users who rated the selected book similarly.")

    collaborative_result = recommend_collaborative(selected_book, top_n)

    if collaborative_result.empty:
        st.warning(
            "No collaborative recommendations found. "
            "The model requires enough common users and ratings."
        )
    else:
        display_collab = collaborative_result.rename(columns={
            "Book-Title": "Book Title",
            "Correlation": "Correlation",
            "Common Users": "Common Users",
            "Number of Ratings": "Number of Ratings"
        })
        st.dataframe(display_collab, use_container_width=True)

with tab4:
    st.subheader("Hybrid Recommendation")
    st.caption("Combines Content-Based, Collaborative, and Popularity scores.")

    hybrid_result = recommend_hybrid(selected_book, top_n)

    if hybrid_result.empty:
        st.warning("No hybrid recommendations found.")
    else:
        display_hybrid = hybrid_result.rename(columns={
            "Book-Title": "Book Title",
            "Book-Author": "Author",
            "Content": "Content Score",
            "Collaborative": "Collaborative Score",
            "Popularity": "Popularity Score",
            "Hybrid Score": "Final Hybrid Score"
        })
        st.dataframe(display_hybrid, use_container_width=True)

        st.subheader("Hybrid Score Breakdown")
        chart_data = display_hybrid.set_index("Book Title")[
            ["Content Score", "Collaborative Score", "Popularity Score"]
        ]
        st.bar_chart(chart_data)

# -----------------------------
# Selected book details
# -----------------------------
st.header("📘 Selected Book Information")

book_info = books[books["Book-Title"] == selected_book][
    ["Book-Title", "Book-Author", "Publisher", "ISBN"]
].drop_duplicates()

if not book_info.empty:
    st.dataframe(
        book_info.rename(columns={
            "Book-Title": "Book Title",
            "Book-Author": "Author"
        }),
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# Method explanation
# -----------------------------
with st.expander("ℹ️ How the recommendation system works"):
    st.markdown("""
    **1. Popularity-Based Recommendation**  
    Uses average rating and rating count to calculate a weighted rating.

    **2. Content-Based Filtering**  
    Uses TF-IDF on book title, author, and publisher, then calculates cosine similarity.

    **3. Collaborative Filtering**  
    Compares rating patterns between books using users who rated both books.

    **4. Hybrid Recommendation**  
    Combines the three methods using:
    - Content-Based = **40%**
    - Collaborative = **45%**
    - Popularity-Based = **15%**
    """)

st.success("✅ Book Recommender System is ready!")