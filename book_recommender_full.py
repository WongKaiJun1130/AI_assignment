# ============================================================
# BOOK RECOMMENDER SYSTEM - STREAMLIT UI
# ============================================================

import math
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

warnings.filterwarnings("ignore")


# ============================================================
# 1. APP SETTINGS
# ============================================================

st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide"
)

UI_SAMPLE_SIZE = 10000
TEST_SAMPLE_SIZE = 1000
RANDOM_STATE = 42
BOOKS_PER_PAGE = 20

CONTENT_WEIGHT = 0.40
COLLABORATIVE_WEIGHT = 0.45
POPULARITY_WEIGHT = 0.15

MIN_COMMON_USERS = 10

EVALUATION_K = 10
EVALUATION_SAMPLE_SIZE = 20
RELEVANT_RATING_THRESHOLD = 8


# ============================================================
# 2. LOAD DATASET + BUILD MODELS
# ============================================================

@st.cache_resource(show_spinner="Preparing Book Recommender System...")
def build_system():

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    books = pd.read_csv(
        "data/Books.csv",
        low_memory=False
    )

    ratings_data = pd.read_csv( "data/Ratings.csv", low_memory=False )

    users = pd.read_csv( "data/Users.csv", low_memory=False )


    # --------------------------------------------------------
    # Data Preprocessing
    # --------------------------------------------------------

    books = books.dropna(
        subset=[
            "ISBN",
            "Book-Title"
        ]
    ).copy()


    ratings_data = ratings_data.dropna( subset=[ "User-ID", "ISBN", "Book-Rating" ] ).copy()

    books[ "Book-Author" ] = books[ "Book-Author" ].fillna( "Unknown" )

    books[ "Publisher" ] = books[ "Publisher" ].fillna( "Unknown" )

    ratings_data[ "Book-Rating" ] = pd.to_numeric( ratings_data[ "Book-Rating" ], errors="coerce" )

    ratings_data = ratings_data.dropna( subset=[ "Book-Rating" ] )


    # Keep explicit ratings from 1 to 10
    ratings_data = ratings_data[
        (
            ratings_data[
                "Book-Rating"
            ] >= 1
        )
        &
        (
            ratings_data[
                "Book-Rating"
            ] <= 10
        )
    ].copy()


    # Remove duplicate books
    books = books.drop_duplicates(
        subset=[
            "ISBN"
        ]
    ).reset_index( drop=True)


    # Remove duplicate user-book rating
    ratings_data = ratings_data.drop_duplicates(
        subset=[
            "User-ID",
            "ISBN"
        ]
    ).reset_index( drop=True)

    # Merge Ratings + Books
    data = pd.merge(ratings_data, books, on="ISBN", how="inner")

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    number_of_users = ratings_data[
        "User-ID"
    ].nunique()


    number_of_books = ratings_data[ "ISBN" ].nunique()


    number_of_ratings = len( ratings_data )

    cleaned_books_count = len( books )
    merged_records_count = len( data )
    average_rating = ratings_data[ "Book-Rating" ].mean()
    minimum_rating = ratings_data[ "Book-Rating" ].min()
    maximum_rating = ratings_data[ "Book-Rating" ].max()

    possible_ratings = ( number_of_users * number_of_books )


    matrix_sparsity = 1 - ( number_of_ratings / possible_ratings )


    # ========================================================
    # METHOD 1: POPULARITY-BASED
    # ========================================================

    book_rating = data.groupby(
        [
            "ISBN",
            "Book-Title",
            "Book-Author"
        ]
    )[
        "Book-Rating"
    ].agg(
        [
            "mean",
            "count"
        ]
    ).reset_index()


    book_rating.columns = [ "ISBN", "Book-Title", "Book-Author", "rating_average", "rating_count" ]


    # Overall average
    C = book_rating[
        "rating_average"
    ].mean()


    # Minimum 50 ratings
    m = max(
        50,
        book_rating[
            "rating_count"
        ].quantile(
            0.90
        )
    )


    def weighted_rating(row):

        v = row[ "rating_count" ]

        R = row[ "rating_average" ]

        return ( (v / (v + m) * R) + (m / (m + v) * C) )


    book_rating[ "weighted_rating" ] = book_rating.apply( weighted_rating, axis=1 )


    q_books = book_rating[ book_rating[ "rating_count" ] >= m ].copy()


    q_books = q_books.sort_values( "weighted_rating", ascending=False )


    # ========================================================
    # METHOD 2: CONTENT-BASED
    # ========================================================

    metadata = books[
        [
            "ISBN",
            "Book-Title",
            "Book-Author",
            "Publisher"
        ]
    ].copy()


    metadata = metadata.drop_duplicates( subset=[ "Book-Title" ] ).reset_index( drop=True )


    metadata[
        "content"
    ] = (
        metadata[
            "Book-Title"
        ].fillna("")
        +
        " "
        +
        metadata[
            "Book-Author"
        ].fillna("")
        +
        " "
        +
        metadata[
            "Publisher"
        ].fillna("")
    )


    tfidf = TfidfVectorizer( stop_words="english", max_features=15000 )


    tfidf_matrix = tfidf.fit_transform( metadata[ "content" ] )


    indices = pd.Series( metadata.index, index=metadata[ "Book-Title" ] ).drop_duplicates()


    # ========================================================
    # METHOD 3: COLLABORATIVE FILTERING
    # ========================================================

    collaborative_data = pd.merge(
        ratings_data,
        books[
            [
                "ISBN",
                "Book-Title"
            ]
        ],
        on="ISBN"
    )


    rating_stats = pd.DataFrame( collaborative_data.groupby( "Book-Title" )[ "Book-Rating" ].mean() )


    rating_stats.columns = [ "rating" ]


    rating_stats[ "num of ratings" ] = collaborative_data.groupby( "Book-Title" )[ "Book-Rating" ].count()


    # Active users
    user_counts = collaborative_data[
        "User-ID"
    ].value_counts()


    active_users = user_counts[ user_counts >= 5 ].index


    # Books with minimum 20 ratings
    book_counts = collaborative_data[
        "Book-Title"
    ].value_counts()


    popular_titles = book_counts[ book_counts >= 20 ].index


    filtered_data = collaborative_data[
        collaborative_data[
            "User-ID"
        ].isin(
            active_users
        )
        &
        collaborative_data[
            "Book-Title"
        ].isin(
            popular_titles
        )
    ].copy()


    # User-Item Matrix
    bookmat = filtered_data.pivot_table(
        index="User-ID",
        columns="Book-Title",
        values="Book-Rating"
    )


    # ========================================================
    # COMMON BOOKS
    # ========================================================

    common_titles = sorted(
        set(
            indices.index
        ).intersection(
            set(
                bookmat.columns
            )
        )
    )


    preferred_books = [
        "Harry Potter and the Order of the Phoenix (Book 5)",
        "Harry Potter and the Chamber of Secrets (Book 2)",
        "The Da Vinci Code",
        "The Lovely Bones: A Novel"
    ]


    example_book = None


    for title in preferred_books:

        if title in common_titles:

            example_book = title

            break


    if example_book is None:

        if common_titles:

            example_book = common_titles[ 0 ]

        else:

            raise ValueError( "No common book found." )


    # ========================================================
    # UI DATASET - RANDOM 10,000 BOOKS
    # ========================================================

    display_columns = [
        "ISBN",
        "Book-Title",
        "Book-Author",
        "Publisher"
    ]

    if "Year-Of-Publication" in books.columns:
        display_columns.append( "Year-Of-Publication" )

    for image_column in [ "Image-URL-M", "Image-URL-L", "Image-URL-S" ]:
        if image_column in books.columns:
            display_columns.append( image_column )
            break

    # Use unique rated books for the user catalogue.
    catalog_pool = books[ display_columns ].drop_duplicates( subset=[ "Book-Title" ] ).copy()
    catalog_pool = catalog_pool.merge( rating_stats.reset_index(), on="Book-Title", how="left" )
    catalog_pool = catalog_pool[ catalog_pool[ "rating" ].notna() ].reset_index( drop=True )

    ui_n = min( UI_SAMPLE_SIZE, len( catalog_pool ) )

    # ========================================================
    # RANDOM 1,000 TESTING BOOKS
    # ========================================================

    # Testing books must be supported by Content-Based and Collaborative methods.
    test_pool = catalog_pool[ catalog_pool[ "Book-Title" ].isin( common_titles ) ].copy()
    test_n = min( TEST_SAMPLE_SIZE, len( test_pool ), ui_n )
    test_books = test_pool.sample( n=test_n, random_state=RANDOM_STATE + 1 ).reset_index( drop=True )

    # Keep the 1,000 testing books inside the 10,000 UI catalogue.
    test_titles = set( test_books[ "Book-Title" ] )
    remaining_pool = catalog_pool[ ~catalog_pool[ "Book-Title" ].isin( test_titles ) ].copy()
    remaining_n = max( 0, ui_n - test_n )

    if remaining_n > 0:
        remaining_books = remaining_pool.sample(
            n=min( remaining_n, len( remaining_pool ) ),
            random_state=RANDOM_STATE
        )

        ui_books = pd.concat( [ test_books, remaining_books ], ignore_index=True )

    else:
        ui_books = test_books.copy()

    ui_books = ui_books.drop_duplicates( subset=[ "Book-Title" ] ).reset_index( drop=True )

    # Keep the example book inside the UI catalogue.
    if (
        example_book not in set( ui_books[ "Book-Title" ] )
        and example_book in set( catalog_pool[ "Book-Title" ] )
        and len( ui_books ) > 0
    ):
        example_row = catalog_pool[ catalog_pool[ "Book-Title" ] == example_book ].head( 1 )
        ui_books = pd.concat( [ example_row, ui_books.iloc[ :-1 ] ], ignore_index=True )


    # Book information lookup
    book_info = books.drop_duplicates(
        subset=[
            "Book-Title"
        ]
    ).set_index(
        "Book-Title"
    )


    return {

        "books":
            books,

        "ratings_data":
            ratings_data,

        "users":
            users,

        "data":
            data,

        "number_of_users":
            number_of_users,

        "number_of_books":
            number_of_books,

        "number_of_ratings":
            number_of_ratings,

        "cleaned_books_count":
            cleaned_books_count,

        "merged_records_count":
            merged_records_count,

        "average_rating":
            average_rating,

        "minimum_rating":
            minimum_rating,

        "maximum_rating":
            maximum_rating,

        "matrix_sparsity":
            matrix_sparsity,

        "book_rating":
            book_rating,

        "q_books":
            q_books,

        "C":
            C,

        "metadata":
            metadata,

        "tfidf_matrix":
            tfidf_matrix,

        "indices":
            indices,

        "rating_stats":
            rating_stats,

        "bookmat":
            bookmat,

        "example_book":
            example_book,

        "ui_books":
            ui_books,

        "test_books":
            test_books,

        "book_info":
            book_info
    }


# ============================================================
# BUILD SYSTEM
# ============================================================

system = build_system()


books = system[ "books" ]

ratings_data = system[ "ratings_data" ]

q_books = system[ "q_books" ]

book_rating = system[ "book_rating" ]

metadata = system[ "metadata" ]

tfidf_matrix = system[ "tfidf_matrix" ]

indices = system[ "indices" ]

rating_stats = system[ "rating_stats" ]

bookmat = system[ "bookmat" ]

C = system[ "C" ]

ui_books = system[ "ui_books" ]

test_books = system[ "test_books" ]

book_info = system[ "book_info" ]

example_book = system[ "example_book" ]


# ============================================================
# 3. METHOD 1 - POPULARITY
# ============================================================

@st.cache_data(show_spinner=False)
def recommend_popular( top_n=10 ):

    result = q_books[
        [
            "Book-Title",
            "Book-Author",
            "rating_count",
            "rating_average",
            "weighted_rating"
        ]
    ].head(
        top_n
    ).copy()


    result[ "rating_average" ] = result[ "rating_average" ].round( 3 )


    result[ "weighted_rating" ] = result[ "weighted_rating" ].round( 3 )


    result.index = range( 1, len( result ) + 1 )


    return result


# ============================================================
# METHOD 2 - CONTENT-BASED
# ============================================================

@st.cache_data(show_spinner=False)
def get_recommendations( title, top_n=10 ):

    if title not in indices:

        return pd.DataFrame()


    idx = indices[ title ]


    cosine_scores = linear_kernel( tfidf_matrix[ idx:idx + 1 ], tfidf_matrix ).flatten()


    similar_indices = cosine_scores.argsort()[ ::-1 ]


    similar_indices = [ i for i in similar_indices if i != idx ][ :top_n ]


    result = metadata[ [ "Book-Title", "Book-Author" ] ].iloc[ similar_indices ].copy()


    result[ "Similarity" ] = cosine_scores[ similar_indices ]


    result[ "Similarity" ] = result[ "Similarity" ].round( 3 )


    result.index = range( 1, len( result ) + 1 )


    return result


# ============================================================
# METHOD 3 - COLLABORATIVE
# ============================================================

@st.cache_data(show_spinner=False)
def recommend_collaborative( title, top_n=10 ):

    if title not in bookmat.columns:

        return pd.DataFrame()


    selected_book = bookmat[ title ]


    recommendation_list = []


    for other_book in bookmat.columns:


        if other_book == title:

            continue


        comparison = pd.concat( [ selected_book, bookmat[ other_book ] ], axis=1 ).dropna()


        if len( comparison ) < MIN_COMMON_USERS:

            continue


        correlation = comparison.iloc[ :, 0 ].corr( comparison.iloc[ :, 1 ] )


        if pd.isna( correlation ):

            continue


        recommendation_list.append(
            {

                "Book-Title":
                    other_book,

                "Correlation":
                    correlation,

                "Common Users":
                    len(
                        comparison
                    ),

                "Number of Ratings":
                    rating_stats.loc[
                        other_book,
                        "num of ratings"
                    ]
            }
        )


    result = pd.DataFrame( recommendation_list )


    if result.empty:

        return result


    result = result[ result[ "Number of Ratings" ] >= 20 ]


    result = result.sort_values( [ "Correlation", "Common Users" ], ascending=[ False, False ] )


    result = result.head( top_n ).copy()


    result[ "Correlation" ] = result[ "Correlation" ].round( 3 )


    result.index = range( 1, len( result ) + 1 )


    return result


# ============================================================
# METHOD 4 - HYBRID
# ============================================================

@st.cache_data(show_spinner=False)
def recommend_hybrid( title, top_n=10 ):

    content_result = get_recommendations( title, top_n=30 )


    collaborative_result = recommend_collaborative( title, top_n=30 )


    popularity_result = q_books[ [ "Book-Title", "weighted_rating" ] ].head( 30 ).copy()


    content_scores = {}

    collaborative_scores = {}


    if not content_result.empty:

        content_scores = content_result.set_index( "Book-Title" )[ "Similarity" ].to_dict()


    if not collaborative_result.empty:

        collaborative_scores = collaborative_result.set_index( "Book-Title" )[ "Correlation" ].to_dict()


    popularity_scores = book_rating.drop_duplicates(
        subset=[
            "Book-Title"
        ]
    ).set_index(
        "Book-Title"
    )[
        "weighted_rating"
    ].to_dict()


    candidate_titles = set( content_scores.keys() )


    candidate_titles.update( collaborative_scores.keys() )


    candidate_titles.update( popularity_result[ "Book-Title" ].tolist() )


    hybrid = pd.DataFrame( { "Book-Title": list( candidate_titles ) } )


    # Content score
    hybrid[
        "Content"
    ] = hybrid[
        "Book-Title"
    ].map(
        content_scores
    ).fillna(
        0
    )


    # Collaborative score
    hybrid[
        "Collaborative"
    ] = hybrid[
        "Book-Title"
    ].map(
        collaborative_scores
    )


    hybrid[ "Collaborative" ] = ( hybrid[ "Collaborative" ] + 1 ) / 2


    hybrid[ "Collaborative" ] = hybrid[ "Collaborative" ].fillna( 0.5 )


    # Popularity score
    hybrid[
        "Popularity"
    ] = hybrid[
        "Book-Title"
    ].map(
        popularity_scores
    ).fillna(
        C
    ) / 10


    # Final score
    hybrid[
        "Hybrid Score"
    ] = (
        CONTENT_WEIGHT
        *
        hybrid[
            "Content"
        ]
        +
        COLLABORATIVE_WEIGHT
        *
        hybrid[
            "Collaborative"
        ]
        +
        POPULARITY_WEIGHT
        *
        hybrid[
            "Popularity"
        ]
    )


    author_data = books[ [ "Book-Title", "Book-Author" ] ].drop_duplicates( subset=[ "Book-Title" ] )


    hybrid = hybrid.merge( author_data, on="Book-Title", how="left" )


    hybrid = hybrid[ hybrid[ "Book-Title" ] != title ]


    hybrid = hybrid.sort_values( "Hybrid Score", ascending=False )


    for column in [ "Content", "Collaborative", "Popularity", "Hybrid Score" ]:

        hybrid[ column ] = hybrid[ column ].round( 3 )


    result = hybrid[
        [
            "Book-Title",
            "Book-Author",
            "Content",
            "Collaborative",
            "Popularity",
            "Hybrid Score"
        ]
    ].head(
        top_n
    ).copy()


    result.index = range( 1, len( result ) + 1 )


    return result


# ============================================================
# 4. EVALUATION - PRECISION@10, RECALL@10, F1@10
# ============================================================

@st.cache_data(show_spinner=False)
def build_evaluation_ground_truth():
    positive_data = system["data"][ system["data"]["Book-Rating"] >= RELEVANT_RATING_THRESHOLD ][ [ "User-ID", "Book-Title" ] ].drop_duplicates()
    eligible_titles = set( bookmat.columns ).intersection( set( indices.index ) )
    positive_data = positive_data[ positive_data["Book-Title"].isin( eligible_titles ) ].copy()

    user_positive_books = positive_data.groupby( "User-ID" )[ "Book-Title" ].apply( list ).to_dict()
    book_positive_users = positive_data.groupby( "Book-Title" )[ "User-ID" ].apply( list ).to_dict()

    ground_truth = {}

    for query_title in test_books[ "Book-Title" ].drop_duplicates().tolist():
        positive_users = book_positive_users.get( query_title, [] )

        if len( positive_users ) < 2:
            continue

        co_like_counts = {}

        for user_id in positive_users:
            for other_title in user_positive_books.get( user_id, [] ):
                if other_title != query_title:
                    co_like_counts[ other_title ] = co_like_counts.get( other_title, 0 ) + 1

        ranked_relevant = [
            title
            for title, count in sorted( co_like_counts.items(), key=lambda item: ( -item[1], item[0] ) )
            if count >= 2
        ][ :EVALUATION_K ]

        if ranked_relevant:
            ground_truth[ query_title ] = set( ranked_relevant )

    return ground_truth


def get_evaluation_recommendations(method_name, title, top_n=10):
    if method_name == "Popularity-Based":
        result = recommend_popular( top_n + 1 )
    elif method_name == "Content-Based":
        result = get_recommendations( title, top_n )
    elif method_name == "Collaborative":
        result = recommend_collaborative( title, top_n )
    else:
        result = recommend_hybrid( title, top_n )

    if result.empty:
        return []

    recommended_titles = result[ "Book-Title" ].astype(str).tolist()
    recommended_titles = [ book_title for book_title in recommended_titles if book_title != title ]

    return recommended_titles[ :top_n ]


@st.cache_data(show_spinner=False)
def evaluate_recommender_system(sample_size=EVALUATION_SAMPLE_SIZE, top_n=EVALUATION_K):
    ground_truth = build_evaluation_ground_truth()
    query_titles = list( ground_truth.keys() )[ :sample_size ]

    methods = [ "Popularity-Based", "Content-Based", "Collaborative", "Hybrid" ]
    evaluation_rows = []

    for method_name in methods:
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for query_title in query_titles:
            relevant_books = ground_truth[ query_title ]
            recommended_books = get_evaluation_recommendations( method_name, query_title, top_n )

            hits = len( set( recommended_books ).intersection( relevant_books ) )
            precision = hits / top_n
            recall = hits / len( relevant_books ) if relevant_books else 0.0
            f1 = ( 2 * precision * recall / ( precision + recall ) ) if ( precision + recall ) > 0 else 0.0

            precision_scores.append( precision )
            recall_scores.append( recall )
            f1_scores.append( f1 )

        evaluation_rows.append(
            {
                "Method": method_name,
                "Precision@10": round( float( np.mean( precision_scores ) ), 4 ) if precision_scores else 0.0,
                "Recall@10": round( float( np.mean( recall_scores ) ), 4 ) if recall_scores else 0.0,
                "F1@10": round( float( np.mean( f1_scores ) ), 4 ) if f1_scores else 0.0,
                "Evaluated Queries": len( query_titles )
            }
        )

    return pd.DataFrame( evaluation_rows )


# ============================================================
# 5. BOOK + RATING HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def check_image_url(url):
    try:
        response = requests.get(
            url,
            timeout=5,
            stream=True,
            headers={
                "User-Agent": "BookRecommenderSystem/1.0"
            }
        )

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        return (
            response.status_code == 200
            and "image" in content_type
        )

    except requests.RequestException:
        return False


@st.cache_data(show_spinner=False)
def get_openlibrary_cover(isbn):
    if isbn is None:
        return None

    isbn = str(isbn).strip()

    isbn = re.sub(
        r"[^0-9Xx]",
        "",
        isbn
    )

    if not isbn:
        return None

    cover_url = (
        f"https://covers.openlibrary.org/"
        f"b/isbn/{isbn}-M.jpg?default=false"
    )

    if check_image_url(cover_url):
        return cover_url

    return None


def get_cover_url(title):
    if title not in book_info.index:
        return None

    row = book_info.loc[title]

    # --------------------------------------------------
    # 1. Try original Book-Crossing image URLs
    # --------------------------------------------------

    for column in [
        "Image-URL-L",
        "Image-URL-M",
        "Image-URL-S"
    ]:

        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        if value.startswith("http://"):
            value = "https://" + value[7:]

        if value.startswith("https://"):
            if check_image_url(value):
                return value

    # --------------------------------------------------
    # 2. If original cover is broken,
    #    try Open Library using ISBN
    # --------------------------------------------------

    isbn = row.get("ISBN")

    openlibrary_cover = get_openlibrary_cover(
        isbn
    )

    if openlibrary_cover:
        return openlibrary_cover

    return None


def show_cover(title, width=150):
    cover_url = get_cover_url(title)

    if cover_url:
        st.image(
            cover_url,
            width=width
        )

    else:
        st.markdown(
            """
            <div style="
                width:145px;
                height:205px;
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
                border:1px solid #cccccc;
                border-radius:6px;
                background:#f5f5f5;
                text-align:center;
            ">
                <div style="font-size:42px;">📕</div>
                <div style="
                    font-size:12px;
                    margin-top:8px;
                    color:#777777;
                ">
                    No Cover Available
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
def get_book_details(title):
    matches = books[ books["Book-Title"] == title ].copy()

    if matches.empty:
        return None

    row = matches.iloc[0]

    return {
        "ISBN": str(row.get("ISBN", "-")),
        "Book-Title": str(row.get("Book-Title", title)),
        "Book-Author": str(row.get("Book-Author", "Unknown")),
        "Publisher": str(row.get("Publisher", "Unknown")),
        "Year-Of-Publication": str(row.get("Year-Of-Publication", "-"))
    }


def get_book_rating_data(title):
    title_isbns = books.loc[ books["Book-Title"] == title, "ISBN" ].astype(str).unique()

    selected_ratings = ratings_data[ ratings_data["ISBN"].astype(str).isin(title_isbns) ][["User-ID", "Book-Rating"]].copy()

    if selected_ratings.empty:
        average_rating = 0.0
        total_users = 0
        most_common_rating = 0.0

        distribution = pd.Series( [0] * 10, index=range(1, 11), name="Users" )

    else:
        average_rating = float( selected_ratings["Book-Rating"].mean() )

        total_users = int( len(selected_ratings) )

        distribution = (
            selected_ratings["Book-Rating"]
            .value_counts()
            .reindex(
                range(1, 11),
                fill_value=0
            )
            .sort_index()
        )

        distribution.name = "Users"
        most_common_rating = float( distribution.idxmax() )

    return ( average_rating, total_users, most_common_rating, distribution, selected_ratings )


# ============================================================
# 5. SERIES GROUPING
# ============================================================

SERIES_STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "in",
    "on",
    "for",
    "to",
    "book",
    "volume",
    "vol",
    "part",
    "edition",
    "novel"
}


def clean_series_text(text):
    text = re.sub( r"[^A-Za-z0-9\s]", " ", str(text) )

    return " ".join( text.split() ).strip()


def get_title_keywords(title):
    clean_title = re.sub( r"\([^)]*\)", " ", str(title) )

    clean_title = clean_series_text( clean_title ).lower()

    words = []

    for word in clean_title.split():
        if word in SERIES_STOP_WORDS:
            continue

        if word.isdigit():
            continue

        words.append(word)

    return words


def get_parenthetical_series(title):
    groups = re.findall( r"\(([^)]*)\)", str(title) )

    for group in groups:
        cleaned = clean_series_text( group )

        # Example:
        # "The Lord of the Rings, Part 3"
        match = re.match(
            r"(.+?)(?:,|\s)+(?:book|part|volume|vol)\s*\d+\b",
            cleaned,
            flags=re.IGNORECASE
        )

        if match:
            candidate = match.group(1).strip()

            if len(candidate) >= 4:
                return candidate

    return None


def build_series_catalog(catalog):
    result = catalog.drop_duplicates( subset=["Book-Title"], keep="first" ).copy()

    result["_Author-Key"] = ( result["Book-Author"] .fillna("Unknown") .astype(str) .str.lower() .str.strip() )

    result["_Parent-Series"] = ( result["Book-Title"] .apply(get_parenthetical_series) .fillna("") .astype(str) )

    result["_Prefix-Key"] = result[ "Book-Title" ].apply( lambda title: " ".join( get_title_keywords(title)[:2] ) )

    # More reliable key if the series is explicitly written
    # in brackets, otherwise use same-author + first 2 meaningful
    # title words only when at least 2 books share it.
    result["_Candidate-Key"] = result.apply(
        lambda row:
            (
                row["_Author-Key"]
                + "|explicit|"
                + str(row["_Parent-Series"]).lower()
            )
            if row["_Parent-Series"]
            else
            (
                row["_Author-Key"]
                + "|prefix|"
                + row["_Prefix-Key"]
            ),
        axis=1
    )

    candidate_counts = result[ "_Candidate-Key" ].value_counts()

    def make_series_key(row):
        candidate_key = row[ "_Candidate-Key" ]

        prefix = row[ "_Prefix-Key" ]

        explicit = str( row["_Parent-Series"] ).strip()

        if explicit:
            return candidate_key

        if ( prefix and len(prefix) >= 6 and candidate_counts.get( candidate_key, 0 ) >= 2 ):
            return candidate_key

        return ( "single|" + str( row["Book-Title"] ).lower() )

    result["Series-Key"] = result.apply( make_series_key, axis=1 )

    def make_series_name(row):
        if str( row["Series-Key"] ).startswith( "single|" ):
            return row[ "Book-Title" ]

        parent_series = str( row["_Parent-Series"] ).strip()

        if parent_series:
            return ( parent_series + " Series" )

        prefix_words = row[ "_Prefix-Key" ].split()

        return ( " ".join( word.capitalize() for word in prefix_words ) + " Series" )

    result["Series-Name"] = result.apply( make_series_name, axis=1 )

    return result


catalog_with_series = build_series_catalog( ui_books )


def get_series_titles(title):
    matches = catalog_with_series[ catalog_with_series[ "Book-Title" ] == title ]

    if matches.empty:
        return [title], title

    series_key = matches.iloc[0][ "Series-Key" ]

    series_name = matches.iloc[0][ "Series-Name" ]

    titles = (
        catalog_with_series[
            catalog_with_series[
                "Series-Key"
            ] == series_key
        ]["Book-Title"]
        .drop_duplicates()
        .tolist()
    )

    return titles, series_name


# ============================================================
# 6. METHOD RESULT GRAPH
# ============================================================

def make_method_chart(
    method_name,
    selected_book,
    top_n=6
):
    if method_name == "Popularity-Based":
        result = recommend_popular( top_n )

        if result.empty:
            return pd.DataFrame()

        chart = result[ ["Book-Title", "weighted_rating"] ].copy()

        chart.columns = [ "Book", "Score" ]

    elif method_name == "Content-Based":
        result = get_recommendations( selected_book, top_n )

        if result.empty:
            return pd.DataFrame()

        chart = result[ ["Book-Title", "Similarity"] ].copy()

        chart.columns = [ "Book", "Score" ]

    elif method_name == "Collaborative":
        result = recommend_collaborative( selected_book, top_n )

        if result.empty:
            return pd.DataFrame()

        chart = result[ ["Book-Title", "Correlation"] ].copy()

        chart.columns = [ "Book", "Score" ]

    else:
        result = recommend_hybrid( selected_book, top_n )

        if result.empty:
            return pd.DataFrame()

        chart = result[ ["Book-Title", "Hybrid Score"] ].copy()

        chart.columns = [ "Book", "Score" ]

    chart["Book"] = chart[ "Book" ].apply( lambda title: title if len( str(title) ) <= 28 else str(title)[:25] + "..." )

    return chart.set_index( "Book" )


# ============================================================
# 7. FIVE DATASET ANALYSIS GRAPHS
# ============================================================

def make_dataset_figure(graph_name):
    fig, ax = plt.subplots( figsize=(6.5, 4.0) )

    if graph_name == "Distribution of User Book Ratings":
        rating_counts = ( ratings_data["Book-Rating"] .value_counts() .reindex( range(1, 11), fill_value=0 ) .sort_index() )

        ax.bar( rating_counts.index, rating_counts.values )

        ax.set_title( "Distribution of User Book Ratings" )

        ax.set_xlabel( "Book Rating" )

        ax.set_ylabel( "Number of Ratings" )

        ax.set_xticks( range(1, 11) )

        ax.grid( axis="y", alpha=0.35 )


    elif graph_name == "Top 10 Most-Rated Books":
        plt.close( fig )
        fig, ax = plt.subplots( figsize=(6.5, 3.2) )

        top_books = (
            system["data"]
            .groupby( "Book-Title" )[ "Book-Rating" ]
            .count()
            .sort_values( ascending=False )
            .head(10)
            .sort_values()
        )

        display_titles = [
            title if len( str(title) ) <= 38 else str(title)[:35] + "..."
            for title in top_books.index
        ]

        ax.barh( display_titles, top_books.values )

        ax.set_title( "Top 10 Most-Rated Books", fontsize=10, pad=6 )
        ax.set_xlabel( "Number of Ratings", fontsize=8 )
        ax.set_ylabel( "Book Title", fontsize=8 )

        ax.tick_params( axis="x", labelsize=7 )
        ax.tick_params( axis="y", labelsize=7 )

        ax.grid( axis="x", alpha=0.25 )

        fig.subplots_adjust(
            left=0.34,
            right=0.97,
            top=0.88,
            bottom=0.17
        )


    elif graph_name == "Distribution of Average Book Ratings":
        ax.hist( rating_stats[ "rating" ].dropna(), bins=20 )

        ax.set_title( "Distribution of Average Book Ratings" )

        ax.set_xlabel( "Average Rating" )

        ax.set_ylabel( "Number of Books" )

        ax.grid( axis="y", alpha=0.35 )


    elif graph_name == "Distribution of Number of Book Ratings":
        ax.hist( rating_stats[ "num of ratings" ].dropna(), bins=60 )

        ax.set_title( "Distribution of Number of Book Ratings" )

        ax.set_xlabel( "Number of Ratings" )

        ax.set_ylabel( "Number of Books" )

        ax.grid( axis="y", alpha=0.35 )


    elif graph_name == "Hybrid Recommendation Weights":
        methods = [ "Content-Based", "Collaborative", "Popularity-Based" ]

        weights = [ CONTENT_WEIGHT * 100, COLLABORATIVE_WEIGHT * 100, POPULARITY_WEIGHT * 100 ]

        ax.bar( methods, weights )

        ax.set_title( "Hybrid Recommendation Weights" )

        ax.set_xlabel( "Recommendation Method" )

        ax.set_ylabel( "Weight (%)" )

        ax.set_ylim( 0, 50 )

        ax.grid( axis="y", alpha=0.35 )

    if graph_name != "Top 10 Most-Rated Books":
        fig.tight_layout()

    return fig


# ============================================================
# 8. DATASET ANALYSIS RECORDS
# ============================================================

def get_dataset_analysis_records( graph_name ):

    if graph_name == "Distribution of User Book Ratings":
        rating_counts = ratings_data[ "Book-Rating" ].value_counts().reindex( range(1, 11), fill_value=0 ).sort_index()

        return pd.DataFrame(
            {
                "Book Rating": rating_counts.index,
                "Number of Ratings": rating_counts.values
            }
        )

    if graph_name == "Hybrid Recommendation Weights":
        return pd.DataFrame(
            {
                "Recommendation Method": [ "Content-Based", "Collaborative", "Popularity-Based" ],
                "Weight (%)": [ CONTENT_WEIGHT * 100, COLLABORATIVE_WEIGHT * 100, POPULARITY_WEIGHT * 100 ]
            }
        )

    if graph_name == "Distribution of Average Book Ratings":
        average_values = rating_stats[ "rating" ].dropna()
        bins = np.linspace( 1, 10, 11 )
        counts, edges = np.histogram( average_values, bins=bins )

        ranges = [
            f"{edges[index]:.1f} - {edges[index + 1]:.1f}"
            for index in range( len(counts) )
        ]

        return pd.DataFrame(
            {
                "Average Rating Range": ranges,
                "Number of Books": counts
            }
        )

    if graph_name == "Distribution of Number of Book Ratings":
        rating_count_values = rating_stats[ "num of ratings" ].dropna()

        bins = [ 0, 5, 10, 20, 50, 100, 200, 400, 800, np.inf ]
        labels = [ "1 - 5", "6 - 10", "11 - 20", "21 - 50", "51 - 100", "101 - 200", "201 - 400", "401 - 800", "801+" ]

        grouped = pd.cut(
            rating_count_values,
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True
        ).value_counts().reindex( labels, fill_value=0 )

        return pd.DataFrame(
            {
                "Number of Ratings Range": grouped.index.astype(str),
                "Number of Books": grouped.values
            }
        )

    if graph_name == "Top 10 Most-Rated Books":
        top_books = (
            system[ "data" ]
            .groupby( "Book-Title" )[ "Book-Rating" ]
            .count()
            .sort_values( ascending=False )
            .head(10)
            .reset_index()
        )

        top_books.columns = [ "Book Title", "Number of Ratings" ]

        return top_books

    return pd.DataFrame()


# ============================================================
# 8. USER RATING DASHBOARD
# ============================================================

def render_user_rating_dashboard(title):
    ( average_rating, total_users, most_common_rating, distribution, selected_ratings ) = get_book_rating_data( title )

    st.subheader( "User Rating Dashboard" )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------
    kpi1, kpi2, kpi3 = st.columns(
        3
    )

    kpi1.metric( "Average Rating", f"{average_rating:.2f} / 10.0" )

    kpi2.metric( "Users Who Rated", f"{total_users:,}" )

    kpi3.metric( "Most Common Rating", ( f"{most_common_rating:.1f} / 10.0" if total_users > 0 else "-" ) )

    # --------------------------------------------------------
    # RATING DISTRIBUTION + SUMMARY
    # --------------------------------------------------------
    graph_col, summary_col = st.columns(
        [2.1, 1.3],
        gap="large"
    )

    with graph_col:
        st.markdown( "#### How Many Users Gave Each Rating?" )

        rating_chart = ( distribution .rename_axis( "Rating" ) .reset_index() )

        st.bar_chart( rating_chart, x="Rating", y="Users", height=330 )

    with summary_col:
        st.markdown( "#### Rating Summary" )

        # Do NOT use custom div/span HTML here.
        # This avoids the raw HTML problem in the UI.
        rating_summary = pd.DataFrame(
            {
                "Rating": [
                    f"⭐ {rating}.0"
                    for rating in range(
                        10,
                        0,
                        -1
                    )
                ],

                "Users": [
                    int(
                        distribution.loc[
                            rating
                        ]
                    )
                    for rating in range(
                        10,
                        0,
                        -1
                    )
                ]
            }
        )

        if total_users > 0:
            rating_summary[ "Share" ] = ( rating_summary[ "Users" ] / total_users * 100 )

        else:
            rating_summary[ "Share" ] = 0.0

        st.dataframe(
            rating_summary,
            hide_index=True,
            width="stretch",
            height=390,
            column_config={
                "Rating":
                    st.column_config.TextColumn(
                        "Rating"
                    ),

                "Users":
                    st.column_config.NumberColumn(
                        "Users",
                        format="%d"
                    ),

                "Share":
                    st.column_config.ProgressColumn(
                        "Share",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%"
                    )
            }
        )

    # --------------------------------------------------------
    # INDIVIDUAL USER RATING RECORDS
    # --------------------------------------------------------
    st.markdown(
        "#### Individual User Rating Records"
    )

    st.caption( "Each row shows the exact score given by one user." )

    if selected_ratings.empty:
        st.info( "No explicit user rating records are available for this book." )

        return

    filter_col, record_col, information_col = st.columns( [1.1, 1.1, 2.5] )

    with filter_col:
        rating_filter = st.selectbox(
            "Rating Filter",
            options=[
                "All",
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1
            ],
            key=f"rating_filter_{title}"
        )

    with record_col:
        records_to_show = st.selectbox( "Show Records", options=[ 10, 20, 50 ], index=1, key=f"record_count_{title}" )

    filtered_records = selected_ratings.copy()

    if rating_filter != "All":
        filtered_records = filtered_records[ filtered_records[ "Book-Rating" ] == rating_filter ].copy()

    filtered_records = filtered_records.sort_values(
        [
            "Book-Rating",
            "User-ID"
        ],
        ascending=[
            False,
            True
        ]
    ).reset_index(
        drop=True
    )

    with information_col:
        st.caption( f"{len(filtered_records):,} matching user record(s)." )

    display_records = filtered_records.head( records_to_show ).copy()

    display_records[ "User" ] = display_records[ "User-ID" ].apply( lambda user_id: f"👤 User {user_id}" )

    display_records[
        "Rating Given"
    ] = display_records[
        "Book-Rating"
    ].apply(
        lambda rating:
            f"⭐ {float(rating):.1f} / 10.0"
    )

    display_records[ "Rating Level" ] = display_records[ "Book-Rating" ].astype(float)

    display_records = display_records[ [ "User", "Rating Given", "Rating Level" ] ]

    st.dataframe(
        display_records,
        hide_index=True,
        width="stretch",
        column_config={
            "User":
                st.column_config.TextColumn(
                    "User"
                ),

            "Rating Given":
                st.column_config.TextColumn(
                    "Rating Given"
                ),

            "Rating Level":
                st.column_config.ProgressColumn(
                    "Rating Level",
                    min_value=0,
                    max_value=10,
                    format="%.1f"
                )
        }
    )


# ============================================================
# 9. SELECTED BOOK DETAILS
# ============================================================

def render_selected_book(title):
    details = get_book_details( title )

    if details is None:
        st.warning( "Book information is not available." )

        return

    image_col, info_col = st.columns( [1, 3], gap="large" )

    with image_col:
        with st.container( border=True, key="selected_book_card" ):
            show_cover( title, width=190 )

    with info_col:
        st.markdown( f"## {details['Book-Title']}" )

        st.write( f"**Author:** " f"{details['Book-Author']}" )

        st.write( f"**Publisher:** " f"{details['Publisher']}" )

        st.write( f"**Publication Year:** " f"{details['Year-Of-Publication']}" )

        st.write( f"**ISBN:** " f"{details['ISBN']}" )

        # Same-series chooser
        series_titles, series_name = get_series_titles(
            title
        )

        if len( series_titles ) > 1:
            st.markdown( f"**📚 {series_name}**" )

            series_selected_title = st.selectbox(
                "Choose a book from this series",
                options=series_titles,
                index=series_titles.index(
                    title
                ),
                key=(
                    "series_detail_"
                    + str(
                        abs(
                            hash(
                                series_name
                            )
                        )
                    )
                )
            )

            if ( series_selected_title != title ):
                if st.button( "Open Selected Series Book", key=( "series_detail_button_" + str( abs( hash( title ) ) ) ) ):
                    st.session_state[ "selected_book" ] = series_selected_title

                    st.session_state[ "detail_book" ] = series_selected_title

                    st.rerun()

    render_user_rating_dashboard( title )


# ============================================================
# 10. RECOMMENDATION CARDS
# ============================================================

def render_recommendation_cards(
    result,
    method_name
):
    if result.empty:
        st.warning( "No recommendation result is available." )

        return

    for row_start in range( 0, len(result), 5 ):
        columns = st.columns( 5, gap="medium" )

        for offset in range(5):
            position = ( row_start + offset )

            if position >= len( result ):
                break

            row = result.iloc[ position ]

            title = row[ "Book-Title" ]

            author = row.get( "Book-Author", "Unknown" )

            with columns[offset]:
                with st.container( border=True, key=f"recommendation_card_{method_name}_{position}" ):
                    show_cover( title, width=125 )

                    st.markdown( f"**{position + 1}. " f"{title}**" )

                    st.caption( str(author) )

                    if method_name == "Popularity-Based":
                        st.write( f"⭐ Weighted: " f"{row['weighted_rating']:.3f}" )

                    elif method_name == "Content-Based":
                        st.write( f"🔎 Similarity: " f"{row['Similarity']:.3f}" )

                    elif method_name == "Collaborative":
                        st.write( f"👥 Correlation: " f"{row['Correlation']:.3f}" )

                        st.caption( f"{int(row['Common Users'])} " f"common users" )

                    else:
                        st.write( f"🔥 Hybrid: " f"{row['Hybrid Score']:.3f}" )

                    if st.button(
                        "View Ratings",
                        key=(
                            f"recommend_"
                            f"{method_name}_"
                            f"{position}_"
                            f"{title}"
                        ),
                        width="stretch"
                    ):
                        st.session_state[ "selected_book" ] = title

                        st.session_state[ "detail_book" ] = title

                        st.rerun()


# ============================================================
# 11. BOOK CATALOGUE CARDS
# ============================================================

def render_catalog_cards(
    catalog_df,
    group_series,
    series_groups
):
    for row_start in range( 0, len(catalog_df), 4 ):
        columns = st.columns( 4, gap="large" )

        for offset in range(4):
            row_index = ( row_start + offset )

            if row_index >= len( catalog_df ):
                break

            row = catalog_df.iloc[ row_index ]

            if group_series:
                series_key = row[ "Series-Key" ]

                titles = series_groups.get( series_key, [ row[ "Book-Title" ] ] )

                series_name = row[ "Series-Name" ]

            else:
                titles = [ row[ "Book-Title" ] ]

                series_name = row[ "Book-Title" ]

            with columns[offset]:
                with st.container( border=True, key=f"catalog_card_{row_index}_{offset}" ):
                    if ( group_series and len(titles) > 1 ):
                        st.markdown( f"**📚 {series_name}**" )

                        st.caption( f"{len(titles)} books combined" )

                        selected_title = st.selectbox(
                            "Choose Book",
                            options=titles,
                            key=(
                                "series_card_"
                                + str(
                                    abs(
                                        hash(
                                            series_key
                                        )
                                    )
                                )
                            )
                        )

                    else:
                        selected_title = titles[ 0 ]

                    selected_rows = catalog_with_series[ catalog_with_series[ "Book-Title" ] == selected_title ]

                    if selected_rows.empty:
                        selected_row = row
                    else:
                        selected_row = selected_rows.iloc[ 0 ]

                    show_cover( selected_title, width=145 )

                    st.markdown( f"### {selected_title}" )

                    st.caption( f"Author: " f"{selected_row.get('Book-Author', 'Unknown')}" )

                    st.caption( f"Publisher: " f"{selected_row.get('Publisher', 'Unknown')}" )

                    score_col, users_col = st.columns( 2 )

                    avg_rating = selected_row.get( "rating", np.nan )

                    rating_count = selected_row.get( "num of ratings", np.nan )

                    with score_col:
                        if pd.notna( avg_rating ):
                            st.markdown( f"**⭐ " f"{float(avg_rating):.2f}/10**" )

                        else:
                            st.markdown( "**⭐ N/A**" )

                    with users_col:
                        if pd.notna( rating_count ):
                            st.caption( f"{int(rating_count):,} users" )

                        else:
                            st.caption( "No ratings" )

                    if st.button(
                        "View Rating Records",
                        key=(
                            "catalog_book_"
                            + str(
                                selected_row.get(
                                    "ISBN",
                                    selected_title
                                )
                            )
                            + "_"
                            + str(
                                row_index
                            )
                        ),
                        width="stretch"
                    ):
                        st.session_state[ "selected_book" ] = selected_title

                        st.session_state[ "detail_book" ] = selected_title

                        st.rerun()


# ============================================================
# 12. BEST METHOD HELPERS
# ============================================================

def get_best_method():
    if "evaluation_results" in st.session_state:
        result = st.session_state[ "evaluation_results" ]

        if isinstance( result, pd.DataFrame ) and not result.empty:
            best_row = result.loc[ result[ "F1@10" ].idxmax() ]
            best_method = str( best_row[ "Method" ] )
            best_f1 = float( best_row[ "F1@10" ] )

            reason = (
                f"{best_method} is currently the best method because it has the highest "
                f"F1@10 score ({best_f1:.4f}) in the latest Developer evaluation."
            )

            return best_method, reason

    return (
        "Popularity-Based",
        "Popularity-Based is used as the current default best method. "
        "Run Evaluation Metrics in Developer to compare all four methods. "
        "After evaluation, Main automatically uses the method with the highest F1@10."
    )


def get_method_result( method_name, title, top_n=10 ):
    if method_name == "Popularity-Based":
        result = recommend_popular( top_n + 1 )

        if result.empty:
            return result

        result = result[ result[ "Book-Title" ] != title ].head( top_n ).copy()
        result.index = range( 1, len( result ) + 1 )

        return result

    if method_name == "Content-Based":
        return get_recommendations( title, top_n )

    if method_name == "Collaborative":
        return recommend_collaborative( title, top_n )

    return recommend_hybrid( title, top_n )


# ============================================================
# 13. PAGE DESIGN
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        [class*="st-key-catalog_card_"] {
            height: 590px !important;
            min-height: 590px !important;
            max-height: 590px !important;
            overflow-y: auto !important;
            border: 2px solid #000000 !important;
            border-radius: 10px !important;
            background-color: #ffffff !important;
            padding: 8px !important;
            box-shadow: none !important;
        }

        [class*="st-key-catalog_card_"] img {
            width: 145px !important;
            height: 205px !important;
            object-fit: contain !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        [class*="st-key-recommendation_card_"] {
            height: 455px !important;
            min-height: 455px !important;
            max-height: 455px !important;
            overflow-y: auto !important;
            border: 2px solid #000000 !important;
            border-radius: 10px !important;
            background-color: #ffffff !important;
            padding: 8px !important;
            box-shadow: none !important;
        }

        [class*="st-key-recommendation_card_"] img {
            width: 125px !important;
            height: 180px !important;
            object-fit: contain !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        .st-key-selected_book_card {
            border: 2px solid #000000 !important;
            border-radius: 10px !important;
            background-color: #ffffff !important;
            box-shadow: none !important;
        }

        [class*="st-key-catalog_card_"] div[data-testid="stVerticalBlockBorderWrapper"],
        [class*="st-key-recommendation_card_"] div[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-selected_book_card div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #000000 !important;
            border-radius: 10px !important;
        }

        div[data-testid="stButton"] > button {
            min-height: 42px;
            border: 1.5px solid #000000 !important;
            border-radius: 8px !important;
            font-weight: 650;
        }

        div[data-testid="stTextInput"] input {
            min-height: 46px;
            border: 1.5px solid #000000 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="select"] > div {
            border: 1.5px solid #000000 !important;
        }

        section[data-testid="stSidebar"] {
            border-right: 1.5px solid #000000 !important;
        }

        /* ==================================================
           TOP NAVIGATION - RECTANGULAR TAB STYLE
           ================================================== */
        .st-key-nav_main button,
        .st-key-nav_developer button,
        .st-key-nav_about button {
            min-height: 48px !important;
            border: 1px solid #4b4b4b !important;
            border-radius: 5px 5px 0 0 !important;
            background-color: #4b4b4b !important;
            color: #ffffff !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            box-shadow: none !important;
        }

        .st-key-nav_main button:hover,
        .st-key-nav_developer button:hover,
        .st-key-nav_about button:hover {
            background-color: #606060 !important;
            color: #ffffff !important;
            border-color: #606060 !important;
        }

        /* Developer Dataset Analysis - normal dashboard size */
        [class*="st-key-analysis_panel_"] {
            height: 390px !important;
            min-height: 390px !important;
            max-height: 390px !important;
            overflow: hidden !important;
            border: 2px solid #000000 !important;
            border-radius: 8px !important;
            background-color: #ffffff !important;
            padding: 10px !important;
            box-shadow: none !important;
        }

        [class*="st-key-analysis_panel_"] div[data-testid="stDataFrame"] {
            max-height: 315px !important;
            overflow: auto !important;
        }

        div[data-baseweb="tab-list"] {
            gap: 6px !important;
        }

        button[data-baseweb="tab"] {
            min-height: 40px !important;
            border: 1.5px solid #000000 !important;
            border-radius: 5px 5px 0 0 !important;
            padding: 6px 15px !important;
            font-weight: 650 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 14. SESSION STATE
# ============================================================

if "selected_method" not in st.session_state:
    st.session_state.selected_method = "Hybrid"

if "selected_book" not in st.session_state:
    if example_book in set( ui_books[ "Book-Title" ] ):
        st.session_state.selected_book = example_book
    else:
        st.session_state.selected_book = ui_books.iloc[ 0 ][ "Book-Title" ]

if "detail_book" not in st.session_state:
    st.session_state.detail_book = st.session_state.selected_book

if "search_query" not in st.session_state:
    st.session_state.search_query = ""


# ============================================================
# 15. MAIN HEADER + TOP NAVIGATION
# ============================================================

with st.container(border=True):

    title_icon, title_text = st.columns(
        [0.6, 9.4],
        vertical_alignment="center"
    )

    with title_icon:
        st.markdown(
            "<div style='font-size:42px;'>📖</div>",
            unsafe_allow_html=True
        )

    with title_text:
        st.markdown(
            "<h1 style='margin:0; padding:0;'>Book Recommender System</h1>",
            unsafe_allow_html=True
        )

    st.caption(
        "Main is for users, Developer is for recommendation testing "
        "and evaluation and About explains the system."
    )

if "current_page" not in st.session_state:
    st.session_state.current_page = "Main"

nav_main, nav_developer, nav_about, nav_space = st.columns( [1.15, 1.35, 1.15, 6.35], gap="small" )

with nav_main:
    if st.button( "🏠 Main", key="nav_main", width="stretch" ):
        st.session_state.current_page = "Main"
        st.rerun()

with nav_developer:
    if st.button( "🛠️ Developer", key="nav_developer", width="stretch" ):
        st.session_state.current_page = "Developer"
        st.rerun()

with nav_about:
    if st.button( "ℹ️ About", key="nav_about", width="stretch" ):
        st.session_state.current_page = "About"
        st.rerun()

navigation = st.session_state.current_page

active_nav_key = {
    "Main": "nav_main",
    "Developer": "nav_developer",
    "About": "nav_about"
}[ navigation ]

st.markdown(
    f"""
    <style>

    /* ================================================
       DEFAULT / NON-SELECTED NAVIGATION BUTTONS
       ================================================ */

    .st-key-nav_main button,
    .st-key-nav_developer button,
    .st-key-nav_about button {{

        background-color: #ffffff !important;
        color: #222222 !important;

        border: 1px solid #222222 !important;
        border-radius: 6px !important;

        min-height: 48px !important;

        box-shadow: none !important;
    }}


    /* ================================================
       NON-SELECTED BUTTON HOVER
       ================================================ */

    .st-key-nav_main button:hover,
    .st-key-nav_developer button:hover,
    .st-key-nav_about button:hover {{

        background-color: #eeeeee !important;
        color: #222222 !important;

        border-color: #222222 !important;
    }}


    /* ================================================
       CURRENTLY SELECTED BUTTON
       ================================================ */

    .st-key-{active_nav_key} button {{

        background-color: #4b4b4b !important;
        color: #ffffff !important;

        border: 1px solid #222222 !important;

        font-weight: 600 !important;
    }}


    /* ================================================
       SELECTED BUTTON HOVER
       ================================================ */

    .st-key-{active_nav_key} button:hover {{

        background-color: #4b4b4b !important;
        color: #ffffff !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 16. MAIN - USER INTERFACE
# ============================================================

if navigation == "Main":

    st.sidebar.title( "📚 Book Explorer" )
    st.sidebar.caption( "Search and organise the user book catalogue." )

    minimum_rating = st.sidebar.slider(
        "Minimum Average Rating",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5
    )

    maximum_rating_count = int(
        min( 500, catalog_with_series[ "num of ratings" ].fillna(0).max() )
    )

    minimum_users = st.sidebar.slider(
        "Minimum Users Rated",
        min_value=0,
        max_value=max( 1, maximum_rating_count ),
        value=0,
        step=1
    )

    sort_option = st.sidebar.selectbox(
        "Sort Catalogue",
        [ "Most Rated", "Highest Rating", "Title A-Z" ]
    )

    group_series = st.sidebar.toggle(
        "Combine Same-Series Books",
        value=True,
        help=(
            "Same-series books are combined into one catalogue card. "
            "You can still choose the exact book inside the card."
        )
    )

    st.sidebar.divider()
    st.sidebar.caption( f"Selected Book: {st.session_state.selected_book}" )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    st.subheader( "1. Search Book" )

    search_col, search_button_col = st.columns( [12, 1] )

    with search_col:
        typed_search = st.text_input(
            "Search",
            value=st.session_state.search_query,
            placeholder="Search title, author or publisher...",
            label_visibility="collapsed",
            key="search_input_box"
        )

    with search_button_col:
        search_clicked = st.button(
            "🔍",
            key="search_icon_button",
            width="stretch",
            help="Search"
        )

    if search_clicked:
        st.session_state.search_query = typed_search.strip()
        st.rerun()

    if typed_search.strip() == "":
        st.session_state.search_query = ""

    # --------------------------------------------------------
    # FILTER + SORT
    # --------------------------------------------------------

    filtered_catalog = catalog_with_series.copy()
    filtered_catalog = filtered_catalog.drop_duplicates(
        subset=[ "Book-Title" ],
        keep="first"
    )

    if st.session_state.search_query:
        keyword = st.session_state.search_query

        mask = (
            filtered_catalog[ "Book-Title" ].str.contains(
                keyword, case=False, na=False, regex=False
            )
            |
            filtered_catalog[ "Book-Author" ].str.contains(
                keyword, case=False, na=False, regex=False
            )
            |
            filtered_catalog[ "Publisher" ].str.contains(
                keyword, case=False, na=False, regex=False
            )
        )

        filtered_catalog = filtered_catalog[ mask ].copy()

    filtered_catalog = filtered_catalog[
        filtered_catalog[ "rating" ].fillna(0) >= minimum_rating
    ].copy()

    filtered_catalog = filtered_catalog[
        filtered_catalog[ "num of ratings" ].fillna(0) >= minimum_users
    ].copy()

    if sort_option == "Most Rated":
        filtered_catalog = filtered_catalog.sort_values(
            "num of ratings",
            ascending=False,
            na_position="last"
        )

    elif sort_option == "Highest Rating":
        filtered_catalog = filtered_catalog.sort_values(
            "rating",
            ascending=False,
            na_position="last"
        )

    else:
        filtered_catalog = filtered_catalog.sort_values(
            "Book-Title",
            ascending=True
        )

    series_groups = (
        filtered_catalog.groupby( "Series-Key" )[ "Book-Title" ]
        .apply( lambda values: list( dict.fromkeys( values.tolist() ) ) )
        .to_dict()
    )

    if group_series:
        filtered_catalog = (
            filtered_catalog
            .sort_values(
                "num of ratings",
                ascending=False,
                na_position="last"
            )
            .drop_duplicates(
                subset=[ "Series-Key" ],
                keep="first"
            )
            .reset_index( drop=True )
        )

    else:
        filtered_catalog = filtered_catalog.reset_index( drop=True )

    # --------------------------------------------------------
    # BOOK CATALOGUE
    # --------------------------------------------------------

    st.subheader( "2. Book Catalogue" )

    if group_series:
        st.caption(
            "Same-series books are combined into one card. "
            "Choose the exact book using the selector inside the card."
        )
    else:
        st.caption( "Each card represents one unique book title." )

    if filtered_catalog.empty:
        st.warning( "No books match the current search and filters." )

    else:
        total_results = len( filtered_catalog )
        total_pages = max( 1, math.ceil( total_results / BOOKS_PER_PAGE ) )

        page_col, information_col = st.columns( [1, 4] )

        with page_col:
            page_number = st.selectbox(
                "Page",
                options=list( range( 1, total_pages + 1 ) ),
                key="main_catalog_page"
            )

        with information_col:
            st.caption(
                f"Showing {total_results:,} catalogue card(s). "
                "Click View Rating Records to select a book."
            )

        start = ( page_number - 1 ) * BOOKS_PER_PAGE
        end = start + BOOKS_PER_PAGE
        page_data = filtered_catalog.iloc[ start:end ]

        render_catalog_cards( page_data, group_series, series_groups )

    # --------------------------------------------------------
    # SELECTED BOOK + RATING RECORDS
    # --------------------------------------------------------

    st.divider()
    st.subheader( "3. Selected Book & User Rating Records" )
    render_selected_book( st.session_state.detail_book )

    # --------------------------------------------------------
    # RECOMMENDATIONS FOR USER
    # --------------------------------------------------------

    st.divider()

    best_method, best_reason = get_best_method()


    st.subheader( "4. Recommended For You" )

    selected_book = st.session_state.selected_book

    st.caption(
        f"Recommendations based on your selected book: **{selected_book}**"
    )

    with st.spinner( "Generating recommendations..." ):
        best_result = get_method_result( best_method, selected_book, 10 )

    render_recommendation_cards( best_result, best_method )


# ============================================================
# 17. DEVELOPER - METHODS + EVALUATION
# ============================================================

elif navigation == "Developer":

    st.sidebar.title( "🛠️ Developer Tools" )
    st.sidebar.caption(
        "Compare recommendation methods and evaluate system performance."
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Dataset analysis graphs are displayed in the Developer dashboard."
    )

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    st.subheader( "1. Dataset Overview" )

    overview1, overview2, overview3, overview4 = st.columns( 4 )

    overview1.metric( "Users", f"{system['number_of_users']:,}" )
    overview2.metric( "Rated Books", f"{system['number_of_books']:,}" )
    overview3.metric( "Ratings", f"{system['number_of_ratings']:,}" )
    overview4.metric( "Sparsity", f"{system['matrix_sparsity'] * 100:.2f}%" )

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    st.divider()
    st.subheader( "2. Dataset Statistics" )
    st.caption( "Key statistics calculated after data preprocessing." )

    stat1, stat2, stat3 = st.columns( 3 )
    stat1.metric( "Cleaned Books", f"{system['cleaned_books_count']:,}" )
    stat2.metric( "Explicit Ratings", f"{system['number_of_ratings']:,}" )
    stat3.metric( "Merged Records", f"{system['merged_records_count']:,}" )

    stat4, stat5, stat6 = st.columns( 3 )
    stat4.metric( "Average Rating", f"{system['average_rating']:.2f}" )
    stat5.metric( "Minimum Rating", f"{system['minimum_rating']:.0f}" )
    stat6.metric( "Maximum Rating", f"{system['maximum_rating']:.0f}" )

    # --------------------------------------------------------
    # DATASET ANALYSIS DASHBOARD
    # --------------------------------------------------------

    st.divider()
    st.subheader( "3. Dataset Analysis" )

    st.caption(
        "Choose Graph or Records for each dataset analysis."
    )

    # --------------------------------------------------------
    # ROW 1 - TWO ANALYSIS PANELS
    # --------------------------------------------------------

    row1_left, row1_right = st.columns( 2, gap="large" )

    with row1_left:
        st.markdown( "### Distribution of User Book Ratings" )

        graph_tab, record_tab = st.tabs( [ "📊 Graph", "📋 Records" ] )

        with graph_tab:
            with st.container( border=True, key="analysis_panel_user_ratings_graph" ):
                figure_user_ratings = make_dataset_figure( "Distribution of User Book Ratings" )
                st.pyplot( figure_user_ratings, clear_figure=True, width="stretch" )
                st.caption( "Shows how frequently users gave ratings from 1 to 10." )

        with record_tab:
            with st.container( border=True, key="analysis_panel_user_ratings_record" ):
                user_rating_records = get_dataset_analysis_records( "Distribution of User Book Ratings" )

                st.dataframe(
                    user_rating_records,
                    hide_index=True,
                    width="stretch",
                    height=300,
                    column_config={
                        "Book Rating": st.column_config.NumberColumn( "Book Rating", format="%d" ),
                        "Number of Ratings": st.column_config.NumberColumn( "Number of Ratings", format="%d" )
                    }
                )


    with row1_right:
        st.markdown( "### Hybrid Recommendation Weights" )

        graph_tab, record_tab = st.tabs( [ "📊 Graph", "📋 Records" ] )

        with graph_tab:
            with st.container( border=True, key="analysis_panel_hybrid_weights_graph" ):
                figure_hybrid_weights = make_dataset_figure( "Hybrid Recommendation Weights" )
                st.pyplot( figure_hybrid_weights, clear_figure=True, width="stretch" )
                st.caption( "Shows Content 40%, Collaborative 45% and Popularity 15%." )

        with record_tab:
            with st.container( border=True, key="analysis_panel_hybrid_weights_record" ):
                hybrid_weight_records = get_dataset_analysis_records( "Hybrid Recommendation Weights" )

                st.dataframe(
                    hybrid_weight_records,
                    hide_index=True,
                    width="stretch",
                    height=300,
                    column_config={
                        "Recommendation Method": st.column_config.TextColumn( "Recommendation Method" ),
                        "Weight (%)": st.column_config.NumberColumn( "Weight (%)", format="%.0f" )
                    }
                )


    # --------------------------------------------------------
    # ROW 2 - TWO ANALYSIS PANELS
    # --------------------------------------------------------

    row2_left, row2_right = st.columns( 2, gap="large" )

    with row2_left:
        st.markdown( "### Distribution of Average Book Ratings" )

        graph_tab, record_tab = st.tabs( [ "📊 Graph", "📋 Records" ] )

        with graph_tab:
            with st.container( border=True, key="analysis_panel_average_ratings_graph" ):
                figure_average_ratings = make_dataset_figure( "Distribution of Average Book Ratings" )
                st.pyplot( figure_average_ratings, clear_figure=True, width="stretch" )
                st.caption( "Shows how average book ratings are distributed." )

        with record_tab:
            with st.container( border=True, key="analysis_panel_average_ratings_record" ):
                average_rating_records = get_dataset_analysis_records( "Distribution of Average Book Ratings" )

                st.dataframe(
                    average_rating_records,
                    hide_index=True,
                    width="stretch",
                    height=300,
                    column_config={
                        "Average Rating Range": st.column_config.TextColumn( "Average Rating Range" ),
                        "Number of Books": st.column_config.NumberColumn( "Number of Books", format="%d" )
                    }
                )


    with row2_right:
        st.markdown( "### Distribution of Number of Book Ratings" )

        graph_tab, record_tab = st.tabs( [ "📊 Graph", "📋 Records" ] )

        with graph_tab:
            with st.container( border=True, key="analysis_panel_rating_counts_graph" ):
                figure_rating_counts = make_dataset_figure( "Distribution of Number of Book Ratings" )
                st.pyplot( figure_rating_counts, clear_figure=True, width="stretch" )
                st.caption( "Shows how many ratings books usually receive." )

        with record_tab:
            with st.container( border=True, key="analysis_panel_rating_counts_record" ):
                rating_count_records = get_dataset_analysis_records( "Distribution of Number of Book Ratings" )

                st.dataframe(
                    rating_count_records,
                    hide_index=True,
                    width="stretch",
                    height=300,
                    column_config={
                        "Number of Ratings Range": st.column_config.TextColumn( "Number of Ratings Range" ),
                        "Number of Books": st.column_config.NumberColumn( "Number of Books", format="%d" )
                    }
                )


    # --------------------------------------------------------
    # ROW 3 - TOP 10 MOST-RATED BOOKS
    # --------------------------------------------------------

    st.markdown( "### Top 10 Most-Rated Books" )

    top10_left_space, top10_col, top10_right_space = st.columns( [1, 2, 1], gap="large" )

    with top10_col:
        graph_tab, record_tab = st.tabs( [ "📊 Graph", "📋 Records" ] )

        with graph_tab:
            with st.container( border=True, key="analysis_panel_most_rated_graph" ):
                figure_most_rated = make_dataset_figure( "Top 10 Most-Rated Books" )
                st.pyplot( figure_most_rated, clear_figure=True, width="stretch" )
                st.caption( "Shows the ten books with the largest number of ratings." )

        with record_tab:
            with st.container( border=True, key="analysis_panel_most_rated_record" ):
                most_rated_records = get_dataset_analysis_records( "Top 10 Most-Rated Books" )

                st.dataframe(
                    most_rated_records,
                    hide_index=True,
                    width="stretch",
                    height=300,
                    column_config={
                        "Book Title": st.column_config.TextColumn( "Book Title" ),
                        "Number of Ratings": st.column_config.NumberColumn( "Number of Ratings", format="%d" )
                    }
                )


    # --------------------------------------------------------
    # CHOOSE RECOMMENDATION METHOD
    # --------------------------------------------------------

    st.divider()
    st.subheader( "4. Choose Recommendation Method" )
    st.caption(
        "Developers can compare Popularity-Based, Content-Based, Collaborative and Hybrid."
    )

    method_columns = st.columns( 4 )

    method_buttons = [
        ( "Popularity-Based", "⭐ Popularity-Based" ),
        ( "Content-Based", "📖 Content-Based" ),
        ( "Collaborative", "👥 Collaborative" ),
        ( "Hybrid", "🔥 Hybrid" )
    ]

    for column, (method_key, method_label) in zip(method_columns, method_buttons):
        with column:
            if st.button(
                method_label,
                key=f"method_{method_key}",
                type=(
                    "primary"
                    if st.session_state.selected_method == method_key
                    else "secondary"
                ),
                width="stretch"
            ):
                st.session_state.selected_method = method_key
                st.rerun()

    method_descriptions = {
        "Popularity-Based":
            "Ranks generally popular books using weighted rating.",

        "Content-Based":
            "Finds similar books using TF-IDF and cosine similarity.",

        "Collaborative":
            "Uses Pearson correlation based on common users.",

        "Hybrid":
            "Combines Content 40%, Collaborative 45% and Popularity 15%."
    }

    st.info( method_descriptions[ st.session_state.selected_method ] )

    developer_book = st.selectbox(
        "Book for Method Testing",
        options=test_books[ "Book-Title" ].drop_duplicates().tolist(),
        index=0,
        key="developer_test_book"
    )

    st.markdown(
        f"#### {st.session_state.selected_method} Result Graph"
    )

    method_chart = make_method_chart(
        st.session_state.selected_method,
        developer_book,
        top_n=6
    )

    if method_chart.empty:
        st.warning(
            "No graph is available for this book and method."
        )
    else:
        st.bar_chart(
            method_chart,
            height=330
        )

    developer_result = get_method_result(
        st.session_state.selected_method,
        developer_book,
        10
    )

    with st.expander( "View Top 10 Method Results" ):
        if developer_result.empty:
            st.warning( "No recommendation result is available." )
        else:
            st.dataframe(
                developer_result,
                width="stretch"
            )

    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    st.divider()
    st.subheader( "5. Evaluation Metrics" )

    st.caption(
        "A rating of 8 or above is treated as positive. "
        "Precision@10 measures recommendation accuracy, Recall@10 measures how many relevant books are found, "
        "and F1@10 balances Precision and Recall."
    )

    st.info(
        f"Choose how many valid query books you want to evaluate from the Random "
        f"{TEST_SAMPLE_SIZE:,} Testing Dataset."
    )

    evaluation_col1, evaluation_col2 = st.columns( [1, 3] )

    with evaluation_col1:
        evaluated_queries = st.number_input(
            "Number of Evaluated Queries",
            min_value=1,
            max_value=TEST_SAMPLE_SIZE,
            value=min( EVALUATION_SAMPLE_SIZE, TEST_SAMPLE_SIZE ),
            step=1
        )

    with evaluation_col2:
        st.caption(
            "A larger number gives a broader evaluation, but Collaborative and Hybrid "
            "methods will take longer to calculate."
        )

    if st.button(
        "Run Evaluation",
        key="run_evaluation_button",
        type="primary"
    ):
        with st.spinner(
            f"Evaluating {evaluated_queries} query books using the four recommendation methods..."
        ):
            st.session_state[ "evaluation_results" ] = evaluate_recommender_system(
                sample_size=int( evaluated_queries ),
                top_n=EVALUATION_K
            )

            st.session_state[ "evaluation_requested_queries" ] = int(
                evaluated_queries
            )

    if "evaluation_results" in st.session_state:
        evaluation_results = st.session_state[ "evaluation_results" ]

        st.markdown( "#### Evaluation Result Table" )

        st.dataframe(
            evaluation_results,
            hide_index=True,
            width="stretch",
            column_config={
                "Method": st.column_config.TextColumn(
                    "Recommendation Method"
                ),
                "Precision@10": st.column_config.NumberColumn(
                    "Precision@10",
                    format="%.4f"
                ),
                "Recall@10": st.column_config.NumberColumn(
                    "Recall@10",
                    format="%.4f"
                ),
                "F1@10": st.column_config.NumberColumn(
                    "F1@10",
                    format="%.4f"
                ),
                "Evaluated Queries": st.column_config.NumberColumn(
                    "Evaluated Queries",
                    format="%d"
                )
            }
        )

        actual_queries = (
            int( evaluation_results[ "Evaluated Queries" ].max() )
            if not evaluation_results.empty
            else 0
        )

        requested_queries = st.session_state.get(
            "evaluation_requested_queries",
            EVALUATION_SAMPLE_SIZE
        )

        if actual_queries < requested_queries:
            st.warning(
                f"You requested {requested_queries} queries, but only {actual_queries} "
                "testing books had enough relevant user-rating data for evaluation."
            )

        st.markdown(
            "#### Precision, Recall and F1 Score Graph"
        )

        evaluation_graph = evaluation_results.set_index(
            "Method"
        )[ [ "Precision@10", "Recall@10", "F1@10" ] ]

        st.bar_chart(
            evaluation_graph,
            height=420,
            y_label="Score"
        )

        if not evaluation_results.empty:
            best_row = evaluation_results.loc[
                evaluation_results[ "F1@10" ].idxmax()
            ]

            st.success(
                f"Best Method: {best_row['Method']} | "
                f"Highest F1@10: {float(best_row['F1@10']):.4f}. "
                "Main will automatically use this method as the best recommendation method."
            )

    # --------------------------------------------------------
    # RANDOM 1,000 TESTING DATASET
    # --------------------------------------------------------

    st.divider()

    with st.expander(
        f"Random {TEST_SAMPLE_SIZE:,} Testing Dataset"
    ):
        st.write(
            f"UI browsing dataset: **{len(ui_books):,} books**"
        )

        st.write(
            f"Random testing sample: **{len(test_books):,} books**"
        )

        st.write(
            f"Random State: **{RANDOM_STATE + 1}**"
        )

        st.dataframe(
            test_books[
                [ "ISBN", "Book-Title", "Book-Author" ]
            ],
            hide_index=True,
            width="stretch"
        )


# ============================================================
# 18. ABOUT
# ============================================================

else:

    # --------------------------------------------------------
    # ABOUT HEADER
    # --------------------------------------------------------

    st.title("ℹ️ About the Book Recommender System")

    st.write(
        "The Book Recommender System is an Artificial Intelligence "
        "application developed to help users discover suitable books "
        "from a large book collection."
    )

    st.write(
        "The system analyses book information, rating popularity and "
        "user-rating behaviour using several recommendation techniques "
        "to generate Top 10 book recommendations."
    )

    st.divider()

    # ========================================================
    # SYSTEM OVERVIEW
    # ========================================================

    st.subheader("📖 System Overview")

    overview1, overview2, overview3, overview4 = st.columns(4)

    with overview1:
        st.metric(
            "Catalogue Books",
            f"{len(ui_books):,}"
        )

    with overview2:
        st.metric(
            "Testing Books",
            f"{len(test_books):,}"
        )

    with overview3:
        st.metric(
            "Recommendation Methods",
            "4"
        )

    with overview4:
        st.metric(
            "Top Recommendations",
            "10"
        )

    st.caption(
        "The catalogue and testing values are generated from the "
        "processed dataset used by the current system."
    )

    st.divider()

    # ========================================================
    # MAIN FEATURES
    # ========================================================

    st.subheader("✨ Main Features")

    feature1, feature2 = st.columns(2, gap="large")

    with feature1:

        with st.container(border=True):

            st.markdown("### 🔍 Book Search")

            st.write(
                "Search for books using the book title, author "
                "or publisher."
            )

        with st.container(border=True):

            st.markdown("### ⭐ User Rating Records")

            st.write(
                "View the average rating, rating distribution and "
                "individual user-rating records for a selected book."
            )

    with feature2:

        with st.container(border=True):

            st.markdown("### 📚 Book Catalogue")

            st.write(
                "Browse books from the catalogue and view important "
                "information such as author, publisher and ratings."
            )

        with st.container(border=True):

            st.markdown("### 🎯 Top 10 Recommendations")

            st.write(
                "Receive Top 10 book recommendations generated by "
                "the recommendation methods implemented in the system."
            )

    st.divider()

    # ========================================================
    # RECOMMENDATION METHODS
    # ========================================================

    st.subheader("🤖 Recommendation Methods")

    method1, method2 = st.columns(2, gap="large")

    with method1:

        with st.container(border=True):

            st.markdown("### ⭐ Popularity-Based")

            st.write(
                "Ranks generally popular books using a weighted-rating "
                "calculation based on average rating and number of ratings."
            )

            st.caption(
                "Technique: Weighted Rating"
            )

        with st.container(border=True):

            st.markdown("### 🔎 Content-Based Filtering")

            st.write(
                "Recommends books that contain similar book information "
                "to the selected book."
            )

            st.caption(
                "Technique: TF-IDF + Cosine Similarity"
            )

    with method2:

        with st.container(border=True):

            st.markdown("### 👥 Collaborative Filtering")

            st.write(
                "Identifies relationships between books by analysing "
                "similar user-rating patterns."
            )

            st.caption(
                "Technique: User-Item Matrix + Pearson Correlation"
            )

        with st.container(border=True):

            st.markdown("### 🔥 Hybrid Recommendation")

            st.write(
                "Combines multiple recommendation signals to produce "
                "a more balanced recommendation score."
            )

            st.caption(
                "40% Content-Based + "
                "45% Collaborative + "
                "15% Popularity-Based"
            )

    st.divider()

    # ========================================================
    # HOW THE SYSTEM WORKS
    # ========================================================

    st.subheader("⚙️ How the System Works")

    st.markdown(
        """
        **1. Select or search for a book**  
        The user searches the catalogue or chooses a book.

        **2. View book and rating information**  
        The system displays book information together with user-rating data.

        **3. Analyse recommendation information**  
        Different recommendation methods analyse book information,
        popularity and user-rating behaviour.

        **4. Generate Top 10 recommendations**  
        Books are ranked according to the selected recommendation method.

        **5. Evaluate recommendation performance**  
        Developer mode compares the recommendation methods using
        Precision@10, Recall@10 and F1@10.
        """
    )

    st.divider()

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.subheader("🗂️ Dataset Information")

    dataset1, dataset2, dataset3 = st.columns(3)

    with dataset1:
        with st.container(border=True):

            st.markdown("### 📘 Books.csv")

            st.write(
                "Contains ISBN, book title, author, publication year "
                "and publisher information."
            )

    with dataset2:
        with st.container(border=True):

            st.markdown("### ⭐ Ratings.csv")

            st.write(
                "Contains User-ID, ISBN and Book-Rating information "
                "used to analyse user-rating behaviour."
            )

    with dataset3:
        with st.container(border=True):

            st.markdown("### 👤 Users.csv")

            st.write(
                "Contains anonymised user information including "
                "User-ID, location and age."
            )

    st.info(
        "The system uses explicit ratings from 1 to 10. "
        "Rating 0 is excluded because it represents implicit feedback."
    )

    st.divider()

    # ========================================================
    # EVALUATION
    # ========================================================

    st.subheader("📊 Evaluation Metrics")

    eval1, eval2, eval3 = st.columns(3)

    with eval1:
        with st.container(border=True):

            st.markdown("### Precision@10")

            st.write(
                "Measures how many of the Top 10 recommended books "
                "are relevant."
            )

    with eval2:
        with st.container(border=True):

            st.markdown("### Recall@10")

            st.write(
                "Measures how many relevant books are successfully "
                "retrieved in the Top 10 recommendation list."
            )

    with eval3:
        with st.container(border=True):

            st.markdown("### F1@10")

            st.write(
                "Combines Precision and Recall into one overall "
                "recommendation-performance score."
            )

    st.caption(
        "A rating of 8 or above is treated as positive feedback "
        "during the current evaluation process."
    )

    st.divider()

    # ========================================================
    # TECHNOLOGY
    # ========================================================

    st.subheader("💻 Technology Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.info(
            "🐍 **Python**\n\n"
            "System programming"
        )

    with tech2:
        st.info(
            "🌐 **Streamlit**\n\n"
            "Web user interface"
        )

    with tech3:
        st.info(
            "📊 **Pandas / NumPy**\n\n"
            "Data processing"
        )

    with tech4:
        st.info(
            "🤖 **Scikit-learn**\n\n"
            "TF-IDF and similarity"
        )

    st.divider()

    # ========================================================
    # TEAM CONTRIBUTION
    # ========================================================

    st.subheader("👨‍💻 Team Contribution")

    team_data = pd.DataFrame(
        {
            "Member": [
                "Wong Kai Jun",
                "Yeong Wei Kin",
                "Heng Chun Wai"
            ],

            "Recommendation Module": [
                "Popularity-Based Recommendation",
                "Content-Based Filtering",
                "Collaborative Filtering"
            ],

            "Main Technique": [
                "Weighted Rating",
                "TF-IDF + Cosine Similarity",
                "Pearson Correlation"
            ]
        }
    )

    st.dataframe(
        team_data,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "The three recommendation approaches are also integrated "
        "into the Hybrid Recommendation method."
    )

    st.divider()

    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.subheader("⚠️ Current Limitations")

    st.write(
        "• Content-Based Filtering is mainly limited to book title, "
        "author and publisher because the dataset does not provide "
        "detailed genres, descriptions or keywords."
    )

    st.write(
        "• Collaborative Filtering can be affected by data sparsity "
        "when only a small number of users rate the same books."
    )

    st.write(
        "• Popularity-Based recommendations are not personalised "
        "because users may receive similar popular books."
    )

    st.write(
        "• The Hybrid recommendation weights are manually configured "
        "and may not represent the optimal combination."
    )

    st.divider()

    st.caption(
        "BMCS2203 Artificial Intelligence • "
        "Book Recommender System"
    )