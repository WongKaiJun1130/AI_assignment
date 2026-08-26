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

UI_SAMPLE_SIZE = 1000
TEST_SAMPLE_SIZE = 300
RANDOM_STATE = 42
BOOKS_PER_PAGE = 20

CONTENT_WEIGHT = 0.40
COLLABORATIVE_WEIGHT = 0.45
POPULARITY_WEIGHT = 0.15

MIN_COMMON_USERS = 10


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
    # UI DATASET - RANDOM 1000 BOOKS
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


    catalog_pool = books[
        books[
            "Book-Title"
        ].isin(
            common_titles
        )
    ][
        display_columns
    ].drop_duplicates(
        subset=[
            "Book-Title"
        ]
    ).copy()


    catalog_pool = catalog_pool.merge( rating_stats.reset_index(), on="Book-Title", how="left" )


    ui_n = min( UI_SAMPLE_SIZE, len( catalog_pool ) )


    ui_books = catalog_pool.sample( n=ui_n, random_state=RANDOM_STATE ).reset_index( drop=True )


    # Keep example book inside UI
    if (
        example_book
        not in
        set(
            ui_books[
                "Book-Title"
            ]
        )
        and
        example_book
        in
        set(
            catalog_pool[
                "Book-Title"
            ]
        )
        and
        len(
            ui_books
        ) > 0
    ):

        example_row = catalog_pool[ catalog_pool[ "Book-Title" ] == example_book ].head( 1 )


        ui_books = pd.concat( [ example_row, ui_books.iloc[ :-1 ] ], ignore_index=True )


    # ========================================================
    # RANDOM 300 TESTING BOOKS
    # ========================================================

    test_n = min(
        TEST_SAMPLE_SIZE,
        len(
            ui_books
        )
    )


    test_books = ui_books.sample( n=test_n, random_state=RANDOM_STATE + 1 ).reset_index( drop=True )


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
# 4. BOOK + RATING HELPERS
# ============================================================

def get_cover_url(title):
    if title not in book_info.index:
        return None

    row = book_info.loc[title]

    for column in ["Image-URL-M", "Image-URL-L", "Image-URL-S"]:
        if column in row.index:
            value = row[column]

            if pd.notna(value):
                value = str(value).strip()

                if value:
                    if value.startswith("http://"):
                        value = "https://" + value[7:]

                    return value

    return None


def show_cover(title, width=150):
    cover_url = get_cover_url(title)

    if cover_url:
        st.image(cover_url, width=width)
    else:
        with st.container(border=True):
            st.markdown("### 📕")
            st.caption("No Cover Available")


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
        top_books = (
            system["data"]
            .groupby(
                "Book-Title"
            )["Book-Rating"]
            .count()
            .sort_values(
                ascending=False
            )
            .head(10)
            .sort_values()
        )

        ax.barh( top_books.index, top_books.values )

        ax.set_title( "Top 10 Most-Rated Books" )

        ax.set_xlabel( "Number of Ratings" )

        ax.set_ylabel( "Book Title" )

        ax.grid( axis="x", alpha=0.35 )


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

    fig.tight_layout()

    return fig


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
# 12. PAGE DESIGN
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 13. SESSION STATE
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
# 14. LEFT SIDEBAR - LIBRARY TOOLS
# ============================================================

st.sidebar.title(
    "📚 Library Tools"
)

st.sidebar.caption( "Use these controls to organise and analyse the book catalogue." )

minimum_rating = st.sidebar.slider( "Minimum Average Rating", min_value=0.0, max_value=10.0, value=0.0, step=0.5 )

maximum_rating_count = int( min( 500, catalog_with_series[ "num of ratings" ].fillna(0).max() ) )

minimum_users = st.sidebar.slider(
    "Minimum Users Rated",
    min_value=0,
    max_value=max(
        1,
        maximum_rating_count
    ),
    value=0,
    step=1
)

sort_option = st.sidebar.selectbox( "Sort Catalogue", [ "Most Rated", "Highest Rating", "Title A-Z" ] )

group_series = st.sidebar.toggle(
    "Combine Same-Series Books",
    value=True,
    help=(
        "Same-series books are combined into one catalogue card. "
        "You can still choose the exact book inside the card."
    )
)

st.sidebar.divider()

st.sidebar.subheader( "Dataset Analysis" )

analysis_graph = st.sidebar.selectbox(
    "Choose Graph",
    [
        "Distribution of User Book Ratings",
        "Top 10 Most-Rated Books",
        "Distribution of Average Book Ratings",
        "Distribution of Number of Book Ratings",
        "Hybrid Recommendation Weights"
    ]
)

analysis_figure = make_dataset_figure( analysis_graph )

with st.sidebar:
    st.pyplot( analysis_figure, clear_figure=True )

st.sidebar.divider()

side1, side2 = st.sidebar.columns( 2 )

side1.metric( "UI Books", f"{len(ui_books):,}" )

side2.metric( "Testing", f"{len(test_books):,}" )

st.sidebar.caption( f"Selected Book: " f"{st.session_state.selected_book}" )


# ============================================================
# 15. MAIN HEADER
# ============================================================

with st.container(border=True):
    st.title("📚 Book Recommender System")
    st.caption(
        "Search books, inspect user rating records, compare four recommendation methods, "
        "and view Top 10 recommendations."
    )


# ============================================================
# 16. FOUR METHODS AT THE TOP
# ============================================================

st.subheader(
    "1. Choose Recommendation Method"
)

st.caption( "Choose one method. Its graph is shown directly below." )

method_columns = st.columns( 4 )

method_buttons = [
    (
        "Popularity-Based",
        "⭐ Popularity-Based"
    ),
    (
        "Content-Based",
        "📖 Content-Based"
    ),
    (
        "Collaborative",
        "👥 Collaborative"
    ),
    (
        "Hybrid",
        "🔥 Hybrid"
    )
]

for column, (method_key, method_label) in zip(method_columns, method_buttons):
    with column:
        if st.button(
            method_label,
            key=f"method_{method_key}",
            type=(
                "primary"
                if st.session_state.selected_method
                == method_key
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

st.markdown( f"#### {st.session_state.selected_method} Result Graph" )

method_chart = make_method_chart( st.session_state.selected_method, st.session_state.selected_book, top_n=6 )

if method_chart.empty:
    st.warning( "No graph is available for this book and method." )

else:
    st.bar_chart( method_chart, height=330 )


# ============================================================
# 17. DATASET OVERVIEW
# ============================================================

st.subheader(
    "2. Dataset Overview"
)

overview1, overview2, overview3, overview4 = st.columns( 4 )

overview1.metric( "Users", f"{system['number_of_users']:,}" )

overview2.metric( "Rated Books", f"{system['number_of_books']:,}" )

overview3.metric( "UI Catalogue", f"{len(ui_books):,}" )

overview4.metric( "Testing Sample", f"{len(test_books):,}" )


# ============================================================
# 18. SEARCH BOOK
# ============================================================

st.subheader(
    "3. Search Book"
)

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
    search_clicked = st.button( "🔍", key="search_icon_button", width="stretch", help="Search" )

if search_clicked:
    st.session_state.search_query = typed_search.strip()
    st.rerun()

if typed_search.strip() == "":
    st.session_state.search_query = ""


# ============================================================
# 19. FILTER + SORT + REMOVE DUPLICATES
# ============================================================

filtered_catalog = catalog_with_series.copy()

# Exact same book name must appear only once.
filtered_catalog = filtered_catalog.drop_duplicates(
    subset=["Book-Title"],
    keep="first"
)

if st.session_state.search_query:
    keyword = st.session_state.search_query

    mask = (
        filtered_catalog[
            "Book-Title"
        ].str.contains(
            keyword,
            case=False,
            na=False,
            regex=False
        )
        |
        filtered_catalog[
            "Book-Author"
        ].str.contains(
            keyword,
            case=False,
            na=False,
            regex=False
        )
        |
        filtered_catalog[
            "Publisher"
        ].str.contains(
            keyword,
            case=False,
            na=False,
            regex=False
        )
    )

    filtered_catalog = filtered_catalog[ mask ].copy()

filtered_catalog = filtered_catalog[ filtered_catalog[ "rating" ].fillna(0) >= minimum_rating ].copy()

filtered_catalog = filtered_catalog[ filtered_catalog[ "num of ratings" ].fillna(0) >= minimum_users ].copy()

if sort_option == "Most Rated":
    filtered_catalog = filtered_catalog.sort_values( "num of ratings", ascending=False, na_position="last" )

elif sort_option == "Highest Rating":
    filtered_catalog = filtered_catalog.sort_values( "rating", ascending=False, na_position="last" )

else:
    filtered_catalog = filtered_catalog.sort_values( "Book-Title", ascending=True )

series_groups = (
    filtered_catalog.groupby(
        "Series-Key"
    )["Book-Title"]
    .apply(
        lambda values:
            list(
                dict.fromkeys(
                    values.tolist()
                )
            )
    )
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
            subset=[
                "Series-Key"
            ],
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )

else:
    filtered_catalog = filtered_catalog.reset_index( drop=True )


# ============================================================
# 20. BOOK CATALOGUE
# ============================================================

st.subheader(
    "4. Book Catalogue"
)

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
        page_number = st.selectbox( "Page", options=list( range( 1, total_pages + 1 ) ) )

    with information_col:
        st.caption(
            f"Showing {total_results:,} catalogue card(s). "
            "Click View Rating Records to inspect exact user ratings."
        )

    start = ( page_number - 1 ) * BOOKS_PER_PAGE

    end = ( start + BOOKS_PER_PAGE )

    page_data = filtered_catalog.iloc[ start:end ]

    render_catalog_cards( page_data, group_series, series_groups )


# ============================================================
# 21. SELECTED BOOK + USER RATING RECORDS
# ============================================================

st.divider()

st.subheader( "5. Selected Book & User Rating Records" )

render_selected_book( st.session_state.detail_book )


# ============================================================
# 22. TOP 10 RECOMMENDATION RESULTS
# ============================================================

st.divider()

st.subheader( "6. Top 10 Recommendation Results" )

selected_book = st.session_state.selected_book
selected_method = st.session_state.selected_method

st.info( f"Selected Book: **{selected_book}**  |  " f"Method: **{selected_method}**" )

with st.spinner( "Generating recommendations..." ):
    if selected_method == "Popularity-Based":
        recommendation_result = recommend_popular( 10 )

    elif selected_method == "Content-Based":
        recommendation_result = get_recommendations( selected_book, 10 )

    elif selected_method == "Collaborative":
        recommendation_result = recommend_collaborative( selected_book, 10 )

    else:
        recommendation_result = recommend_hybrid( selected_book, 10 )

render_recommendation_cards( recommendation_result, selected_method )


# ============================================================
# 23. RANDOM 300 TESTING DATASET
# ============================================================

with st.expander(
    "Random 300 Testing Dataset"
):
    st.write( f"UI browsing dataset: " f"**{UI_SAMPLE_SIZE:,} books**" )

    st.write( f"Random testing sample: " f"**{TEST_SAMPLE_SIZE:,} books**" )

    st.write( f"Random State: " f"**{RANDOM_STATE + 1}**" )

    st.dataframe( test_books[ [ "ISBN", "Book-Title", "Book-Author" ] ], hide_index=True, width="stretch" )
