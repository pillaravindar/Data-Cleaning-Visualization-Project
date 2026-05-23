"""
Generate the MovieDB data cleaning and visualization project.

This project uses the real local mymoviedb.csv dataset copied into data/raw.
It creates:
- A cleaned movie dataset
- 18 exported visualizations
- Summary tables
- A polished Jupyter Notebook
- A static HTML dashboard
- A markdown insights report
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "mymoviedb_raw.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "mymoviedb_cleaned.csv"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "MovieDB_Data_Cleaning_Visualization_Project.ipynb"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
MPL_CACHE_DIR = BASE_DIR / "outputs" / "matplotlib_cache"

for folder in [CHARTS_DIR, REPORTS_DIR, TABLES_DIR, MPL_CACHE_DIR, CLEAN_DATA_PATH.parent, NOTEBOOK_PATH.parent]:
    folder.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns


DASHBOARD_PATH = REPORTS_DIR / "movie_dashboard.html"
INSIGHTS_REPORT_PATH = REPORTS_DIR / "movie_insights_report.md"
PROJECT_SUMMARY_PATH = REPORTS_DIR / "project_summary.json"


LANGUAGE_MAP = {
    "en": "English",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "ko": "Korean",
    "zh": "Chinese",
    "cn": "Chinese",
    "it": "Italian",
    "ru": "Russian",
    "de": "German",
    "pt": "Portuguese",
    "hi": "Hindi",
    "da": "Danish",
    "no": "Norwegian",
    "sv": "Swedish",
    "nl": "Dutch",
    "tr": "Turkish",
    "pl": "Polish",
    "th": "Thai",
}


def configure_plot_style() -> None:
    """Use one consistent chart style across the whole project."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titleweight": "bold",
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_movie_data(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, str]:
    """
    Load the movie CSV safely.

    The fast default CSV reader can fail on this file because movie overviews are
    long text fields. The Python engine is slower but more tolerant, which is a
    practical real-world loading technique.
    """
    try:
        return pd.read_csv(path), "default"
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip"), "python_engine"


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names into clean snake_case names."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return cleaned


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extra spaces and convert placeholder strings into missing values."""
    cleaned = df.copy()
    placeholders = {
        "": np.nan,
        "nan": np.nan,
        "none": np.nan,
        "null": np.nan,
        "n/a": np.nan,
        "na": np.nan,
        "unknown": np.nan,
    }
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace(placeholders)
    return cleaned


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dates and numeric columns into correct data types."""
    cleaned = df.copy()
    cleaned["release_date"] = pd.to_datetime(cleaned["release_date"], errors="coerce", format="mixed")
    for column in ["popularity", "vote_count", "vote_average"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def remove_unusable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows that do not contain enough valid movie information."""
    before = len(df)
    valid_mask = df["release_date"].notna() & df["title"].notna()
    cleaned = df.loc[valid_mask].copy().reset_index(drop=True)
    return cleaned, before - len(cleaned)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with sensible movie-dataset rules."""
    cleaned = df.copy()
    cleaned["overview"] = cleaned["overview"].fillna("Overview not available")
    cleaned["genre"] = cleaned["genre"].fillna("Unknown")
    cleaned["poster_url"] = cleaned["poster_url"].fillna("Poster not available")
    cleaned["original_language"] = cleaned["original_language"].fillna("unknown")

    for column in ["popularity", "vote_count", "vote_average"]:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["vote_count"] = cleaned["vote_count"].round().astype(int)
    cleaned["vote_average"] = cleaned["vote_average"].clip(0, 10)
    cleaned["popularity"] = cleaned["popularity"].clip(lower=0)
    return cleaned


def detect_iqr_bounds(series: pd.Series) -> tuple[float, float, float, float]:
    """Return Q1, Q3, lower bound, and upper bound for IQR outlier detection."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, lower, upper


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cap numeric outliers using the IQR method.

    Popularity and vote count naturally have very large blockbuster values.
    Capping creates analysis-friendly columns while preserving the original raw
    values for reference.
    """
    cleaned = df.copy()
    report_rows = []
    for column in columns:
        raw_column = f"{column}_original"
        cleaned[raw_column] = cleaned[column]
        q1, q3, lower, upper = detect_iqr_bounds(cleaned[column])
        outliers_before = int(((cleaned[column] < lower) | (cleaned[column] > upper)).sum())
        cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
        outliers_after = int(((cleaned[column] < lower) | (cleaned[column] > upper)).sum())
        report_rows.append(
            {
                "column": column,
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "lower_bound": round(lower, 3),
                "upper_bound": round(upper, 3),
                "outliers_before_capping": outliers_before,
                "outliers_after_capping": outliers_after,
            }
        )
    return cleaned, pd.DataFrame(report_rows)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create movie-specific features for analysis."""
    processed = df.copy()
    processed["release_year"] = processed["release_date"].dt.year
    processed["release_month"] = processed["release_date"].dt.month
    processed["release_month_name"] = processed["release_date"].dt.month_name()
    processed["release_decade"] = (processed["release_year"] // 10 * 10).astype(int).astype(str) + "s"
    processed["genre_list"] = processed["genre"].astype("string").str.split(",")
    processed["primary_genre"] = processed["genre_list"].apply(lambda genres: genres[0].strip() if isinstance(genres, list) and genres else "Unknown")
    processed["genre_count"] = processed["genre_list"].apply(lambda genres: len(genres) if isinstance(genres, list) else 0)
    processed["overview_word_count"] = processed["overview"].astype("string").str.split().str.len().fillna(0).astype(int)
    processed["language_name"] = processed["original_language"].astype("string").str.lower().map(LANGUAGE_MAP).fillna(processed["original_language"].astype("string").str.upper())
    processed["is_english"] = np.where(processed["original_language"].astype("string").str.lower() == "en", "English", "Non-English")
    processed["movie_age"] = 2026 - processed["release_year"]
    processed["is_recent"] = np.where(processed["release_year"] >= 2015, "Recent Movie", "Older Movie")
    processed["rating_category"] = pd.cut(
        processed["vote_average"],
        bins=[-0.01, 4.99, 6.49, 7.49, 10],
        labels=["Low Rated", "Average", "Good", "Excellent"],
    ).astype("string")
    processed["popularity_level"] = pd.qcut(
        processed["popularity"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
    ).astype("string")
    processed["weighted_score"] = (
        (processed["vote_average"] * np.log1p(processed["vote_count"])) / np.log1p(processed["vote_count"].max())
    ).round(3)
    processed["release_date"] = processed["release_date"].dt.date
    return processed


def validate_cleaned_data(df: pd.DataFrame) -> dict[str, object]:
    """Create final quality checks for the cleaned movie dataset."""
    duplicate_check_df = df.drop(columns=["genre_list"], errors="ignore")
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(duplicate_check_df.duplicated().sum()),
        "duplicate_title_release_rows": int(df.duplicated(subset=["title", "release_date"]).sum()),
        "invalid_vote_average_rows": int(((df["vote_average"] < 0) | (df["vote_average"] > 10)).sum()),
        "negative_popularity_rows": int((df["popularity"] < 0).sum()),
        "date_range": f"{df['release_date'].min()} to {df['release_date'].max()}",
    }


def clean_movie_data(raw_df: pd.DataFrame, load_method: str) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Run the complete MovieDB cleaning and processing workflow."""
    summary: dict[str, object] = {
        "load_method": load_method,
        "raw_shape": tuple(raw_df.shape),
        "missing_values_before": raw_df.isna().sum().to_dict(),
        "duplicates_before": int(raw_df.duplicated().sum()),
    }
    df = standardize_column_names(raw_df)
    df = clean_text_columns(df)
    df = fix_data_types(df)
    df, unusable_rows_removed = remove_unusable_rows(df)
    summary["unusable_rows_removed"] = unusable_rows_removed
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop_duplicates(subset=["title", "release_date"]).reset_index(drop=True)
    df = fill_missing_values(df)
    df, outlier_report = cap_outliers_iqr(df, ["popularity", "vote_count", "vote_average"])
    df = engineer_features(df)

    summary["clean_shape"] = tuple(df.shape)
    summary["missing_values_after"] = df.isna().sum().to_dict()
    summary["duplicates_after"] = int(df.drop(columns=["genre_list"], errors="ignore").duplicated().sum())
    summary["validation"] = validate_cleaned_data(df)
    return df, summary, outlier_report


def save_chart(fig: plt.Figure, filename: str) -> str:
    """Save a chart image and close the figure."""
    fig.savefig(CHARTS_DIR / filename)
    plt.close(fig)
    return filename


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per movie genre for genre-level analysis."""
    columns = [
        "title",
        "release_year",
        "popularity",
        "popularity_original",
        "vote_count",
        "vote_count_original",
        "vote_average",
        "vote_average_original",
        "weighted_score",
        "genre_list",
    ]
    genre_df = df[columns].explode("genre_list").copy()
    genre_df["genre_name"] = genre_df["genre_list"].astype("string").str.strip()
    genre_df = genre_df.drop(columns=["genre_list"])
    genre_df = genre_df[genre_df["genre_name"].notna() & (genre_df["genre_name"] != "")]
    return genre_df


def create_visualizations(raw_df: pd.DataFrame, clean_df: pd.DataFrame, outlier_report: pd.DataFrame) -> list[dict[str, str]]:
    """Create the complete set of movie-specific visualizations."""
    chart_meta: list[dict[str, str]] = []
    genre_df = explode_genres(clean_df)

    missing = raw_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(missing) > 0:
        sns.barplot(x=missing.values, y=missing.index, hue=missing.index, palette="crest", legend=False, ax=ax)
        for patch in ax.patches:
            width = patch.get_width()
            ax.annotate(str(int(width)), (width, patch.get_y() + patch.get_height() / 2), va="center", xytext=(5, 0), textcoords="offset points")
    ax.set_title("Missing Values Before Cleaning")
    ax.set_xlabel("Missing Value Count")
    ax.set_ylabel("Column")
    chart_meta.append(
        {
            "filename": save_chart(fig, "01_missing_values_before_cleaning.png"),
            "title": "Missing Values Before Cleaning",
            "interpretation": "The raw movie dataset has a small number of missing values in title, overview, language, genre, poster URL, and numeric rating fields. These must be handled before analysis.",
        }
    )

    numeric_raw = standardize_column_names(raw_df)
    numeric_raw = clean_text_columns(numeric_raw)
    numeric_raw = fix_data_types(numeric_raw)
    numeric_raw, _ = remove_unusable_rows(numeric_raw)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, column, color in zip(axes, ["popularity", "vote_count", "vote_average"], ["#4C78A8", "#F58518", "#54A24B"]):
        sns.boxplot(x=numeric_raw[column], ax=ax, color=color)
        ax.set_title(f"Outlier Check: {column.replace('_', ' ').title()}")
        ax.set_xlabel(column.replace("_", " ").title())
    fig.suptitle("Boxplot-Based Outlier Detection Before IQR Capping", y=1.05, fontsize=17, fontweight="bold")
    chart_meta.append(
        {
            "filename": save_chart(fig, "02_outlier_boxplots_before_treatment.png"),
            "title": "Outlier Detection Before Treatment",
            "interpretation": "Popularity and vote count contain very large blockbuster values. IQR capping creates stable analysis columns while preserving original values separately.",
        }
    )

    top_popular = clean_df.sort_values("popularity_original", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top_popular, x="popularity_original", y="title", hue="title", palette="mako", legend=False, ax=ax)
    ax.set_title("Top 10 Movies by Original Popularity")
    ax.set_xlabel("Original Popularity")
    ax.set_ylabel("Movie")
    chart_meta.append(
        {
            "filename": save_chart(fig, "03_top_10_movies_by_popularity.png"),
            "title": "Top 10 Movies by Popularity",
            "interpretation": "The most popular movies are the highest-attention titles in the dataset. These films often represent strong public interest or platform visibility.",
        }
    )

    top_votes = clean_df.sort_values("vote_count_original", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top_votes, x="vote_count_original", y="title", hue="title", palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 10 Movies by Vote Count")
    ax.set_xlabel("Original Vote Count")
    ax.set_ylabel("Movie")
    chart_meta.append(
        {
            "filename": save_chart(fig, "04_top_10_movies_by_vote_count.png"),
            "title": "Top 10 Movies by Vote Count",
            "interpretation": "Vote count shows audience engagement. Movies with many votes have stronger evidence behind their rating than movies with only a few votes.",
        }
    )

    qualified = clean_df[clean_df["vote_count_original"] >= clean_df["vote_count_original"].quantile(0.75)]
    top_rated = qualified.sort_values("vote_average_original", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top_rated, x="vote_average_original", y="title", hue="title", palette="crest", legend=False, ax=ax)
    ax.set_title("Top Rated Movies with High Vote Count")
    ax.set_xlabel("Original Vote Average")
    ax.set_ylabel("Movie")
    ax.set_xlim(0, 10)
    chart_meta.append(
        {
            "filename": save_chart(fig, "05_top_rated_qualified_movies.png"),
            "title": "Top Rated Movies with High Vote Count",
            "interpretation": "Filtering by vote count avoids ranking movies highly just because very few people rated them. This produces a more reliable top-rated list.",
        }
    )

    genre_counts = genre_df["genre_name"].value_counts().head(15).reset_index()
    genre_counts.columns = ["genre_name", "movie_count"]
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=genre_counts, x="movie_count", y="genre_name", hue="genre_name", palette="rocket", legend=False, ax=ax)
    ax.set_title("Top 15 Genres by Movie Count")
    ax.set_xlabel("Number of Movies")
    ax.set_ylabel("Genre")
    chart_meta.append(
        {
            "filename": save_chart(fig, "06_top_15_genres_by_count.png"),
            "title": "Top 15 Genres by Movie Count",
            "interpretation": "Genre count shows which types of movies appear most often. Drama, comedy, action, and thriller-type genres often dominate large movie databases.",
        }
    )

    genre_popularity = genre_df.groupby("genre_name", as_index=False).agg(avg_popularity=("popularity_original", "mean"), movie_count=("title", "count"))
    genre_popularity = genre_popularity[genre_popularity["movie_count"] >= 30].sort_values("avg_popularity", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=genre_popularity, x="avg_popularity", y="genre_name", hue="genre_name", palette="flare", legend=False, ax=ax)
    ax.set_title("Average Popularity by Genre")
    ax.set_xlabel("Average Original Popularity")
    ax.set_ylabel("Genre")
    chart_meta.append(
        {
            "filename": save_chart(fig, "07_average_popularity_by_genre.png"),
            "title": "Average Popularity by Genre",
            "interpretation": "This chart shows which genres attract higher average attention. It is useful for understanding audience demand by content type.",
        }
    )

    language_counts = clean_df["language_name"].value_counts().head(12).reset_index()
    language_counts.columns = ["language_name", "movie_count"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=language_counts, x="language_name", y="movie_count", hue="language_name", palette="cubehelix", legend=False, ax=ax)
    ax.set_title("Top Languages by Movie Count")
    ax.set_xlabel("Language")
    ax.set_ylabel("Number of Movies")
    ax.tick_params(axis="x", rotation=35)
    chart_meta.append(
        {
            "filename": save_chart(fig, "08_top_languages_by_movie_count.png"),
            "title": "Top Languages by Movie Count",
            "interpretation": "Language distribution shows dataset coverage. English movies dominate, while Japanese, Spanish, French, Korean, and other languages add international variety.",
        }
    )

    year_counts = clean_df.groupby("release_year", as_index=False).agg(movie_count=("title", "count"))
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=year_counts, x="release_year", y="movie_count", linewidth=2.2, color="#2F6F73", ax=ax)
    ax.set_title("Number of Movie Releases by Year")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Movie Count")
    chart_meta.append(
        {
            "filename": save_chart(fig, "09_movie_releases_by_year.png"),
            "title": "Movie Releases by Year",
            "interpretation": "Release trends show how the dataset changes over time. Recent years contain many movies because modern movie databases usually have richer coverage.",
        }
    )

    decade_counts = clean_df.groupby("release_decade", as_index=False).agg(movie_count=("title", "count"))
    decade_counts["decade_num"] = decade_counts["release_decade"].str.replace("s", "", regex=False).astype(int)
    decade_counts = decade_counts.sort_values("decade_num")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=decade_counts, x="release_decade", y="movie_count", hue="release_decade", palette="magma", legend=False, ax=ax)
    ax.set_title("Movie Count by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Number of Movies")
    ax.tick_params(axis="x", rotation=45)
    chart_meta.append(
        {
            "filename": save_chart(fig, "10_movie_count_by_decade.png"),
            "title": "Movie Count by Decade",
            "interpretation": "Decade analysis summarizes long-term coverage. It helps compare older cinema representation with modern releases.",
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(clean_df["vote_average"], bins=30, kde=True, color="#4C78A8", ax=ax)
    ax.set_title("Distribution of Movie Ratings")
    ax.set_xlabel("Vote Average")
    ax.set_ylabel("Number of Movies")
    chart_meta.append(
        {
            "filename": save_chart(fig, "11_rating_distribution_histogram.png"),
            "title": "Rating Distribution",
            "interpretation": "Most movies cluster around middle-to-good ratings. Very low and perfect ratings are less common after cleaning.",
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(clean_df["popularity"], bins=35, kde=True, color="#D19C2C", ax=ax)
    ax.set_title("Distribution of Capped Popularity")
    ax.set_xlabel("Popularity After IQR Capping")
    ax.set_ylabel("Number of Movies")
    chart_meta.append(
        {
            "filename": save_chart(fig, "12_popularity_distribution_histogram.png"),
            "title": "Popularity Distribution",
            "interpretation": "Most movies have modest popularity, while a smaller group receives much higher attention. Capping makes the distribution easier to study.",
        }
    )

    scatter_sample = clean_df.sample(n=min(1200, len(clean_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=scatter_sample, x="vote_count_original", y="popularity_original", hue="is_english", alpha=0.65, ax=ax)
    ax.set_title("Vote Count vs Popularity")
    ax.set_xlabel("Original Vote Count")
    ax.set_ylabel("Original Popularity")
    ax.legend(title="Language Group")
    chart_meta.append(
        {
            "filename": save_chart(fig, "13_vote_count_vs_popularity_scatter.png"),
            "title": "Vote Count vs Popularity",
            "interpretation": "Popular movies often receive more votes, but the relationship is not perfect. Some movies are highly rated or voted without being the most popular at the moment.",
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=scatter_sample, x="overview_word_count", y="vote_average_original", hue="rating_category", alpha=0.65, ax=ax)
    ax.set_title("Overview Length vs Rating")
    ax.set_xlabel("Overview Word Count")
    ax.set_ylabel("Original Vote Average")
    ax.legend(title="Rating Category")
    chart_meta.append(
        {
            "filename": save_chart(fig, "14_overview_length_vs_rating_scatter.png"),
            "title": "Overview Length vs Rating",
            "interpretation": "Overview length does not strongly determine rating, but this chart checks whether richer descriptions are associated with different rating patterns.",
        }
    )

    rating_counts = clean_df["rating_category"].value_counts().reindex(["Low Rated", "Average", "Good", "Excellent"]).dropna()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pie(rating_counts.values, labels=rating_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
    ax.set_title("Rating Category Share")
    chart_meta.append(
        {
            "filename": save_chart(fig, "15_rating_category_pie_chart.png"),
            "title": "Rating Category Share",
            "interpretation": "The pie chart groups movies into simple rating bands. This makes it easy to explain the overall quality mix of the dataset.",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=clean_df, x="primary_genre", y="vote_average", hue="primary_genre", palette="Set2", legend=False, ax=ax)
    ax.set_title("Rating Distribution by Primary Genre")
    ax.set_xlabel("Primary Genre")
    ax.set_ylabel("Vote Average")
    ax.tick_params(axis="x", rotation=45)
    chart_meta.append(
        {
            "filename": save_chart(fig, "16_rating_by_primary_genre_boxplot.png"),
            "title": "Rating by Primary Genre",
            "interpretation": "Boxplots compare rating ranges across genres. They show whether some genres tend to have higher median ratings or more variation.",
        }
    )

    corr_columns = ["popularity", "vote_count", "vote_average", "weighted_score", "genre_count", "overview_word_count", "movie_age"]
    corr = clean_df[corr_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    chart_meta.append(
        {
            "filename": save_chart(fig, "17_correlation_heatmap.png"),
            "title": "Correlation Heatmap",
            "interpretation": "The heatmap shows relationships between numeric features. Popularity and vote count usually move together, while rating behaves differently.",
        }
    )

    pair_columns = ["popularity", "vote_count", "vote_average", "weighted_score"]
    pair_sample = clean_df[pair_columns + ["is_english"]].sample(n=min(500, len(clean_df)), random_state=42)
    pair = sns.pairplot(pair_sample, vars=pair_columns, hue="is_english", diag_kind="hist", corner=True, plot_kws={"alpha": 0.65, "s": 22})
    pair.fig.suptitle("Pairplot of Key Movie Metrics", y=1.02, fontsize=16, fontweight="bold")
    pair.fig.savefig(CHARTS_DIR / "18_pairplot_key_movie_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(pair.fig)
    chart_meta.append(
        {
            "filename": "18_pairplot_key_movie_metrics.png",
            "title": "Pairplot of Key Movie Metrics",
            "interpretation": "The pairplot gives a compact view of multiple relationships at once. It is useful for spotting clusters, patterns, and unusual movies.",
        }
    )

    return chart_meta


def create_summary_tables(clean_df: pd.DataFrame, outlier_report: pd.DataFrame) -> None:
    """Export summary tables used by the notebook and report."""
    genre_df = explode_genres(clean_df)
    tables = {
        "genre_summary": genre_df.groupby("genre_name", as_index=False).agg(
            movie_count=("title", "count"),
            avg_popularity=("popularity_original", "mean"),
            avg_rating=("vote_average_original", "mean"),
            avg_vote_count=("vote_count_original", "mean"),
        ).sort_values("movie_count", ascending=False),
        "language_summary": clean_df.groupby("language_name", as_index=False).agg(
            movie_count=("title", "count"),
            avg_popularity=("popularity_original", "mean"),
            avg_rating=("vote_average_original", "mean"),
        ).sort_values("movie_count", ascending=False),
        "year_summary": clean_df.groupby("release_year", as_index=False).agg(
            movie_count=("title", "count"),
            avg_popularity=("popularity_original", "mean"),
            avg_rating=("vote_average_original", "mean"),
        ).sort_values("release_year"),
        "top_movies_by_weighted_score": clean_df.sort_values("weighted_score", ascending=False)[
            ["title", "release_year", "primary_genre", "language_name", "vote_average_original", "vote_count_original", "popularity_original", "weighted_score"]
        ].head(25),
        "outlier_report": outlier_report,
    }
    for name, table in tables.items():
        rounded = table.copy()
        for column in rounded.select_dtypes(include=["float", "float64", "float32"]).columns:
            rounded[column] = rounded[column].round(3)
        rounded.to_csv(TABLES_DIR / f"{name}.csv", index=False)


def build_insights(clean_df: pd.DataFrame, summary: dict[str, object], outlier_report: pd.DataFrame) -> dict[str, object]:
    """Calculate metrics, insights, and recommendations."""
    genre_df = explode_genres(clean_df)
    top_popular_movie = clean_df.sort_values("popularity_original", ascending=False).iloc[0]
    top_voted_movie = clean_df.sort_values("vote_count_original", ascending=False).iloc[0]
    qualified = clean_df[clean_df["vote_count_original"] >= clean_df["vote_count_original"].quantile(0.75)]
    top_rated_movie = qualified.sort_values("vote_average_original", ascending=False).iloc[0]
    top_genre = genre_df["genre_name"].value_counts().idxmax()
    top_language = clean_df["language_name"].value_counts().idxmax()
    most_common_decade = clean_df["release_decade"].value_counts().idxmax()
    best_avg_genre = (
        genre_df.groupby("genre_name")
        .agg(avg_rating=("vote_average_original", "mean"), movie_count=("title", "count"))
        .query("movie_count >= 30")
        .sort_values("avg_rating", ascending=False)
        .index[0]
    )
    recent_share = float((clean_df["release_year"] >= 2015).mean())
    english_share = float((clean_df["original_language"].astype("string").str.lower() == "en").mean())

    metrics = {
        "movie_count": int(len(clean_df)),
        "feature_count": int(clean_df.shape[1]),
        "average_rating": float(clean_df["vote_average_original"].mean()),
        "median_rating": float(clean_df["vote_average_original"].median()),
        "average_popularity": float(clean_df["popularity_original"].mean()),
        "top_popular_movie": str(top_popular_movie["title"]),
        "top_voted_movie": str(top_voted_movie["title"]),
        "top_rated_qualified_movie": str(top_rated_movie["title"]),
        "top_genre": str(top_genre),
        "top_language": str(top_language),
        "most_common_decade": str(most_common_decade),
        "best_avg_genre": str(best_avg_genre),
        "recent_share": recent_share,
        "english_share": english_share,
        "rows_removed": int(summary["unusable_rows_removed"]),
        "outlier_columns_treated": outlier_report["column"].tolist(),
    }
    insights = [
        f"The cleaned dataset contains {metrics['movie_count']:,} valid movies across {metrics['feature_count']} columns.",
        f"{metrics['top_popular_movie']} has the highest popularity score.",
        f"{metrics['top_voted_movie']} has the highest vote count.",
        f"{metrics['top_rated_qualified_movie']} stands out after filtering for movies with strong vote counts.",
        f"{metrics['top_genre']} is the most common genre in the dataset.",
        f"{metrics['best_avg_genre']} has the highest average rating among genres with enough records for comparison.",
        f"{metrics['top_language']} is the dominant language, covering {metrics['english_share'] * 100:.1f}% of the cleaned dataset.",
        f"The {metrics['most_common_decade']} contain the largest number of movies in this dataset.",
        "Popularity and rating measure different things. A movie can be widely watched without being the highest rated.",
    ]
    recommendations = [
        "Use weighted score when ranking movies so both rating and vote count are considered.",
        "Compare genres separately because audience behavior changes by content type.",
        "Consider release year when comparing older and newer movies.",
        "Use language analysis when studying international content demand.",
        "Use original popularity and vote-count values for top-movie lists, but capped values for cleaner statistical charts.",
    ]
    return {"metrics": metrics, "insights": insights, "recommendations": recommendations}


def pct(value: float) -> str:
    """Format a decimal share as a percent string."""
    return f"{value * 100:.1f}%"


def write_dashboard(insight_bundle: dict[str, object], chart_meta: list[dict[str, str]]) -> None:
    """Write a polished static dashboard as HTML."""
    metrics = insight_bundle["metrics"]
    insights_list = "\n".join(f"<li>{item}</li>" for item in insight_bundle["insights"])
    recommendations_list = "\n".join(f"<li>{item}</li>" for item in insight_bundle["recommendations"])
    chart_cards = "\n".join(
        f"""
        <article class="chart-card">
          <img src="../charts/{chart['filename']}" alt="{chart['title']}">
          <h3>{chart['title']}</h3>
          <p>{chart['interpretation']}</p>
        </article>
        """
        for chart in chart_meta[2:]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MovieDB Data Cleaning and Visualization Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --ink: #151a20;
      --muted: #5e6874;
      --panel: #ffffff;
      --accent: #2f6f73;
      --accent2: #9b5f3d;
      --line: #dce1e8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.55;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 38px 7vw 28px;
    }}
    header h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: 0;
    }}
    header p {{
      max-width: 920px;
      margin: 0;
      color: var(--muted);
      font-size: 17px;
    }}
    main {{ padding: 28px 7vw 48px; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi, .section, .chart-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .kpi {{ padding: 18px; }}
    .kpi span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    .kpi strong {{
      display: block;
      font-size: 24px;
      color: var(--accent);
    }}
    .section {{
      padding: 22px;
      margin-bottom: 24px;
    }}
    .section h2 {{ margin: 0 0 12px; font-size: 24px; }}
    .section li {{ margin: 8px 0; }}
    .explain-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .explain-item {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fbfcfe;
    }}
    .explain-item h3 {{ margin: 0 0 8px; color: var(--accent); font-size: 17px; }}
    .explain-item p {{ margin: 0; color: var(--muted); font-size: 14px; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 20px;
    }}
    .chart-card {{ overflow: hidden; }}
    .chart-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    .chart-card h3 {{ margin: 16px 18px 8px; color: var(--accent); font-size: 18px; }}
    .chart-card p {{ margin: 0 18px 18px; color: var(--muted); font-size: 14px; }}
    footer {{
      background: #fff;
      border-top: 1px solid var(--line);
      padding: 24px 7vw;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>MovieDB Data Cleaning and Visualization Dashboard</h1>
    <p>A cleaned and visualized MovieDB dataset showing movie popularity, ratings, genres, languages, and release trends.</p>
  </header>
  <main>
    <section class="kpis">
      <div class="kpi"><span>Clean Movies</span><strong>{metrics['movie_count']:,}</strong></div>
      <div class="kpi"><span>Average Rating</span><strong>{metrics['average_rating']:.2f}</strong></div>
      <div class="kpi"><span>Top Genre</span><strong>{metrics['top_genre']}</strong></div>
      <div class="kpi"><span>Top Language</span><strong>{metrics['top_language']}</strong></div>
      <div class="kpi"><span>Recent Share</span><strong>{pct(metrics['recent_share'])}</strong></div>
      <div class="kpi"><span>English Share</span><strong>{pct(metrics['english_share'])}</strong></div>
    </section>
    <section class="section">
      <h2>About This Project</h2>
      <p>The raw CSV was cleaned, converted into analysis-ready columns, and used to build charts around audience interest and movie quality. This is the same kind of workflow a streaming or review platform could use before building dashboards or recommendations.</p>
    </section>
    <section class="section"><h2>Key Insights</h2><ul>{insights_list}</ul></section>
    <section class="section"><h2>Recommendations</h2><ul>{recommendations_list}</ul></section>
    <section class="chart-grid">{chart_cards}</section>
  </main>
  <footer>Generated by Python from the local mymoviedb.csv dataset.</footer>
</body>
</html>
"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")


def write_insights_report(insight_bundle: dict[str, object], summary: dict[str, object], outlier_report: pd.DataFrame, chart_meta: list[dict[str, str]]) -> None:
    """Write a concise markdown insight report."""
    metrics = insight_bundle["metrics"]
    insight_lines = "\n".join(f"- {item}" for item in insight_bundle["insights"])
    recommendation_lines = "\n".join(f"- {item}" for item in insight_bundle["recommendations"])
    chart_lines = "\n".join(f"- {i + 1}. {chart['title']}: `{chart['filename']}`" for i, chart in enumerate(chart_meta))
    outlier_lines = "\n".join(
        f"- {row.column}: {int(row.outliers_before_capping)} outliers capped using bounds {row.lower_bound} to {row.upper_bound}"
        for row in outlier_report.itertuples()
    )
    markdown = f"""# MovieDB Data Cleaning and Visualization Project

## Overview

This project uses the real `mymoviedb.csv` file to analyze movie ratings, popularity, genres, languages, and release patterns. The raw data was cleaned first so the final charts and insights are based on reliable values.

The dataset reflects a common real-world problem: data usually does not arrive perfectly clean. Before analysis, dates need to be parsed, numeric columns need to be converted, missing values need to be handled, and extreme values need to be checked.

## Data Cleaning Summary

- Raw dataset shape: {summary['raw_shape']}
- Cleaned dataset shape: {summary['clean_shape']}
- CSV loading method used: {summary['load_method']}
- Unusable rows removed: {summary['unusable_rows_removed']}
- Missing values after cleaning: {summary['validation']['missing_values_total']}

## Key Metrics

- Clean movies: {metrics['movie_count']:,}
- Average rating: {metrics['average_rating']:.2f}
- Median rating: {metrics['median_rating']:.2f}
- Top popular movie: {metrics['top_popular_movie']}
- Highest vote-count movie: {metrics['top_voted_movie']}
- Top qualified rated movie: {metrics['top_rated_qualified_movie']}
- Most common genre: {metrics['top_genre']}
- Dominant language: {metrics['top_language']}
- Most common decade: {metrics['most_common_decade']}

## Outlier Treatment

{outlier_lines}

## Key Insights

{insight_lines}

## Recommendations

{recommendation_lines}

## Exported Visualizations

{chart_lines}

## Conclusion

The cleaned MovieDB dataset is ready for reporting and further analysis. The project shows how raw movie data can be turned into a clean dataset, visual dashboard, and practical insights about audience interest, ratings, genres, and release trends.
"""
    INSIGHTS_REPORT_PATH.write_text(markdown, encoding="utf-8")


def write_project_summary(insight_bundle: dict[str, object], summary: dict[str, object], chart_meta: list[dict[str, str]]) -> None:
    """Write machine-readable project metadata."""
    payload = {
        "project_title": "MovieDB Data Cleaning and Visualization Project",
        "raw_dataset": str(RAW_DATA_PATH.relative_to(BASE_DIR)),
        "cleaned_dataset": str(CLEAN_DATA_PATH.relative_to(BASE_DIR)),
        "notebook": str(NOTEBOOK_PATH.relative_to(BASE_DIR)),
        "dashboard": str(DASHBOARD_PATH.relative_to(BASE_DIR)),
        "insights_report": str(INSIGHTS_REPORT_PATH.relative_to(BASE_DIR)),
        "cleaning_summary": {
            "raw_shape": list(summary["raw_shape"]),
            "clean_shape": list(summary["clean_shape"]),
            "load_method": summary["load_method"],
            "unusable_rows_removed": summary["unusable_rows_removed"],
            "validation": summary["validation"],
        },
        "metrics": insight_bundle["metrics"],
        "visualizations": chart_meta,
    }
    PROJECT_SUMMARY_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def md(text: str) -> nbf.NotebookNode:
    """Create a markdown notebook cell."""
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    """Create a code notebook cell."""
    return nbf.v4.new_code_cell(dedent(text).strip())


def chart_display_code(filename: str) -> str:
    """Notebook code to display an exported chart."""
    return f"""
from IPython.display import Image, display

display(Image(filename='../outputs/charts/{filename}'))
"""


def create_notebook(summary: dict[str, object], outlier_report: pd.DataFrame, insight_bundle: dict[str, object], chart_meta: list[dict[str, str]]) -> None:
    """Generate the polished Jupyter Notebook."""
    metrics = insight_bundle["metrics"]
    validation = summary["validation"]
    missing_before = pd.Series(summary["missing_values_before"]).sort_values(ascending=False)
    missing_text = missing_before[missing_before > 0].to_string()
    outlier_table = outlier_report.to_markdown(index=False)

    chart_cells: list[nbf.NotebookNode] = []
    for i, chart in enumerate(chart_meta, start=1):
        chart_cells.extend(
            [
                md(
                    f"""
                    ### Visualization {i}: {chart['title']}

                    """
                ),
                code(chart_display_code(chart["filename"])),
                md(f"**Interpretation:** {chart['interpretation']}"),
            ]
        )

    cells = [
        md(
            """
            # MovieDB Data Cleaning and Visualization Project

            **Project type:** Data Cleaning and Visualization  
            **Dataset:** Real MovieDB CSV dataset  
            **Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn  
            **Goal:** Convert raw movie data into a clean dataset, charts, insights, recommendations, and a visual report.

            ## 1. Project Title

            MovieDB Data Cleaning and Visualization Project
            """
        ),
        md(
            """
            ## Project Overview

            Raw datasets usually need cleaning before they can be trusted. In this project, I cleaned the MovieDB CSV, fixed missing and incorrect values, created useful features, and built charts to understand the movie collection.

            This kind of analysis is useful for movie websites and streaming platforms because it helps compare genres, track audience interest, and support recommendation ideas.
            """
        ),
        md(
            """
            ## 2. Problem Statement

            The dataset contains movie details such as release date, title, overview, popularity, vote count, rating, language, genre, and poster URL. The goal is to clean the raw file and use it to answer practical questions about movie performance and audience interest.

            The workflow covers loading, cleaning, feature engineering, exploratory analysis, visualization, insights, and final recommendations.
            """
        ),
        md("## 3. Import Libraries\n\nSet up the main Python libraries used for cleaning, analysis, and visualization."),
        code(
            """
            import os
            from pathlib import Path

            PROJECT_ROOT = Path("..").resolve()
            os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "matplotlib_cache"))

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme(style="whitegrid", context="notebook")
            pd.set_option("display.max_columns", 50)
            """
        ),
        md(
            """
            ## 4. Load Dataset

            The CSV is loaded with a fallback method because long text fields can sometimes break the default reader. This keeps the project stable on real data.
            """
        ),
        code(
            """
            raw_path = PROJECT_ROOT / "data" / "raw" / "mymoviedb_raw.csv"

            def load_movie_data(path):
                try:
                    df = pd.read_csv(path)
                    method = "default"
                except Exception:
                    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
                    method = "python_engine"
                return df, method

            raw_df, load_method = load_movie_data(raw_path)
            print("Loaded with:", load_method)
            print("Shape:", raw_df.shape)
            raw_df.head()
            """
        ),
        md("## 5. Dataset Overview\n\nBefore cleaning, we inspect rows, columns, data types, missing values, duplicates, and sample records."),
        code(
            """
            print("Rows and columns:", raw_df.shape)
            print("Columns:", raw_df.columns.tolist())
            display(raw_df.head())
            display(raw_df.info())
            display(raw_df.describe(include="all").T)
            """
        ),
        code(
            """
            print("Missing values:")
            display(raw_df.isna().sum().sort_values(ascending=False))

            print("Duplicate rows:", raw_df.duplicated().sum())
            """
        ),
        md("## 6. Data Cleaning\n\nClean the dataset so the final analysis is based on reliable values."),
        code(
            """
            def standardize_column_names(df):
                cleaned = df.copy()
                cleaned.columns = (
                    cleaned.columns.str.strip()
                    .str.lower()
                    .str.replace(r"[^a-z0-9]+", "_", regex=True)
                    .str.strip("_")
                )
                return cleaned

            df = standardize_column_names(raw_df)
            df.columns.tolist()
            """
        ),
        code(
            """
            def clean_text_columns(df):
                cleaned = df.copy()
                placeholders = {"": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan, "n/a": np.nan, "na": np.nan, "unknown": np.nan}
                for column in cleaned.select_dtypes(include=["object", "string"]).columns:
                    cleaned[column] = cleaned[column].astype("string").str.strip()
                    cleaned[column] = cleaned[column].replace(placeholders)
                return cleaned

            df = clean_text_columns(df)
            print("Text columns cleaned.")
            """
        ),
        code(
            """
            def fix_data_types(df):
                cleaned = df.copy()
                cleaned["release_date"] = pd.to_datetime(cleaned["release_date"], errors="coerce", format="mixed")
                for column in ["popularity", "vote_count", "vote_average"]:
                    cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
                return cleaned

            df = fix_data_types(df)
            display(df.dtypes)
            """
        ),
        code(
            """
            before_rows = len(df)
            df = df[df["release_date"].notna() & df["title"].notna()].copy()
            print("Unusable rows removed:", before_rows - len(df))

            duplicate_before = df.duplicated().sum()
            df = df.drop_duplicates().reset_index(drop=True)
            df = df.drop_duplicates(subset=["title", "release_date"]).reset_index(drop=True)
            print("Exact duplicates before removal:", duplicate_before)
            print("Duplicates after removal:", df.duplicated().sum())
            """
        ),
        md("### Missing Value Handling\n\nFill missing text values with clear labels and numeric values with the median."),
        code(
            """
            df["overview"] = df["overview"].fillna("Overview not available")
            df["genre"] = df["genre"].fillna("Unknown")
            df["poster_url"] = df["poster_url"].fillna("Poster not available")
            df["original_language"] = df["original_language"].fillna("unknown")

            for column in ["popularity", "vote_count", "vote_average"]:
                df[column] = df[column].fillna(df[column].median())

            df["vote_count"] = df["vote_count"].round().astype(int)
            df["vote_average"] = df["vote_average"].clip(0, 10)
            df["popularity"] = df["popularity"].clip(lower=0)

            print("Missing values after filling:", df.isna().sum().sum())
            """
        ),
        md("### Outlier Detection and Treatment\n\nWe use boxplots and IQR. Outliers are capped so extreme blockbuster values do not dominate normal charts."),
        code(
            """
            def detect_iqr_bounds(series):
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                return q1, q3, lower, upper

            def cap_outliers_iqr(df, columns):
                cleaned = df.copy()
                report = []
                for column in columns:
                    cleaned[f"{column}_original"] = cleaned[column]
                    q1, q3, lower, upper = detect_iqr_bounds(cleaned[column])
                    outlier_count = ((cleaned[column] < lower) | (cleaned[column] > upper)).sum()
                    cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
                    report.append({"column": column, "lower_bound": lower, "upper_bound": upper, "outliers_before_capping": int(outlier_count)})
                return cleaned, pd.DataFrame(report)

            df, outlier_report = cap_outliers_iqr(df, ["popularity", "vote_count", "vote_average"])
            display(outlier_report)
            """
        ),
        md(f"**Expected outlier report from generated project:**\n\n{outlier_table}"),
        md("## 7. Data Processing and Feature Engineering\n\nCreate useful columns such as release year, decade, primary genre, language name, movie age, rating category, and weighted score."),
        code(
            """
            language_map = {
                "en": "English", "ja": "Japanese", "es": "Spanish", "fr": "French", "ko": "Korean",
                "zh": "Chinese", "cn": "Chinese", "it": "Italian", "ru": "Russian", "de": "German",
                "pt": "Portuguese", "hi": "Hindi"
            }

            df["release_year"] = df["release_date"].dt.year
            df["release_month"] = df["release_date"].dt.month
            df["release_month_name"] = df["release_date"].dt.month_name()
            df["release_decade"] = (df["release_year"] // 10 * 10).astype(int).astype(str) + "s"
            df["genre_list"] = df["genre"].astype("string").str.split(",")
            df["primary_genre"] = df["genre_list"].apply(lambda genres: genres[0].strip() if isinstance(genres, list) and genres else "Unknown")
            df["genre_count"] = df["genre_list"].apply(lambda genres: len(genres) if isinstance(genres, list) else 0)
            df["overview_word_count"] = df["overview"].astype("string").str.split().str.len().fillna(0).astype(int)
            df["language_name"] = df["original_language"].astype("string").str.lower().map(language_map).fillna(df["original_language"].astype("string").str.upper())
            df["is_english"] = np.where(df["original_language"].astype("string").str.lower() == "en", "English", "Non-English")
            df["movie_age"] = 2026 - df["release_year"]
            df["is_recent"] = np.where(df["release_year"] >= 2015, "Recent Movie", "Older Movie")
            df["rating_category"] = pd.cut(df["vote_average"], bins=[-0.01, 4.99, 6.49, 7.49, 10], labels=["Low Rated", "Average", "Good", "Excellent"]).astype("string")
            df["popularity_level"] = pd.qcut(df["popularity"].rank(method="first"), q=4, labels=["Low", "Medium", "High", "Very High"]).astype("string")
            df["weighted_score"] = ((df["vote_average"] * np.log1p(df["vote_count"])) / np.log1p(df["vote_count"].max())).round(3)
            display(df.head())
            """
        ),
        md("### Validate Cleaned Data\n\nCheck that the cleaned dataset has no missing values, no duplicate title/date records, and valid numeric ranges."),
        code(
            """
            validation = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "missing_values_total": df.isna().sum().sum(),
                "duplicate_rows": df.drop(columns=["genre_list"], errors="ignore").duplicated().sum(),
                "invalid_vote_average_rows": ((df["vote_average"] < 0) | (df["vote_average"] > 10)).sum(),
                "negative_popularity_rows": (df["popularity"] < 0).sum(),
            }
            validation
            """
        ),
        md("## 8. Exploratory Data Analysis\n\nEDA means exploring the dataset to find patterns before giving final conclusions."),
        code(
            """
            display(df[["popularity", "vote_count", "vote_average", "weighted_score", "genre_count", "overview_word_count", "movie_age"]].describe().T)
            display(df[["language_name", "primary_genre", "rating_category", "popularity_level"]].describe().T)
            """
        ),
        code(
            """
            genre_df = df[["title", "release_year", "popularity", "vote_count", "vote_average", "weighted_score", "genre_list"]].explode("genre_list").copy()
            genre_df["genre_name"] = genre_df["genre_list"].astype("string").str.strip()

            genre_summary = genre_df.groupby("genre_name", as_index=False).agg(
                movie_count=("title", "count"),
                avg_popularity=("popularity", "mean"),
                avg_rating=("vote_average", "mean"),
            ).sort_values("movie_count", ascending=False)

            display(genre_summary.head(10))
            """
        ),
        md("## 9. Visualizations\n\nThe charts below show the main patterns found in the cleaned dataset."),
        *chart_cells,
        md("## 10. Insights"),
        md("\n".join(f"- {item}" for item in insight_bundle["insights"])),
        md("## 11. Recommendations"),
        md("\n".join(f"- {item}" for item in insight_bundle["recommendations"])),
        md(
            """
            ## 12. Conclusion

            This project cleaned and analyzed the MovieDB dataset from start to finish. The raw CSV was loaded safely, columns were standardized, text was cleaned, dates and numbers were converted, unusable rows were removed, missing values were handled, outliers were treated with IQR, and useful features were created.

            The analysis shows that movie popularity, vote count, rating, genre, language, and release year all tell different parts of the movie-data story. The final cleaned dataset is ready for dashboarding, reporting, portfolio submission, and future recommendation-system projects.
            """
        ),
        md("## 13. Export Cleaned Dataset"),
        code(
            """
            output_path = PROJECT_ROOT / "data" / "processed" / "mymoviedb_cleaned.csv"
            df.to_csv(output_path, index=False)
            print("Cleaned dataset exported to:", output_path)
            """
        ),
        md(
            f"""
            ## Project Summary

            - Project title: MovieDB Data Cleaning and Visualization Project
            - Raw dataset shape: `{summary['raw_shape']}`
            - Cleaned dataset shape: `{summary['clean_shape']}`
            - Cleaned movies: `{metrics['movie_count']:,}`
            - Average rating: `{metrics['average_rating']:.2f}`
            - Top popular movie: `{metrics['top_popular_movie']}`
            - Top genre: `{metrics['top_genre']}`
            - Top language: `{metrics['top_language']}`
            - Dashboard: `outputs/reports/movie_dashboard.html`
            """
        ),
    ]

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, NOTEBOOK_PATH)


def write_readme(insight_bundle: dict[str, object], summary: dict[str, object]) -> None:
    """Write the project README."""
    metrics = insight_bundle["metrics"]
    readme = f"""# MovieDB Data Cleaning and Visualization Project

This project cleans and explores a MovieDB dataset using Python.

## Overview

### What This Project Does

I cleaned and analyzed a movie dataset. I loaded the CSV file, checked the data, fixed problems, created useful columns, made charts, generated insights, wrote recommendations, and exported the cleaned dataset.

### Why It Matters

Raw data is often messy. It may contain missing values, wrong data types, unusable rows, text problems, or extreme values. Cleaning helps make the data accurate and trustworthy.

### Purpose

The purpose of this project is to convert raw movie data into useful information about popularity, ratings, genres, languages, release years, and audience engagement.

### Real-World Use

Streaming apps, movie websites, OTT platforms, and review systems use this kind of analysis to recommend movies, understand audience interest, compare genres, and decide which content to promote.

## Project Objective

The goal is to clean, process, visualize, and interpret the MovieDB dataset using Python.

## Folder Structure

```text
MovieDB_Data_Cleaning_Visualization_Project/
|-- data/
|   |-- raw/
|   |   |-- mymoviedb_raw.csv
|   |-- processed/
|       |-- mymoviedb_cleaned.csv
|-- notebooks/
|   |-- MovieDB_Data_Cleaning_Visualization_Project.ipynb
|-- outputs/
|   |-- charts/
|   |-- reports/
|   |   |-- movie_dashboard.html
|   |   |-- movie_insights_report.md
|   |   |-- project_summary.json
|   |-- tables/
|-- scripts/
|   |-- generate_movie_project_assets.py
|-- README.md
|-- requirements.txt
```

## Key Results

- Raw dataset shape: {summary['raw_shape']}
- Cleaned dataset shape: {summary['clean_shape']}
- Cleaned movies: {metrics['movie_count']:,}
- Average rating: {metrics['average_rating']:.2f}
- Top popular movie: {metrics['top_popular_movie']}
- Most common genre: {metrics['top_genre']}
- Dominant language: {metrics['top_language']}

## How to Run

```bash
python scripts/generate_movie_project_assets.py
```

## Deliverables

- Cleaned movie CSV dataset
- Professional Jupyter Notebook
- 18 exported charts
- Summary tables
- HTML dashboard
- Markdown insights report
- Final insights and conclusion
"""
    (BASE_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_requirements() -> None:
    """Write required Python packages."""
    (BASE_DIR / "requirements.txt").write_text("pandas\nnumpy\nmatplotlib\nseaborn\nnbformat\njupyter\n", encoding="utf-8")


def main() -> None:
    """Run the full generation pipeline."""
    configure_plot_style()
    print("Loading movie data...")
    raw_df, load_method = load_movie_data()
    print(f"Loaded shape: {raw_df.shape} using {load_method}")

    print("Cleaning movie data...")
    clean_df, summary, outlier_report = clean_movie_data(raw_df, load_method)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to: {CLEAN_DATA_PATH}")

    print("Creating charts...")
    chart_meta = create_visualizations(raw_df, clean_df, outlier_report)

    print("Creating summary tables...")
    create_summary_tables(clean_df, outlier_report)

    print("Building insights and reports...")
    insight_bundle = build_insights(clean_df, summary, outlier_report)
    write_dashboard(insight_bundle, chart_meta)
    write_insights_report(insight_bundle, summary, outlier_report, chart_meta)
    write_project_summary(insight_bundle, summary, chart_meta)
    write_readme(insight_bundle, summary)
    write_requirements()

    print("Creating Jupyter notebook...")
    create_notebook(summary, outlier_report, insight_bundle, chart_meta)

    print("\nMovieDB project generation complete.")
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Dashboard: {DASHBOARD_PATH}")
    print(f"Charts exported: {len(chart_meta)}")


if __name__ == "__main__":
    main()
