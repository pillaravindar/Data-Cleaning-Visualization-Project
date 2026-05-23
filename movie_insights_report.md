# MovieDB Data Cleaning and Visualization Project

## Overview

This project uses the real `mymoviedb.csv` file to analyze movie ratings, popularity, genres, languages, and release patterns. The raw data was cleaned first so the final charts and insights are based on reliable values.

The dataset reflects a common real-world problem: data usually does not arrive perfectly clean. Before analysis, dates need to be parsed, numeric columns need to be converted, missing values need to be handled, and extreme values need to be checked.

## Data Cleaning Summary

- Raw dataset shape: (9837, 9)
- Cleaned dataset shape: (9827, 27)
- CSV loading method used: python_engine
- Unusable rows removed: 10
- Missing values after cleaning: 0

## Key Metrics

- Clean movies: 9,827
- Average rating: 6.44
- Median rating: 6.50
- Top popular movie: Spider-Man: No Way Home
- Highest vote-count movie: Inception
- Top qualified rated movie: Dilwale Dulhania Le Jayenge
- Most common genre: Drama
- Dominant language: English
- Most common decade: 2010s

## Outlier Treatment

- popularity: 1048 outliers capped using bounds -12.44 to 63.743
- vote_count: 1129 outliers capped using bounds -1699.0 to 3221.0
- vote_average: 243 outliers capped using bounds 4.1 to 8.9

## Key Insights

- The cleaned dataset contains 9,827 valid movies across 27 columns.
- Spider-Man: No Way Home has the highest popularity score.
- Inception has the highest vote count.
- Dilwale Dulhania Le Jayenge stands out after filtering for movies with strong vote counts.
- Drama is the most common genre in the dataset.
- History has the highest average rating among genres with enough records for comparison.
- English is the dominant language, covering 77.0% of the cleaned dataset.
- The 2010s contain the largest number of movies in this dataset.
- Popularity and rating measure different things. A movie can be widely watched without being the highest rated.

## Recommendations

- Use weighted score when ranking movies so both rating and vote count are considered.
- Compare genres separately because audience behavior changes by content type.
- Consider release year when comparing older and newer movies.
- Use language analysis when studying international content demand.
- Use original popularity and vote-count values for top-movie lists, but capped values for cleaner statistical charts.

## Exported Visualizations

- 1. Missing Values Before Cleaning: `01_missing_values_before_cleaning.png`
- 2. Outlier Detection Before Treatment: `02_outlier_boxplots_before_treatment.png`
- 3. Top 10 Movies by Popularity: `03_top_10_movies_by_popularity.png`
- 4. Top 10 Movies by Vote Count: `04_top_10_movies_by_vote_count.png`
- 5. Top Rated Movies with High Vote Count: `05_top_rated_qualified_movies.png`
- 6. Top 15 Genres by Movie Count: `06_top_15_genres_by_count.png`
- 7. Average Popularity by Genre: `07_average_popularity_by_genre.png`
- 8. Top Languages by Movie Count: `08_top_languages_by_movie_count.png`
- 9. Movie Releases by Year: `09_movie_releases_by_year.png`
- 10. Movie Count by Decade: `10_movie_count_by_decade.png`
- 11. Rating Distribution: `11_rating_distribution_histogram.png`
- 12. Popularity Distribution: `12_popularity_distribution_histogram.png`
- 13. Vote Count vs Popularity: `13_vote_count_vs_popularity_scatter.png`
- 14. Overview Length vs Rating: `14_overview_length_vs_rating_scatter.png`
- 15. Rating Category Share: `15_rating_category_pie_chart.png`
- 16. Rating by Primary Genre: `16_rating_by_primary_genre_boxplot.png`
- 17. Correlation Heatmap: `17_correlation_heatmap.png`
- 18. Pairplot of Key Movie Metrics: `18_pairplot_key_movie_metrics.png`

## Conclusion

The cleaned MovieDB dataset is ready for reporting and further analysis. The project shows how raw movie data can be turned into a clean dataset, visual dashboard, and practical insights about audience interest, ratings, genres, and release trends.
