# MovieDB Data Cleaning and Visualization Project

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

- Raw dataset shape: (9837, 9)
- Cleaned dataset shape: (9827, 27)
- Cleaned movies: 9,827
- Average rating: 6.44
- Top popular movie: Spider-Man: No Way Home
- Most common genre: Drama
- Dominant language: English

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
