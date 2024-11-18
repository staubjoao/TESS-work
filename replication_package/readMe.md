# Advanced Topics in Software Engineering

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Scraping](#data-scraping)
  - [Data Processing](#data-processing)
  - [Data Analysis](#data-analysis)
  - [Visualizations](#visualizations)
- [Results](#results)
- [Contact](#contact)

## Overview

This project explores advanced topics in Software Engineering by analyzing the evolution and contributions of various GitHub repositories. Leveraging data scraped from GitHub, the project involves:

- **Data Collection:** Scraping commits, issues, and pull requests from selected repositories.
- **Data Processing:** Cleaning and enriching the collected data with additional metrics.
- **Data Analysis:** Conducting various analyses to understand community engagement, author contributions, issue resolution times, and more.
- **Visualization:** Presenting findings through insightful plots and charts.

The analyses aim to answer key research questions related to community engagement trends, prominent contributors, cross-project contributions, and the impact of pull request sizes on issue resolution times.

## Project Structure

```
├── data
│   ├── commits
│   │   └── *.json
│   ├── issues
│   │   └── *.json
│   ├── pull_request
│   │   └── *.json
│   └── dataset_filtrado.csv
├── src
│   ├── analyse_utils.py
│   ├── scraping_commits.ipynb
│   ├── scraping_issues.ipynb
│   ├── scraping_pull_request.ipynb
│   ├── data_processing.py
│   └── requirements.txt
├── notebooks
│   └── analysis_notebook.ipynb
├── plots
│   └── *.png
├── .env
├── README.md
└── LICENSE
```

- **data/**: Contains raw and processed data files.
- **src/**: Source code including utility scripts and scraping notebooks.
- **notebooks/**: Jupyter notebooks for data analysis and visualization.
- **plots/**: Generated plots and charts from the analysis.
- **.env**: Environment variables for API tokens.
- **README.md**: Project documentation.
- **LICENSE**: Licensing information.

## Setup

### 1. **Clone the repository:**

```bash
git clone https://github.com/staubjoao/TESS-work
```

### 2. **Install dependencies:**

Ensure you have Python 3.8 or higher installed. Then, install the required packages:

```bash
pip install -r src/requirements.txt
```

**Requirements:**

- pandas
- matplotlib
- seaborn
- requests
- python-dotenv

### 3. **Set up environment variables:**

Create a `.env` file in the root directory and add your GitHub tokens:

```env
TOKEN1=your_github_token_here
TOKEN2=your_github_token_here
```

## Usage

### Data Scraping

- **Scrape Commits:**  
  Run the notebook `scraping_commits.ipynb` to fetch commit data from GitHub repositories.

- **Scrape Issues:**  
  Run the notebook `scraping_issues.ipynb` to fetch issue data from GitHub repositories.

- **Scrape Pull Requests:**  
  Run the notebook `scraping_pull_request.ipynb` to fetch pull request data from GitHub repositories.

### Data Analysis

- **Perform Analysis:**  
  Run the main analysis notebook to generate insights and visualizations.

  ```bash
  jupyter notebook notebooks/analysis_notebook.ipynb
  ```

## Results

## Contact

For any inquiries or contributions, please contact:

- **João Staub**: joao.staub42@gmail.com
- **Sergio Alvarez**: sasjsergioalvarezjunior@gmai.com

```

```
