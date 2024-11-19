# Advanced Topics in Software Engineering

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Scraping](#data-scraping)
  - [Data Processing](#data-processing)
  - [Data Analysis](#data-analysis)
- [Results](#results)
- [Contact](#contact)

## Overview

This project was made for the course **Tópicos em Engenharia de Sistemas de Software I - DIN4096**. It is primarily based on the paper:

**Amoroso D'Aragona, Dario et al.** _A Dataset of Microservices-based Open-Source Projects_. In: **Proceedings of the 21st International Conference on Mining Software Repositories**, 2024, pp. 504-509.

In their study, the authors provided a CSV file containing information about open microservice repositories. This project aims to analyze the evolution and contributions of various GitHub repositories. The project involves:

- **Data Collection:** Extracting commits, issues, and pull requests from the selected repositories.
- **Data Processing:** Cleaning and enriching the collected data with additional metrics.
- **Data Analysis:** Conducting comprehensive analyses to understand community engagement, author contributions, issue resolution times, and more.

The analyses are designed to answer key research questions (RQs) related to community engagement trends, prominent contributors, cross-project contributions, and the impact of pull request sizes on issue resolution times.

### Research Questions (RQs)

**RQ1:** What is the current status and evolution of open microservice repositories?

- **RQ1.1:** Is community engagement increasing?
- **RQ1.2:** Are there prominent authors or development teams?
- **RQ1.3:** Do authors contribute to multiple projects?

**RQ2:** How can closed PR data be utilized to assess the maintenance time of open microservice repositories?

- **RQ2.1:** What is the average issue resolution time?
- **RQ2.2:** What is the impact of PR size on issue resolution time?
- **RQ2.3:** What is the impact of contributor size on issue resolution time?
- **RQ2.4:** What is the proportion of issues that are PRs?
- **RQ2.5:** Is there a statistical difference between issue closure times with repository microservice size?

## Project Structure

```
├── data/
│   ├── commits/
│   │   └── *.json
│   ├── issues/
│   │   └── *.json
│   ├── pull_requests/
│   │   └── *.json
│   ├── dataset.csv
│   └── dataset_filtered.csv
├── src/
│   ├── scraping/
│   │   ├── scraping_commits.ipynb
│   │   ├── scraping_issues.ipynb
│   │   ├── scraping_pull_requests.ipynb
│   │   └── scraping_utils.py
│   ├── analysis_utils.py
│   ├── token_manager.py
│   ├── filter_dataset.ipynb
│   ├── analyse-2-groups.ipynb
│   └── requirements.txt
├── .env
└── README.md
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/staubjoao/TESS-work.git
cd TESS-work
```

### 2. Install Dependencies

Ensure you have **Python 3.8** or higher installed. Then, install the required packages:

```bash
pip install -r src/requirements.txt
```

**Dependencies:**

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- requests
- python-dotenv
- jupyter
- scikit-posthocs

### 3. Set Up Environment Variables

Create a `.env` file in the root directory and add your GitHub tokens:

```env
TOKEN1=your_github_token_here
TOKEN2=your_github_token_here
```

## Usage

### Data Processing

- **Filter Dataset:**  
  Open and run `src/filter_dataset.ipynb` to clean and filter the selected repositories of Amoroso dataset for analysis.

### Data Scraping

1. **Scrape Commits:**  
   Open and run `src/scraping/scraping_commits.ipynb` to fetch commit data from GitHub repositories.

2. **Scrape Issues:**  
   Open and run `src/scraping/scraping_issues.ipynb` to fetch issue data from GitHub repositories.

3. **Scrape Pull Requests:**  
   Open and run `src/scraping/scraping_pull_requests.ipynb` to fetch pull request data from GitHub repositories.

4. **Result Scraping:** 
   The full scraped dataset is available [here](https://figshare.com/s/66687efc93d8352f9156).


### Data Analysis

- **Perform Analysis:**  
  Open and run the main analysis notebook `src/analyse-2-groups.ipynb` to generate analyses and visualizations addressing the research questions.

## Results

Results analysis can be seen in file `src/analyse-2-groups.ipynb`

## Contact

- **João Staub**: [joao.staub42@gmail.com](mailto:joao.staub42@gmail.com)
- **Sergio Alvarez**: [sasjsergioalvarezjunior@gmail.com](mailto:sasjsergioalvarezjunior@gmail.com)
