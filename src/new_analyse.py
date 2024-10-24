import json
import os
import re
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

COMMITS_DATA_PATH = "data/commits/*.json"
ISSUES_DATA_PATH = "data/issues/*.json"
CSV_FILE_PATH = "dataset/dataset_filtrado.csv"
PULL_FILES_DATA_PATH = "data/pull_request/*.json"

CORE_DEVELOPER_PERCENTILE = 0.8


def load_json_data(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of dictionaries loaded from the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def is_high_quality(message):
    """
    Determine if a commit message is of high quality based on conventional commit guidelines.

    Args:
        message (str): Commit message.

    Returns:
        bool: True if high quality, False otherwise.
    """
    # Check if the message follows the conventional commits format
    pattern = r"^(feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert|merge)(\(\w+\))?: .+"
    return bool(re.match(pattern, message.strip()))


def categorize_phase(age_days):
    """
    Categorize the repository phase based on its age in days.

    Args:
        age_days (int): Age of the repository in days.

    Returns:
        str: Phase category.
    """
    if age_days < 30:
        return "Initial Development"
    elif 30 <= age_days < 180:
        return "Growth"
    elif 180 <= age_days < 365:
        return "Stabilization"
    else:
        return "Maturity"


def categorize_commit_message(message):
    """
    Categorize commit message into predefined categories.

    Args:
        message (str): Commit message.

    Returns:
        str: Category label.
    """
    message = message.lower()
    if "fix" in message or "bug" in message:
        return "Bug Fix"
    elif "feature" in message or "feat" in message or "add" in message or "implement" in message:
        return "Feature"
    elif "refactor" in message or "clean up" in message or "cleanup" in message:
        return "Refactor"
    elif "docs" in message or "documentation" in message:
        return "Documentation"
    elif "test" in message or "tests" in message:
        return "Test"
    else:
        return "Other"


def categorize_issue(issue):
    """
    Categorize issue based on its labels.

    Args:
        issue (dict): Issue dictionary.

    Returns:
        str: Issue category.
    """
    labels = [label["name"].lower() for label in issue.get("labels", [])]
    if "bug" in labels:
        return "Bug"
    elif "enhancement" in labels or "feature" in labels:
        return "Feature Request"
    elif "documentation" in labels or "docs" in labels:
        return "Documentation"
    else:
        return "Other"


# ---------------------------
# Data Loading and Preprocessing
# ---------------------------


def load_all_commits(data_path):
    """
    Load and aggregate commit data from all JSON files in the specified directory.

    Args:
        data_path (str): Glob pattern to match JSON files.

    Returns:
        pd.DataFrame: DataFrame containing all commit data with additional metadata.
    """
    all_commits = []
    for file_path in glob(data_path):
        repo_name = os.path.splitext(os.path.basename(file_path))[0]
        data = load_json_data(file_path)
        for commit in data:
            commit_data = {
                "repo_name": repo_name,
                "author": commit["commit"]["author"]["name"],
                "author_email": commit["commit"]["author"]["email"],
                "committer": commit["committer"]["login"] if commit.get("committer") else None,
                "date": datetime.strptime(commit["commit"]["author"]["date"], "%Y-%m-%dT%H:%M:%SZ"),
                "message": commit["commit"]["message"],
                "parent_count": len(commit["parents"]),
                "verified": commit["commit"]["verification"]["verified"],
                #
                "additions": commit["stats"]["additions"] if commit.get("stats") else 0,
                "deletions": commit["stats"]["deletions"] if commit.get("stats") else 0,
                "total_changes": commit["stats"]["total"] if commit.get("stats") else 0,
            }
            all_commits.append(commit_data)
    commits_df = pd.DataFrame(all_commits)
    return commits_df


def load_all_issues(data_path):
    """
    Load and aggregate issue data from all JSON files in the specified directory.

    Args:
        data_path (str): Glob pattern to match JSON files.

    Returns:
        pd.DataFrame: DataFrame containing all issue data.
    """
    all_issues = []
    for file_path in glob(data_path):
        repo_name = os.path.splitext(os.path.basename(file_path))[0]
        data = load_json_data(file_path)
        for issue in data:
            if "pull_request" in issue:
                is_pull_request = True
                # Extract pull number from the pull_request URL
                pull_request_url = issue["pull_request"]["url"]
                # The URL format is: https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}
                pull_number = pull_request_url.rstrip("/").split("/")[-1] if pull_request_url else None
            else:
                is_pull_request = False
                pull_number = None
            issue_data = {
                "repo_name": repo_name,
                "issue_number": issue["number"],
                "pull_number": pull_number,
                "title": issue["title"],
                "user": issue["user"]["login"] if issue.get("user") else None,
                "state": issue["state"],
                "created_at": datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
                "closed_at": datetime.strptime(issue["closed_at"], "%Y-%m-%dT%H:%M:%SZ"),
                "labels": issue.get("labels", []),
                "is_pull_request": is_pull_request,
                "comments": issue.get("comments", 0),
                "body": issue.get("body", ""),
            }
            all_issues.append(issue_data)
    issues_df = pd.DataFrame(all_issues)
    return issues_df


def load_all_pull_requests(data_path):
    """
    Load and aggregate pull request data from all JSON files in the specified directory.

    Args:
        data_path (str): Glob pattern to match JSON files.

    Returns:
        pd.DataFrame: DataFrame containing all pull request data.
    """
    all_prs = []
    for file_path in glob(data_path):
        repo_name = os.path.splitext(os.path.basename(file_path))[0]
        # Remove 'pull_files_' prefix to get the repo name
        if repo_name.startswith("pull_files_"):
            repo_name = repo_name[len("pull_files_") :]
        data = load_json_data(file_path)
        for pr in data:
            pr_data = {
                "repo_name": repo_name,
                "pull_number": pr.get("pull_number"),
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "changed_files": pr.get("changed_files", 0),
            }
            all_prs.append(pr_data)
    prs_df = pd.DataFrame(all_prs)
    return prs_df


# ---------------------------
# Data Enrichment
# ---------------------------


def enrich_commit_data(commits_df):
    """
    Enrich the commit DataFrame with additional metrics and categorizations.

    Args:
        commits_df (pd.DataFrame): DataFrame containing commit data.

    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    # Convert date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(commits_df["date"]):
        commits_df["date"] = pd.to_datetime(commits_df["date"])

    # Extract date only for grouping
    commits_df["date_only"] = commits_df["date"].dt.date

    # Calculate repository age in days
    commits_df["repo_start_date"] = commits_df.groupby("repo_name")["date"].transform("min")
    commits_df["repo_age_days"] = (commits_df["date"] - commits_df["repo_start_date"]).dt.days

    # Categorize repository phase
    commits_df["phase"] = commits_df["repo_age_days"].apply(categorize_phase)

    # Assess commit message quality
    commits_df["high_quality_msg"] = commits_df["message"].apply(is_high_quality)

    # Categorize commit messages
    commits_df["commit_category"] = commits_df["message"].apply(categorize_commit_message)

    # Calculate commit size
    commits_df["commit_size"] = commits_df["additions"] + commits_df["deletions"]

    return commits_df


def enrich_issue_data(issues_df):
    """
    Enrich the issue DataFrame with additional metrics and categorizations.

    Args:
        issues_df (pd.DataFrame): DataFrame containing issue data.

    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    # Calculate issue resolution time
    issues_df["resolution_time"] = (issues_df["closed_at"] - issues_df["created_at"]) / pd.Timedelta(days=1)
    # Categorize issues
    issues_df["issue_category"] = issues_df.apply(categorize_issue, axis=1)

    return issues_df


# ---------------------------
# RQ1 Analysis Functions
# ---------------------------


def analyze_commit_trends(commits_df):
    """
    Analyze commit trends over time to determine if community engagement is increasing.
    Includes overall trends and per-repository trends.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        tuple:
            - overall_commit_counts (pd.Series): Overall commit counts per month.
            - repo_commit_counts (pd.DataFrame): Commit counts per repository per month.
    """
    # Overall commit counts per month
    overall_commit_counts = commits_df.groupby(commits_df["date"].dt.to_period("Q")).size().rename("commit_count")
    overall_commit_counts.index = overall_commit_counts.index.to_timestamp()

    # Per-repository commit counts per month
    repo_commit_counts = (
        commits_df.groupby(["repo_name", commits_df["date"].dt.to_period("Q")]).size().unstack(level=0).fillna(0)
    )
    repo_commit_counts.index = repo_commit_counts.index.to_timestamp()

    return overall_commit_counts, repo_commit_counts


# ---------------------------
# Visualization Functions (Updated)
# ---------------------------


def plot_commit_trends_per_repo(overall_commit_counts, repo_commit_counts):
    """
    Plot the overall commit trend and per-repository commit trends over time with improved scalability.

    Args:
        overall_commit_counts (pd.Series): Overall commit counts per month.
        repo_commit_counts (pd.DataFrame): Commit counts per repository per month.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from math import ceil

    num_repos = len(repo_commit_counts.columns)

    # Dynamic figure sizing: Increase width with more repositories
    base_width = 14
    additional_width = ceil(num_repos / 10) * 2  # Adjust based on the number of repos
    fig_width = base_width + additional_width
    fig_height = 8
    plt.figure(figsize=(fig_width, fig_height))

    # Use a colormap that can handle more colors
    cmap = plt.cm.get_cmap("tab20", num_repos)

    # Plot per-repository commit trends first with transparency
    for idx, repo in enumerate(repo_commit_counts.columns):
        plt.plot(
            repo_commit_counts.index, repo_commit_counts[repo], label=repo, color=cmap(idx), linewidth=1.0, alpha=0.6
        )

    # Plot overall commit trend on top with higher visibility
    plt.plot(
        overall_commit_counts.index, overall_commit_counts.values, label="Overall Commits", color="black", linewidth=2.5
    )

    plt.title("Commit Trends Over Time (Overall and Per Repository)", fontsize=16)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Number of Commits", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Optimize legend placement and styling
    if num_repos > 15:
        # For many repositories, place the legend outside and reduce font size
        plt.legend(title="Repositories", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    else:
        plt.legend(title="Repositories", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.show()


def identify_prominent_authors(commits_df):
    """
    Identify prominent authors based on the number of commits.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        pd.DataFrame: DataFrame with authors and their commit counts.
    """
    author_commit_counts = commits_df.groupby("author").size().rename("commit_count").reset_index()
    prominent_authors = author_commit_counts.sort_values(by="commit_count", ascending=False)
    return prominent_authors


def analyze_author_cross_project_contributions(commits_df):
    """
    Determine if authors contribute to multiple projects.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        pd.DataFrame: DataFrame with authors and the number of projects they contribute to.
    """
    author_projects = commits_df.groupby("author")["repo_name"].nunique().rename("project_count").reset_index()
    return author_projects


# ---------------------------
# RQ2 Analysis Functions
# ---------------------------


def calculate_average_issue_resolution_time(issues_df):
    """
    Calculate the average issue resolution time in the repository.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.

    Returns:
        float: Average resolution time in hours.
    """
    resolved_issues = issues_df[issues_df["state"] == "closed"]
    average_resolution_time = resolved_issues["resolution_time"].mean()
    return average_resolution_time


def find_most_common_issue_types(issues_df):
    """
    Find the most common types of issues in the repository.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.

    Returns:
        pd.Series: Issue category counts.
    """
    issue_type_counts = issues_df["issue_category"].value_counts()
    return issue_type_counts


def impact_of_pr_size_on_resolution_time(issues_df, prs_df):
    """
    Analyze the impact of pull request size on issue resolution time.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.
        prs_df (pd.DataFrame): DataFrame containing pull request sizes.

    Returns:
        pd.DataFrame: DataFrame correlating PR size with resolution time.
    """
    # Filter closed pull requests in issues_df
    # prs_issues = issues_df[issues_df["is_pull_request"] & (issues_df["state"] == "closed")]

    print(issues_df.head())

    print(prs_df.head())

    # tranform the pull_number to string
    prs_df["pull_number"] = prs_df["pull_number"].astype(str)
    issues_df["pull_number"] = issues_df["pull_number"].astype(str)

    prs_issues = issues_df.merge(prs_df, on=["pull_number", "pull_number"], how="inner")

    # Calculate total changes
    prs_issues["total_changes"] = prs_issues["additions"] + prs_issues["deletions"]

    # Analyze the impact
    impact_df = prs_issues[["total_changes", "resolution_time"]]
    print(impact_df.head())
    return impact_df


def identify_active_issue_contributors(issues_df):
    """
    Identify the most active contributors in issue resolution.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.

    Returns:
        pd.DataFrame: DataFrame with users and the number of issues they closed.
    """
    closed_issues = issues_df[issues_df["state"] == "closed"]
    user_issue_counts = closed_issues.groupby("user").size().rename("issues_closed").reset_index()
    active_contributors = user_issue_counts.sort_values(by="issues_closed", ascending=False)
    return active_contributors


def calculate_pr_issue_proportion(issues_df):
    """
    Calculate the proportion of issues that are pull requests.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.

    Returns:
        float: Proportion of issues that are PRs.
    """
    total_issues = len(issues_df)
    total_prs = issues_df["is_pull_request"].sum()
    proportion = total_prs / total_issues if total_issues > 0 else 0
    return proportion


def analyze_issue_resolution_time_by_microservice_size(issues_df, csv_df):
    """
    Analyze whether there is a statistical difference in issue closure time across different
    microservice sizes of repositories.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.
        csv_df (pd.DataFrame): Enriched CSV DataFrame containing microservice size information.

    Returns:
        None
    """

    # Process repo names to standardize them
    # For issues_df
    def extract_owner_repo_from_issues(repo_name):
        if repo_name.startswith("closed_issues_"):
            repo_name = repo_name[len("closed_issues_") :]
        elif repo_name.startswith("issues_"):
            repo_name = repo_name[len("issues_") :]
        elif repo_name.startswith("commits_"):
            repo_name = repo_name[len("commits_") :]
        owner_repo = repo_name.replace("_", "/")
        return owner_repo

    issues_df["repo_name_standardized"] = issues_df["repo_name"].apply(extract_owner_repo_from_issues)

    # For csv_df
    def extract_owner_repo_from_url(url):
        try:
            parts = url.rstrip("/").split("/")
            owner = parts[-2]
            repo = parts[-1]
            owner_repo = f"{owner}/{repo}"
            return owner_repo
        except IndexError:
            return None

    csv_df["repo_name_standardized"] = csv_df["URL"].apply(extract_owner_repo_from_url)

    # Now merge on 'repo_name_standardized'
    merged_df = pd.merge(
        issues_df, csv_df[["repo_name_standardized", "microservice_category"]], on="repo_name_standardized", how="left"
    )

    # Remove issues with missing 'microservice_category' or 'resolution_time'
    merged_df = merged_df.dropna(subset=["microservice_category", "resolution_time"])

    # Group resolution times by microservice category
    groups = merged_df.groupby("microservice_category")["resolution_time"].apply(list)

    # Perform Kruskal-Wallis H-test
    from scipy.stats import kruskal

    # Prepare data for the test
    data = [groups[cat] for cat in groups.index]

    # Check if we have at least two groups
    if len(data) < 2:
        print("Not enough groups for statistical testing.")
        return

    stat, p = kruskal(*data)

    print("Kruskal-Wallis H-test:")
    print(f"Statistic: {stat}, p-value: {p}")

    if p < 0.05:
        print(
            "There is a statistically significant difference in issue closure times between microservice size categories."
        )
    else:
        print(
            "There is no statistically significant difference in issue closure times between microservice size categories."
        )

    # Plot the results
    plot_issue_resolution_time_by_microservice_size(merged_df)


# ---------------------------
# Visualization Functions (Updated)
# ---------------------------


def plot_commit_trends(commit_counts):
    """
    Plot the trend of commits over time.

    Args:
        commit_counts (pd.DataFrame): DataFrame with commit counts over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(commit_counts.index, commit_counts.values, marker="o")
    plt.title("Community Engagement Over Time (Commits per Month)")
    plt.xlabel("Month")
    plt.ylabel("Number of Commits")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_prominent_authors(prominent_authors):
    """
    Plot the top authors based on the number of commits.

    Args:
        prominent_authors (pd.DataFrame): DataFrame with authors and their commit counts.
    """
    top_authors = prominent_authors.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_authors["author"], top_authors["commit_count"], color="skyblue")
    plt.xlabel("Number of Commits")
    plt.ylabel("Author")
    plt.title("Top 10 Prominent Authors")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_author_project_contributions(author_projects):
    """
    Plot the distribution of the number of projects authors contribute to.

    Args:
        author_projects (pd.DataFrame): DataFrame with authors and project counts.
    """
    project_counts = author_projects["project_count"].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    project_counts.plot(kind="bar")
    plt.xlabel("Number of Projects")
    plt.ylabel("Number of Authors")
    plt.title("Authors' Cross-Project Contributions")
    plt.tight_layout()
    plt.show()


def plot_issue_resolution_time(issues_df):
    """
    Plot the distribution of issue resolution times.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.
    """
    resolved_issues = issues_df[issues_df["state"] == "closed"]
    plt.figure(figsize=(10, 6))
    plt.hist(resolved_issues["resolution_time"], bins=50, color="coral")
    plt.xlabel("Resolution Time (hours)")
    plt.ylabel("Number of Issues")
    plt.title("Distribution of Issue Resolution Times")
    plt.tight_layout()
    plt.show()


def plot_issue_types(issue_type_counts):
    """
    Plot the most common types of issues.

    Args:
        issue_type_counts (pd.Series): Issue category counts.
    """
    issue_type_counts.plot(kind="bar", color="mediumseagreen")
    plt.xlabel("Issue Type")
    plt.ylabel("Number of Issues")
    plt.title("Most Common Issue Types")
    plt.tight_layout()
    plt.show()


def plot_pr_size_vs_resolution_time(impact_df):
    """
    Plot the impact of PR size on issue resolution time.

    Args:
        impact_df (pd.DataFrame): DataFrame correlating PR size with resolution time.
    """
    plt.figure(figsize=(10, 6))
    print(impact_df)
    # save to xlsx
    impact_df.to_excel("impact_df.xlsx")

    plt.scatter(x=impact_df["total_changes"], y=impact_df["resolution_time"], alpha=0.7)

    plt.xlabel("PR Size (Lines Changed)")
    plt.ylabel("Resolution Time (hours)")
    plt.title("Impact of PR Size on Issue Resolution Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_active_issue_contributors(active_contributors):
    """
    Plot the most active contributors in issue resolution.

    Args:
        active_contributors (pd.DataFrame): DataFrame with users and issues closed.
    """
    top_contributors = active_contributors.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_contributors["user"], top_contributors["issues_closed"], color="orchid")
    plt.xlabel("Number of Issues Closed")
    plt.ylabel("Contributor")
    plt.title("Top 10 Active Issue Resolvers")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_approval_label_effect(comparison):
    """
    Plot the effect of approval labels on issue closure time.

    Args:
        comparison (pd.DataFrame): DataFrame comparing resolution times with approval labels.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(comparison["Label"], comparison["Average Resolution Time (hours)"], color=["gold", "silver"])
    plt.xlabel("Approval Label Presence")
    plt.ylabel("Average Resolution Time (hours)")
    plt.title("Effect of Approval Labels on Issue Resolution Time")
    plt.tight_layout()
    plt.show()


def plot_issue_resolution_time_by_microservice_size(merged_df):
    """
    Plot the distribution of issue resolution times per microservice size category.

    Args:
        merged_df (pd.DataFrame): DataFrame containing issues and microservice size categories.
    """
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="microservice_category", y="resolution_time", data=merged_df)
    plt.xlabel("Microservice Size Category", fontsize=12)
    plt.ylabel("Resolution Time (hours)", fontsize=12)
    plt.title("Issue Resolution Time by Microservice Size Category", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ----------------------------------------


def load_csv_data(file_path):
    """
    Load and process data from the CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    import ast

    csv_df = pd.read_csv(file_path, delimiter=";")

    # Process the languages column
    if "languages" in csv_df.columns:

        def parse_languages(x):
            try:
                # Safely evaluate the string to a list
                return ast.literal_eval(x)
            except:
                # Handle cases where parsing fails
                return []

        csv_df["languages_list"] = csv_df["languages"].apply(parse_languages)

    # Ensure 'number_of_microservices' is numeric
    if "num_services" in csv_df.columns:
        csv_df["number_of_microservices"] = pd.to_numeric(csv_df["num_services"], errors="coerce")

    return csv_df


def enrich_csv_data(csv_df):
    """
    Enrich the CSV DataFrame with additional metrics and categorizations.

    Args:
        csv_df (pd.DataFrame): DataFrame containing CSV data.

    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    # Example: Count the number of languages used in each project
    if "languages_list" in csv_df.columns:
        csv_df["language_count"] = csv_df["languages_list"].apply(len)
    else:
        csv_df["language_count"] = 0

    # Categorize projects based on the number of microservices
    if "number_of_microservices" in csv_df.columns:

        def categorize_microservices(num):
            if pd.isnull(num):
                return "Unknown"
            elif num == 1:
                return "Monolith"
            elif 2 <= num <= 4:
                return "Small Microservice Architecture"
            elif 4 <= num <= 9:
                return "Medium Microservice Architecture"
            else:
                return "Large Microservice Architecture"

        csv_df["microservice_category"] = csv_df["number_of_microservices"].apply(categorize_microservices)
    else:
        csv_df["microservice_category"] = "Unknown"

    return csv_df


# ---------------------------
# Analysis Functions for CSV Data
# ---------------------------


def analyze_language_usage(csv_df):
    """
    Analyze the usage frequency of programming languages.

    Args:
        csv_df (pd.DataFrame): Enriched CSV DataFrame.

    Returns:
        pd.Series: Language usage counts.
    """
    # Flatten the list of languages into a single list
    all_languages = csv_df["languages_list"].explode()
    print(all_languages)
    language_counts = all_languages.value_counts()
    return language_counts


def analyze_microservice_distribution(csv_df):
    """
    Analyze the distribution of the number of microservices across projects.

    Args:
        csv_df (pd.DataFrame): Enriched CSV DataFrame.

    Returns:
        pd.Series: Counts of projects per microservice category.
    """
    microservice_counts = csv_df["microservice_category"].value_counts()
    return microservice_counts


# ---------------------------
# Visualization Functions for CSV Data
# ---------------------------


def plot_language_usage(language_counts):
    """
    Plot the usage frequency of programming languages.

    Args:
        language_counts (pd.Series): Language usage counts.
    """
    top_languages = language_counts.head(15)
    plt.figure(figsize=(12, 6))
    top_languages.plot(kind="bar", color="teal")
    plt.xlabel("Programming Language")
    plt.ylabel("Number of Projects")
    plt.title("Top Programming Languages Used in Projects")
    plt.tight_layout()
    plt.show()


def plot_microservice_distribution(microservice_counts):
    """
    Plot the distribution of microservice architectures across projects.

    Args:
        microservice_counts (pd.Series): Counts of projects per microservice category.
    """
    plt.figure(figsize=(10, 6))
    microservice_counts.plot(kind="bar", color="skyblue")
    plt.xlabel("Microservice Size Category")
    plt.ylabel("Number of Projects")
    plt.title("Distribution of Microservice Architectures")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main Analysis Function
# ---------------------------


def main():
    # Step 1: Load and preprocess commit data
    print("Loading commit data...")
    commits_df = load_all_commits(COMMITS_DATA_PATH)
    print(f"Total commits loaded: {len(commits_df)}")

    # Step 2: Enrich commit data with additional metrics
    print("Enriching commit data...")
    commits_df = enrich_commit_data(commits_df)

    # Step 3: Load and preprocess issue data
    print("Loading issue data...")
    issues_df = load_all_issues(ISSUES_DATA_PATH)
    print(f"Total issues loaded: {len(issues_df)}")

    # Step 4: Enrich issue data with additional metrics
    print("Enriching issue data...")
    issues_df = enrich_issue_data(issues_df)
    # Step 5: Load and process CSV data
    print("Loading CSV data...")
    csv_df = load_csv_data(CSV_FILE_PATH)
    print(f"Total projects loaded from CSV: {len(csv_df)}")

    print("Loading pull request data...")
    prs_df = load_all_pull_requests(PULL_FILES_DATA_PATH)
    print(f"Total pull requests loaded: {len(prs_df)}")

    # Step 6: Enrich CSV data
    print("Enriching CSV data...")
    csv_df = enrich_csv_data(csv_df)

    # ---------------------------
    # CSV Data Analysis
    # ---------------------------

    # # Analyze language usage
    # print("Analyzing programming language usage...")
    # language_counts = analyze_language_usage(csv_df)
    # plot_language_usage(language_counts)

    # # Analyze microservice distribution
    # print("Analyzing microservice distribution...")
    # microservice_counts = analyze_microservice_distribution(csv_df)
    # plot_microservice_distribution(microservice_counts)

    # analyze_language_usage(csv_df)
    # # ---------------------------
    # # RQ1 Analysis
    # # ---------------------------

    # # RQ1.1: Is community engagement increasing?
    # print("Analyzing community engagement over time...")
    # overall_commit_trends, repo_commit_trends = analyze_commit_trends(commits_df)
    # plot_commit_trends_per_repo(overall_commit_trends, repo_commit_trends)

    # # RQ1.2: Are there prominent authors or development teams?
    # print("Identifying prominent authors...")
    # prominent_authors = identify_prominent_authors(commits_df)
    # plot_prominent_authors(prominent_authors)

    # # RQ1.3: Do authors contribute to multiple projects?
    # print("Analyzing cross-project contributions...")
    # author_projects = analyze_author_cross_project_contributions(commits_df)
    # plot_author_project_contributions(author_projects)

    # # ---------------------------
    # # RQ2 Analysis
    # # ---------------------------

    # # RQ2.1: What is the average issue resolution time?
    # print("Calculating average issue resolution time...")
    # avg_resolution_time = calculate_average_issue_resolution_time(issues_df)
    # print(f"Average Issue Resolution Time: {avg_resolution_time:.2f} hours")
    # plot_issue_resolution_time(issues_df)

    # # RQ2.2: What types of issues are most common?
    # print("Finding most common issue types...")
    # issue_type_counts = find_most_common_issue_types(issues_df)
    # plot_issue_types(issue_type_counts)

    # RQ2.3: Impact of PR size on issue resolution time
    print("Analyzing impact of PR size on issue resolution time...")
    impact_df = impact_of_pr_size_on_resolution_time(issues_df, prs_df)
    plot_pr_size_vs_resolution_time(impact_df)

    # # RQ2.4: Most active contributors in issue resolution
    # print("Identifying most active contributors in issue resolution...")
    # active_contributors = identify_active_issue_contributors(issues_df)
    # plot_active_issue_contributors(active_contributors)

    # # RQ2.5: Proportion of issues that are PRs
    # print("Calculating proportion of issues that are PRs...")
    # pr_issue_proportion = calculate_pr_issue_proportion(issues_df)
    # print(f"Proportion of Issues that are PRs: {pr_issue_proportion:.2%}")

    # # RQ2.6: Is there a statistical difference between issue closure times with repository microservice size?
    # print("Analyzing issue resolution time by microservice size...")
    # analyze_issue_resolution_time_by_microservice_size(issues_df, csv_df)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
