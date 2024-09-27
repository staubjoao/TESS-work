import json
import os
import re
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# ---------------------------
# Configuration and Constants
# ---------------------------

# Paths to the directories containing JSON files
COMMITS_DATA_PATH = "data/commits/*.json"
ISSUES_DATA_PATH = "data/issues/*.json"

# Threshold percentile to define core developers (e.g., top 20%)
CORE_DEVELOPER_PERCENTILE = 0.8

# Approval labels to consider in RQ2.6
APPROVAL_LABELS = ["lgtm", "approved"]

# ---------------------------
# Utility Functions
# ---------------------------


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


def extract_approval_labels(labels):
    """
    Extract approval labels from a list of labels.

    Args:
        labels (list): List of label dictionaries.

    Returns:
        list: List of approval labels present.
    """
    label_names = [label["name"].lower() for label in labels]
    return [label for label in label_names if label in APPROVAL_LABELS]


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
                # Include commit stats if available
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
            else:
                is_pull_request = False
            if issue.get("closed_at"):
                closed_at = datetime.strptime(issue["closed_at"], "%Y-%m-%dT%H:%M:%SZ")
            else:
                closed_at = None
            issue_data = {
                "repo_name": repo_name,
                "issue_number": issue["number"],
                "title": issue["title"],
                "user": issue["user"]["login"] if issue.get("user") else None,
                "state": issue["state"],
                "created_at": datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
                "closed_at": closed_at,
                "labels": issue.get("labels", []),
                "is_pull_request": is_pull_request,
                "comments": issue.get("comments", 0),
                "body": issue.get("body", ""),
            }
            all_issues.append(issue_data)
    issues_df = pd.DataFrame(all_issues)
    return issues_df


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
    issues_df["resolution_time"] = (
        issues_df["closed_at"] - issues_df["created_at"]
    ).dt.total_seconds() / 3600  # in hours

    # Categorize issues
    issues_df["issue_category"] = issues_df.apply(categorize_issue, axis=1)

    # Extract approval labels
    issues_df["approval_labels"] = issues_df["labels"].apply(extract_approval_labels)

    # Flag issues with approval labels
    issues_df["has_approval_label"] = issues_df["approval_labels"].apply(lambda x: len(x) > 0)

    return issues_df


# ---------------------------
# RQ1 Analysis Functions
# ---------------------------


def analyze_commit_trends(commits_df):
    """
    Analyze commit trends over time to determine if community engagement is increasing.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        pd.DataFrame: DataFrame with commit counts over time.
    """
    commit_counts = commits_df.groupby(commits_df["date"].dt.to_period("M")).size().rename("commit_count")
    commit_counts.index = commit_counts.index.to_timestamp()
    return commit_counts


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


def impact_of_pr_size_on_resolution_time(issues_df, commits_df):
    """
    Analyze the impact of pull request size on issue resolution time.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        pd.DataFrame: DataFrame correlating PR size with resolution time.
    """
    # Filter pull requests
    prs = issues_df[issues_df["is_pull_request"] & issues_df["state"] == "closed"]

    # Assume we have a way to get the size of each PR (e.g., additions and deletions)
    # For this example, we will simulate PR sizes using commit sizes
    pr_sizes = (
        commits_df.groupby("repo_name").agg({"commit_size": "mean"}).rename(columns={"commit_size": "average_pr_size"})
    )
    pr_sizes = pr_sizes.reset_index()

    # Merge PR sizes with PRs
    prs = prs.merge(pr_sizes, on="repo_name", how="left")

    # Analyze the impact
    impact_df = prs[["average_pr_size", "resolution_time"]]

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


def analyze_approval_labels_effect(issues_df):
    """
    Analyze how different approval labels influence issue closure time.

    Args:
        issues_df (pd.DataFrame): Enriched issue DataFrame.

    Returns:
        pd.DataFrame: DataFrame comparing resolution times with approval labels.
    """
    labeled_issues = issues_df[issues_df["has_approval_label"]]
    unlabeled_issues = issues_df[~issues_df["has_approval_label"]]

    avg_resolution_labeled = labeled_issues["resolution_time"].mean()
    avg_resolution_unlabeled = unlabeled_issues["resolution_time"].mean()

    comparison = pd.DataFrame(
        {
            "Label": ["With Approval Label", "Without Approval Label"],
            "Average Resolution Time (hours)": [avg_resolution_labeled, avg_resolution_unlabeled],
        }
    )

    return comparison


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
    plt.scatter(impact_df["average_pr_size"], impact_df["resolution_time"], alpha=0.7)
    plt.xlabel("Average PR Size (Lines Changed)")
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

    # ---------------------------
    # RQ1 Analysis
    # ---------------------------

    # RQ1.1: Is community engagement increasing?
    print("Analyzing community engagement over time...")
    commit_trends = analyze_commit_trends(commits_df)
    plot_commit_trends(commit_trends)

    # RQ1.2: Are there prominent authors or development teams?
    print("Identifying prominent authors...")
    prominent_authors = identify_prominent_authors(commits_df)
    plot_prominent_authors(prominent_authors)

    # RQ1.3: Do authors contribute to multiple projects?
    print("Analyzing cross-project contributions...")
    author_projects = analyze_author_cross_project_contributions(commits_df)
    plot_author_project_contributions(author_projects)

    # ---------------------------
    # RQ2 Analysis
    # ---------------------------

    # RQ2.1: What is the average issue resolution time?
    print("Calculating average issue resolution time...")
    avg_resolution_time = calculate_average_issue_resolution_time(issues_df)
    print(f"Average Issue Resolution Time: {avg_resolution_time:.2f} hours")
    plot_issue_resolution_time(issues_df)

    # RQ2.2: What types of issues are most common?
    print("Finding most common issue types...")
    issue_type_counts = find_most_common_issue_types(issues_df)
    plot_issue_types(issue_type_counts)

    # RQ2.3: Impact of PR size on issue resolution time
    print("Analyzing impact of PR size on issue resolution time...")
    impact_df = impact_of_pr_size_on_resolution_time(issues_df, commits_df)
    plot_pr_size_vs_resolution_time(impact_df)

    # RQ2.4: Most active contributors in issue resolution
    print("Identifying most active contributors in issue resolution...")
    active_contributors = identify_active_issue_contributors(issues_df)
    plot_active_issue_contributors(active_contributors)

    # RQ2.5: Proportion of issues that are PRs
    print("Calculating proportion of issues that are PRs...")
    pr_issue_proportion = calculate_pr_issue_proportion(issues_df)
    print(f"Proportion of Issues that are PRs: {pr_issue_proportion:.2%}")

    # RQ2.6: Effect of approval labels on issue closure time
    print("Analyzing effect of approval labels on issue closure time...")
    approval_label_effect = analyze_approval_labels_effect(issues_df)
    plot_approval_label_effect(approval_label_effect)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
