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

# Path to the directory containing commit JSON files
COMMITS_DATA_PATH = "data/commits/*.json"

# Threshold percentile to define core developers (e.g., top 20%)
CORE_DEVELOPER_PERCENTILE = 0.8

# ---------------------------
# Utility Functions
# ---------------------------


def load_commit_data(file_path):
    """
    Load commit data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of commit dictionaries.
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
        data = load_commit_data(file_path)
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


# ---------------------------
# Author Contribution Analysis
# ---------------------------


def analyze_author_contributions(commits_df):
    """
    Analyze author contributions to identify core and occasional contributors.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        pd.DataFrame, list: Updated DataFrame with 'is_core_developer' flag and list of core developers.
    """
    # Calculate number of commits per author
    author_commit_counts = commits_df.groupby("author").size().rename("commit_count")

    # Define threshold for core developers
    threshold = author_commit_counts.quantile(CORE_DEVELOPER_PERCENTILE)
    core_developers = author_commit_counts[author_commit_counts >= threshold].index.tolist()

    # Flag core developers
    commits_df["is_core_developer"] = commits_df["author"].isin(core_developers)

    return commits_df, core_developers


def analyze_cross_project_contributions(commits_df):
    """
    Analyze authors contributing to multiple repositories.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        pd.DataFrame: DataFrame with authors and the number of repositories they have contributed to.
    """
    author_repo_counts = commits_df.groupby("author")["repo_name"].nunique().reset_index()
    author_repo_counts.rename(columns={"repo_name": "repositories_contributed"}, inplace=True)
    return author_repo_counts


# ---------------------------
# Repository Activity Analysis
# ---------------------------


def analyze_repository_activity(commits_df):
    """
    Analyze the activity levels of repositories.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        pd.DataFrame: DataFrame with repositories and their activity metrics.
    """
    repo_activity = commits_df.groupby("repo_name").agg(
        {"date": ["min", "max"], "author": "nunique", "message": "count"}
    )
    repo_activity.columns = ["start_date", "end_date", "unique_authors", "total_commits"]
    repo_activity["active_days"] = (repo_activity["end_date"] - repo_activity["start_date"]).dt.days + 1
    repo_activity["commits_per_day"] = repo_activity["total_commits"] / repo_activity["active_days"]
    return repo_activity.reset_index()


# ---------------------------
# Social Network Analysis
# ---------------------------


def perform_social_network_analysis(commits_df):
    """
    Perform social network analysis to identify collaboration patterns.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.

    Returns:
        networkx.Graph, dict: Collaboration graph and centrality measures.
    """
    collaboration_graph = nx.Graph()

    # Group commits by date and add edges between authors who committed on the same day within the same repository
    grouped = commits_df.groupby(["repo_name", "date_only"])
    for (repo, date), group in grouped:
        authors = group["author"].unique()
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                collaboration_graph.add_edge(authors[i], authors[j])

    # Calculate degree centrality
    centrality = nx.degree_centrality(collaboration_graph)
    top_authors = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    return collaboration_graph, centrality, top_authors


# ---------------------------
# Visualization Functions
# ---------------------------


def plot_commits_and_authors(commits_df):
    """
    Plot commits over time and unique authors over time.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.
    """
    commit_counts = commits_df.groupby("date_only").size().rename("commits")
    unique_authors = commits_df.groupby("date_only")["author"].nunique().rename("unique_authors")

    analysis_df = pd.concat([commit_counts, unique_authors], axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # Commits over time
    ax[0].plot(analysis_df.index, analysis_df["commits"], label="Commits", color="blue")
    ax[0].set_title("Commits Over Time")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Number of Commits")
    ax[0].grid(True)
    ax[0].tick_params(axis="x", rotation=45)

    # Unique authors over time
    ax[1].plot(analysis_df.index, analysis_df["unique_authors"], label="Unique Authors", color="green")
    ax[1].set_title("Unique Authors Over Time")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Number of Unique Authors")
    ax[1].grid(True)
    ax[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_core_vs_occasional_commits(commits_df, core_developers):
    """
    Plot commits by core developers vs. occasional contributors over time.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame with 'is_core_developer' flag.
        core_developers (list): List of core developer names.
    """
    core_commit_counts = commits_df[commits_df["is_core_developer"]].groupby("date_only").size()
    occasional_commit_counts = commits_df[~commits_df["is_core_developer"]].groupby("date_only").size()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(core_commit_counts.index, core_commit_counts, label="Core Developers", color="red")
    ax.plot(occasional_commit_counts.index, occasional_commit_counts, label="Occasional Contributors", color="orange")
    ax.set_title("Commits by Developer Type Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Commits")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_collaboration_network(collaboration_graph, top_authors):
    """
    Visualize the collaboration network of authors.

    Args:
        collaboration_graph (networkx.Graph): Collaboration graph.
        top_authors (list): List of top central authors.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(collaboration_graph, k=0.15, iterations=20)

    # Nodes
    nx.draw_networkx_nodes(collaboration_graph, pos, node_size=50, alpha=0.7)

    # Edges
    nx.draw_networkx_edges(collaboration_graph, pos, alpha=0.5)

    # Highlight top authors
    top_author_names = [author for author, _ in top_authors]
    nx.draw_networkx_nodes(
        collaboration_graph, pos, nodelist=top_author_names, node_size=300, node_color="yellow", label="Top Authors"
    )

    plt.title("Collaboration Network of Authors")
    plt.axis("off")
    plt.legend()
    plt.show()


def plot_commit_trends_over_age(commits_df):
    """
    Plot commit frequency over repository age.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.
    """
    age_commit_counts = commits_df.groupby("repo_age_days").size().rename("commits")

    plt.figure(figsize=(10, 6))
    plt.plot(age_commit_counts.index, age_commit_counts, color="purple")
    plt.title("Commit Frequency Over Repository Age")
    plt.xlabel("Repository Age (Days)")
    plt.ylabel("Number of Commits")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_commit_message_quality(commits_df):
    """
    Plot the proportion of high-quality commit messages over time.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.
    """
    quality_trends = commits_df.groupby("date_only")["high_quality_msg"].mean().rename("high_quality_proportion")

    plt.figure(figsize=(10, 6))
    plt.plot(quality_trends.index, quality_trends, color="teal")
    plt.title("Proportion of High-Quality Commit Messages Over Time")
    plt.xlabel("Date")
    plt.ylabel("Proportion of High-Quality Messages")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_repository_phase_metrics(commits_df):
    """
    Plot maintenance metrics by repository phase.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.
    """
    phase_metrics = (
        commits_df.groupby("phase")
        .agg(
            total_commits=pd.NamedAgg(column="repo_name", aggfunc="count"),
            average_commit_message_quality=pd.NamedAgg(column="high_quality_msg", aggfunc="mean"),
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for total commits
    ax1.bar(phase_metrics["phase"], phase_metrics["total_commits"], color="skyblue", label="Total Commits")
    ax1.set_xlabel("Repository Phase")
    ax1.set_ylabel("Total Commits", color="skyblue")
    ax1.tick_params(axis="y", labelcolor="skyblue")

    # Line plot for average commit message quality
    ax2 = ax1.twinx()
    ax2.plot(
        phase_metrics["phase"],
        phase_metrics["average_commit_message_quality"],
        color="teal",
        marker="o",
        label="Avg. Commit Msg Quality",
    )
    ax2.set_ylabel("Average Commit Message Quality", color="teal")
    ax2.tick_params(axis="y", labelcolor="teal")

    # Titles and legends
    plt.title("Maintenance Metrics by Repository Phase")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()


def plot_cross_project_contributions(author_repo_counts):
    """
    Plot the distribution of the number of repositories contributed to by authors.

    Args:
        author_repo_counts (pd.DataFrame): DataFrame with authors and their repository counts.
    """
    repo_contrib_counts = author_repo_counts["repositories_contributed"].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    repo_contrib_counts.plot(kind="bar", color="coral")
    plt.title("Distribution of Repositories Contributed to by Authors")
    plt.xlabel("Number of Repositories")
    plt.ylabel("Number of Authors")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_commit_size_over_time(commits_df):
    """
    Plot average commit size over time.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.
    """
    average_sizes = commits_df.groupby("date_only")["commit_size"].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(average_sizes.index, average_sizes.values, label="Average Commit Size", color="magenta")
    plt.title("Average Commit Size Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Number of Lines Changed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_commit_categories(commits_df):
    """
    Plot the distribution of commit categories over time.

    Args:
        commits_df (pd.DataFrame): Enriched commit DataFrame.
    """
    category_counts = commits_df.groupby(["date_only", "commit_category"]).size().unstack(fill_value=0)
    category_counts.plot(kind="area", stacked=True, figsize=(12, 6), colormap="Set3")
    plt.title("Commit Categories Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Commits")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_repository_activity(repo_activity):
    """
    Plot repository activity levels.

    Args:
        repo_activity (pd.DataFrame): DataFrame with repository activity metrics.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(repo_activity["unique_authors"], repo_activity["commits_per_day"], color="mediumseagreen")
    plt.title("Repository Activity Levels")
    plt.xlabel("Number of Unique Authors")
    plt.ylabel("Commits per Day")
    plt.grid(True)
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

    # Step 2: Enrich data with additional metrics
    print("Enriching commit data...")
    commits_df = enrich_commit_data(commits_df)

    # Step 3: Analyze author contributions
    print("Analyzing author contributions...")
    commits_df, core_developers = analyze_author_contributions(commits_df)
    print(f"Number of core developers: {len(core_developers)}")

    # Step 3.1: Analyze authors contributing to multiple repositories
    print("Analyzing authors contributing to multiple repositories...")
    author_repo_counts = analyze_cross_project_contributions(commits_df)
    multi_repo_authors = author_repo_counts[author_repo_counts["repositories_contributed"] > 1]
    print(f"Number of authors contributing to multiple repositories: {len(multi_repo_authors)}")

    # Step 3.2: Analyze repository activity
    print("Analyzing repository activity levels...")
    repo_activity = analyze_repository_activity(commits_df)
    print(repo_activity.head())

    # Step 4: Perform social network analysis
    print("Performing social network analysis...")
    collaboration_graph, centrality, top_authors = perform_social_network_analysis(commits_df)
    print(f"Number of nodes (authors): {collaboration_graph.number_of_nodes()}")
    print(f"Number of edges (collaborations): {collaboration_graph.number_of_edges()}")
    print("Top 10 Central Authors:")
    for author, cent in top_authors:
        print(f"{author}: {cent:.4f}")

    # Step 5: Generate Visualizations
    print("Generating visualizations...")

    # 5.1 Commits and Unique Authors Over Time
    plot_commits_and_authors(commits_df)

    # 5.2 Commits by Developer Type Over Time
    plot_core_vs_occasional_commits(commits_df, core_developers)

    # 5.3 Collaboration Network
    plot_collaboration_network(collaboration_graph, top_authors)

    # 5.4 Commit Frequency Over Repository Age
    plot_commit_trends_over_age(commits_df)

    # 5.5 Commit Message Quality Over Time
    plot_commit_message_quality(commits_df)

    # 5.6 Maintenance Metrics by Repository Phase
    plot_repository_phase_metrics(commits_df)

    # 5.7 Cross-Project Contributions
    plot_cross_project_contributions(author_repo_counts)

    # 5.8 Average Commit Size Over Time
    plot_commit_size_over_time(commits_df)

    # 5.9 Commit Categories Over Time
    plot_commit_categories(commits_df)

    # 5.10 Repository Activity Levels
    plot_repository_activity(repo_activity)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
