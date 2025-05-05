import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chisquare

# Plotting histogram of a specified column for a given category and saving the figure 
def plotDistribution(save_path:Path, category_name:str, column:str, categoryDataFrame:pd.DataFrame):
    plt.figure(figsize=(20, 20))
    sb.histplot(data=categoryDataFrame, x=column)
    plt.title(f"{column}_Distribution_{category_name}")
    plt.savefig(save_path / f"{column}_Distribution_{category_name}")
    plt.close()


# Saving summary statistics (descriptive and skew) of 'Value1' and 'Value2' for a given category as JSON
def save_category_summary(category_path:Path, category_name:str, categoryDataFrame:pd.DataFrame):
    summary = {
        "x_description": categoryDataFrame["Value1"].describe().to_dict(),
        "x_skew": categoryDataFrame["Value1"].skew(),
        "y_description": categoryDataFrame["Value2"].describe().to_dict(),
        "y_skew": categoryDataFrame["Value2"].skew()
    }
    with open(category_path / f"summary_{category_name}.json",'w') as f:
        json.dump(summary, f, indent=4)


# Analysing a single category by generating plots and saving summary statistics
def analyse_category(context_path:Path, category_name:str, categoryDataFrame:pd.DataFrame):
    category_path = context_path / f"category_{category_name}"
    category_path.mkdir(exist_ok=True)
    plotDistribution(category_path, category_name, "Value1", categoryDataFrame)
    plotDistribution(category_path, category_name, "Value2", categoryDataFrame)
    save_category_summary(category_path, category_name, categoryDataFrame)


# Analysing all categories by grouping the dataframe and analysing each group
def analyse_categories(top_folder:str, df:pd.DataFrame):
    context_path = Path(top_folder)
    context_path.mkdir(exist_ok=True)
    for name, group in df.groupby("Category1"):
        analyse_category(context_path, name, group)
    save_category_summary(context_path, "All", df)


# Generating a new version of a single category using normal distributions
def generate_similar_category(category:str, category_dataframe:pd.DataFrame):
    x_description = category_dataframe["Value1"].describe()
    y_description = category_dataframe["Value2"].describe()
    new_category_dataframe = pd.DataFrame(
        {
            "Category1": np.repeat(category, int(x_description["count"])),
            "Value1": np.random.normal(
                x_description["mean"],
                x_description["std"],
                int(x_description["count"])),
            "Value2": np.random.normal(
                y_description["mean"],
                y_description["std"],
                int(y_description["count"]))
        }
    )
    return new_category_dataframe


# Generating a new version of the full dataset by applying generation to each category
def generate_similar_dataframe(df:pd.DataFrame):
    category_groups = df.groupby("Category1")
    new_data = pd.concat(
        [generate_similar_category(name, group) for name, group in category_groups],
        ignore_index=True
    )
    return new_data



# Plotting and saving bar plots comparing original vs new category distributions
def plot_category_distribution(original_df, new_df, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    original_df["Category1"].value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax[0], title="Original Category Distribution")
    new_df["Category1"].value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax[1], title="New Category Distribution")
    ax[0].set_ylabel("Proportion")
    ax[1].set_ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(save_path / "category_distribution_comparison.png")
    plt.close()


# Plotting and saving KDE charts of continuous variables between original and new data
def plot_continuous_distributions(original_df, new_df, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    for col in ["Value1", "Value2"]:
        plt.figure(figsize=(8, 4))
        sb.kdeplot(original_df[col], label="Original", fill=True)
        sb.kdeplot(new_df[col], label="New", fill=True)
        plt.title(f"Distribution of {col}")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path / f"{col}_distribution_comparison.png")
        plt.close()


# KS test to compare distributions of continuous variables
def compare_continuous_stats(original_df, new_df):
    print("\n=== Kolmogorov–Smirnov Test (Continuous Variables) ===")
    for col in ["Value1", "Value2"]:
        stat, p = ks_2samp(original_df[col], new_df[col])
        print(f"{col}: KS statistic = {stat:.4f}, p-value = {p:.4f} -- {'Similar since p>0.05' if p > 0.05 else 'Different since p<0.05'}")


# Chi-square test to compare categorical distributions
def compare_categorical_stats(original_df, new_df):
    print("\n=== Chi-Square Test (Categorical Variable) ===")
    orig_counts = original_df["Category1"].value_counts().sort_index()
    new_counts = new_df["Category1"].value_counts().reindex(orig_counts.index, fill_value=0) 
    stat, p = chisquare(f_obs=new_counts, f_exp=orig_counts)
    print(f"Chi-square statistic = {stat:.4f}, p-value = {p:.4f} -- {'Similar since p>0.05' if p > 0.05 else 'Different since p<0.05'}")


# Running verification functions for visualisations and statistical tests
def verify_similarity(original_df, new_df, output_dir: str = "verification_of_samples"):
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plot_category_distribution(original_df, new_df, save_path)
    compare_categorical_stats(original_df, new_df)

    plot_continuous_distributions(original_df, new_df, save_path)
    compare_continuous_stats(original_df, new_df)

# Entry point from task description to generate original dataset
if __name__ == "__main__":
    # Generate synthetic dataset
    np.random.seed(42)
    num_samples = 500000
    df = pd.DataFrame(
        {
            "Category1": np.random.choice(["A", "B", "C", "D", "E"],
            num_samples, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
            "Value1": np.random.normal(10, 2,
            num_samples), # Continuous variable
            "Value2": np.random.normal(20, 6,
            num_samples), # Continuous variable
        }
    )

    # Analysing and saving original data summaries
    analyse_categories("original_data", df)
    
    # Generating and analysing new dataset
    new_df = generate_similar_dataframe(df)
    analyse_categories("new_data", new_df)

    # Comparing original and new datasets
    verify_similarity(df, new_df, output_dir="verification_of_samples")
