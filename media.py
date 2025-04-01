import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Load API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY is not set.")
    sys.exit(1)

genai.configure(api_key=api_key)

# Function to load dataset and convert date column to datetime
def load_data(file_path):
    """Loads a CSV file and returns a Pandas DataFrame with parsed dates."""
    try:
        df = pd.read_csv(file_path, encoding="latin1")
        print(f"Dataset '{file_path}' loaded successfully!")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Dataset Shape: {df.shape}\n")
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y", errors="coerce")
            print("Converted 'date' column to datetime format.\n")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

# Function for basic analysis
def basic_analysis(df):
    """Performs basic analysis on the dataset."""
    print("Basic Analysis:")
    print("===============")
    print(df.head(), "\n")
    print("Summary Statistics:\n", df.describe(include="all"), "\n")
    print("Missing Values Count:\n", df.isnull().sum(), "\n")

# Function for generating visualizations
def generate_visualizations(df, output_dir="media"):
    """Generates and saves visualizations for the dataset."""
    print("Generating Visualizations...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualization 1: Distribution of Overall Ratings
    if "overall" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["overall"].dropna(), bins=range(int(df["overall"].min()), int(df["overall"].max()) + 2), kde=True)
        plt.title("Distribution of Overall Ratings")
        plt.xlabel("Overall Rating")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/overall_distribution.png")
        plt.close()
        print("Saved: overall_distribution.png")
    
    # Visualization 2: Overall Ratings over Time by Language
    if "date" in df.columns and "overall" in df.columns and "language" in df.columns:
        plt.figure(figsize=(10, 6))
        # Sort values by date to ensure proper line plot
        df_sorted = df.sort_values("date")
        sns.lineplot(data=df_sorted, x="date", y="overall", hue="language", marker="o")
        plt.title("Overall Ratings Over Time by Language")
        plt.xlabel("Date")
        plt.ylabel("Overall Rating")
        plt.legend(title="Language", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_trends.png")
        plt.close()
        print("Saved: overall_trends.png")

# Function to generate story using Gemini
def generate_story_gemini(df, output_dir="media"):
    """Uses Gemini AI to generate a story based on data analysis."""
    print("Generating Story...")

    num_media = df.shape[0]
    avg_overall = df["overall"].mean() if "overall" in df.columns else "N/A"
    languages = df["language"].value_counts().to_dict() if "language" in df.columns else {}
    total_missing = df.isnull().sum().sum()
    
    prompt = f"""
    You are an AI data analyst. Summarize the following movie dataset analysis as a compelling story:
    - The dataset contains {num_media} movie records.
    - The average overall rating is {avg_overall:.2f} (if available).
    - media are reviewed in multiple languages with the following distribution: {languages}.
    - The dataset includes details such as date, language, type, title, by, overall rating, quality, and repeatability.
    - There are a total of {total_missing} missing value(s) across the dataset.
    
    Provide insights on trends, notable observations, and any interesting patterns in movie reviews and ratings.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        story = response.text
        
        readme_path = f"{output_dir}/README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Movie Dataset Analysis\n\n")
            f.write(story)
        
        print(f"Story saved to {readme_path}")
    except Exception as e:
        print(f"Error generating story: {e}")

# Main Execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python media.py path/to/media.csv")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    df = load_data(dataset_path)
    
    basic_analysis(df)
    generate_visualizations(df, output_dir="media")
    generate_story_gemini(df, output_dir="media")
