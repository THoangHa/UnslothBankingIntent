"""
    This file contains the script to load and preprocess data
"""

import os
import re
import pandas as pd
from datasets import load_dataset    

def clean_text(text):
    """
        Clean the text by removing special characters and extra spaces
    """
    if pd.isna(text):
        return text
    
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def main():
    print("Downloading and loading the BANKING77 dataset")
    # Load the dataset from Hugging Face
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)

    # Convert the dataset splits into Pandas DataFrames for easier manipulation
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()

    print(f"Original training set size: {len(df_train)} rows")
    print(f"Original test set size: {len(df_test)} rows")

    # 1. Remove null values
    print("\nRemoving null values")
    df_train.dropna(subset=['text', 'label'], inplace=True)
    df_test.dropna(subset=['text', 'label'], inplace=True)

    # 2. Normalise the text column 
    print("Normalising the text data")
    df_train['text'] = df_train['text'].apply(clean_text)
    df_test['text'] = df_test['text'].apply(clean_text)

    # 3. Remove internal duplicates within each dataset
    print("Removing internal duplicates")
    df_train.drop_duplicates(subset=['text'], inplace=True)
    df_test.drop_duplicates(subset=['text'], inplace=True)

    # 4. Remove cross-split duplicates (Data Leakage Prevention)
    # If a query is in both the train and test sets, we remove it from the train set
    print("Removing overlapping examples between train and test sets")
    initial_train_size = len(df_train)
    df_train = df_train[~df_train['text'].isin(df_test['text'])]
    overlap_removed = initial_train_size - len(df_train)
    print(f"Removed {overlap_removed} overlapping rows from the training set.")

    # 5. Sample a subset of the cleaned data
    # Taking 10% of the dataset
    # The random_state ensures reproducibility
    sample_fraction = 0.1 
    
    df_train_sampled = df_train.sample(frac=sample_fraction, random_state=42)
    df_test_sampled = df_test.sample(frac=sample_fraction, random_state=42)

    print(f"\nFinal sampled training set size: {len(df_train_sampled)} rows")
    print(f"Final sampled test set size: {len(df_test_sampled)} rows")

    # 6. Prepare the output directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "sample_data")
    
    os.makedirs(output_dir, exist_ok=True)

    # 7. Save the processed data to CSV files
    train_output_path = os.path.join(output_dir, "train.csv")
    test_output_path = os.path.join(output_dir, "test.csv")

    # Save without the pandas index column to keep the files clean
    df_train_sampled.to_csv(train_output_path, index=False)
    df_test_sampled.to_csv(test_output_path, index=False)

    print(f"\nSuccess! Data saved to:")
    print(f"- {train_output_path}")
    print(f"- {test_output_path}")

if __name__ == "__main__":
    main()


