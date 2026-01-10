import pandas as pd

def get_input_data(start_idx, end_idx):
    """
    Read rows from start_idx to end_idx (inclusive, 0-indexed) from train.csv
    Returns: DataFrame with book_name, char, and content columns
    """
    # Read all rows up to end_idx + 1
    df = pd.read_csv('train.csv')
    return df.iloc[start_idx : end_idx + 1][['book_name', 'char', 'content']]

if __name__ == "__main__":
    num_queries = int(input("Enter number of queries (rows to read): "))
    data = get_input_data(num_queries)
    print(f"\nLoaded {len(data)} rows from train.csv")
    print(data.head())