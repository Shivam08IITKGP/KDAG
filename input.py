import pandas as pd

def get_input_data(row_indices):
    """
    Read specific rows from train.csv based on provided indices (0-indexed).
    Returns: DataFrame with book_name, char, and content columns
    """
    df = pd.read_csv('train.csv')
    # Filter by specific indices
    return df.iloc[row_indices][['book_name', 'char', 'content']]

if __name__ == "__main__":
    input_str = input("Enter space-separated row indices (e.g., '0 5'): ")
    indices = [int(x) for x in input_str.split()]
    data = get_input_data(indices)
    print(f"\nLoaded {len(data)} rows from train.csv")
    print(data.head())