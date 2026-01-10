import pandas as pd

def get_input_data(num_rows):
    """
    Read specified number of rows from train.csv
    Returns: DataFrame with book_name, char, and content columns
    """
    df = pd.read_csv('train.csv', nrows=num_rows)
    return df[['book_name', 'char', 'content']]

if __name__ == "__main__":
    num_queries = int(input("Enter number of queries (rows to read): "))
    data = get_input_data(num_queries)
    print(f"\nLoaded {len(data)} rows from train.csv")
    print(data.head())