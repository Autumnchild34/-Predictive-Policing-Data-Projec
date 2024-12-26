import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads and returns the dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Performs basic preprocessing such as filtering, feature extraction, and encoding."""
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Hour'] = df['Time'].str.split(':').str[0].astype(int)

    # Group by relevant features
    crime_counts = df.groupby(['Year', 'ZipCode', 'Hour']).size().reset_index(name='CrimeCount')
    return crime_counts

# Example usage
if __name__ == "__main__":
    data = load_data("../data/sf_crime_2016 (1).csv")
    processed_data = preprocess_data(data)
    processed_data.to_csv("outputs/processed_data.csv", index=False)
