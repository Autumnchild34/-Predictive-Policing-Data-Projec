import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

def plot_crime_trends(data):
    """Plots crime trends over time."""
    sns.lineplot(data=data, x='Hour', y='CrimeCount', hue='ZipCode')
    plt.title('Crime Trends by Hour')
    plt.show()

def interactive_map(data):
    """Creates an interactive map."""
    # Inspect the columns of the data to ensure correct names
    print("Columns in the dataset:", data.columns)

    # Check for missing values in relevant columns
    print("Missing values in columns:", data[['Latitude', 'Longitude', 'CrimeCount']].isnull().sum())

    # Remove rows with missing or invalid coordinates
    data_clean = data.dropna(subset=['Latitude', 'Longitude', 'CrimeCount'])

    # Ensure all necessary columns are numeric
    data_clean['CrimeCount'] = pd.to_numeric(data_clean['CrimeCount'], errors='coerce')
    data_clean = data_clean.dropna(subset=['CrimeCount'])  # Remove rows where 'CrimeCount' is NaN

    # Create the interactive map
    fig = px.scatter_geo(data_clean,
                         lat='Latitude',
                         lon='Longitude',
                         color='CrimeCount',
                         scope='usa',
                         title='Interactive Crime Map')
    fig.show()

# Example usage
if __name__ == "__main__":
    data = pd.read_csv("outputs/processed_data.csv")
    plot_crime_trends(data)
    interactive_map(data)
