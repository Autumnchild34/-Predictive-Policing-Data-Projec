from scripts.data_wrangling import load_data, preprocess_data
from scripts.models import train_model
from scripts.visualization import plot_crime_trends, interactive_map

def main():
    # Load and preprocess data
    filepath = "data/sf_crime_2016 (1).csv"
    raw_data = load_data(filepath)
    processed_data = preprocess_data(raw_data)

    # Train and evaluate model
    model = train_model(processed_data, target_col='CrimeCount', algo='random_forest')

    # Visualize results
    plot_crime_trends(processed_data)
    interactive_map(processed_data)

if __name__ == "__main__":
    main()
