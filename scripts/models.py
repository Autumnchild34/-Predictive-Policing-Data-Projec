from h2o import h2o
import h2o
from h2o.frame import H2OFrame

# Initialize H2O
h2o.init()


def train_model(data, target_col, algo='random_forest'):
    """
    Trains and returns an H2O model.
    Args:
        data: Processed data as a Pandas DataFrame.
        target_col: Target column name.
        algo: Algorithm to use ('random_forest', 'xgboost', etc.).
    """
    h2o_df = H2OFrame(data)

    # Split data
    train, test = h2o_df.split_frame(ratios=[0.8], seed=42)

    # Select predictors
    predictors = [col for col in data.columns if col != target_col]

    # Define model
    if algo == 'random_forest':
        model = h2o.estimators.H2ORandomForestEstimator()
    elif algo == 'xgboost':
        model = h2o.estimators.H2OXGBoostEstimator()
    else:
        raise ValueError("Algorithm not supported")

    # Train model
    model.train(x=predictors, y=target_col, training_frame=train)

    # Evaluate
    perf = model.model_performance(test)
    print(perf)
    return model


# Example usage
if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("outputs/processed_data.csv")
    model = train_model(data, target_col="CrimeCount")
    model.download_mojo("outputs/model_results/random_forest_model.mojo")
