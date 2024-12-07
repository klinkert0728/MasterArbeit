import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(df):
    # Use version_1 as features and version_2 as the target
    X = df[['version_1']].values.reshape(-1, 1)
    y = df['version_2'].values
    
    # Print initial info about missing values
    print("\nMissing values before cleaning:")
    print(np.isnan(X).sum(), "total NaN values in features")
    print(np.isnan(y).sum(), "total NaN values in target")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    y_imputed = imputer.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Print info about data shape
    print("\nData shape after cleaning:")
    print(f"Features: {X_scaled.shape}")
    print(f"Target: {y_imputed.shape}")
    
    return X_scaled, y_imputed, ['version_1']

def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

def analyze_results(model, X_train, X_test, y_train, y_test, y_pred, feature_names):
    # Calculate metrics
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"R² Score (Training): {r2_train:.4f}")
    print(f"R² Score (Testing): {r2_test:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    })
    importance['abs_coef'] = abs(importance['coefficient'])
    importance = importance.sort_values('abs_coef', ascending=False)
    
    # Plot top 20 most important features
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance.head(20), x='coefficient', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Save feature importance to CSV
    importance.to_csv('feature_importance.csv', index=False)
    
    return importance

def main():
    # Read the data
    df = pd.read_csv('regression_analysis/regression_data.csv')
    
    # Prepare data
    X, y, feature_names = prepare_data(df)
    
    # Train model
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    
    # Analyze results
    importance = analyze_results(model, X_train, X_test, y_train, y_test, y_pred, feature_names)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10)[['feature', 'coefficient']].to_string())

if __name__ == "__main__":
    main() 