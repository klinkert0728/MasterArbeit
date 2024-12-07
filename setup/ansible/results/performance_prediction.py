import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, classification_report
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(df):
    # Group by version and calculate mean for each benchmark
    v1_means = df[df['version'] == 1].drop(['version', 'run'], axis=1).mean()
    v2_means = df[df['version'] == 2].drop(['version', 'run'], axis=1).mean()
    
    # Calculate percentage change: (v2 - v1) / v1 * 100
    perf_change = ((v2_means - v1_means) / v1_means * 100)
    
    # Use all measurements as features
    X = df.drop(['version', 'run'], axis=1)
    
    # Convert versions 1,2 to 0,1 for the model
    le = LabelEncoder()
    y = le.fit_transform(df['version'])
    
    # Print performance change statistics
    print("\nPerformance Changes Summary:")
    print(f"Total benchmarks: {len(perf_change)}")
    print(f"Degraded benchmarks: {sum(perf_change > 0)}")
    print(f"Improved benchmarks: {sum(perf_change <= 0)}")
    
    # Save detailed performance changes
    changes_df = pd.DataFrame({
        'benchmark': perf_change.index,
        'percent_change': perf_change,
        'degraded': (perf_change > 0).astype(int)
    }).sort_values('percent_change', ascending=False)
    
    changes_df.to_csv('performance_changes.csv', index=False)
    
    return X, y, X.columns

def create_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', 
                                        max_iter=1000,
                                        random_state=42))
    ])

def calculate_pseudo_r2(model, X, y):
    # Get probability predictions
    y_pred_proba = model.predict_proba(X)
    
    # Calculate log likelihood
    ll_model = np.sum(np.log(y_pred_proba[range(len(y)), y]))
    
    # Calculate null model log likelihood
    null_proba = np.bincount(y) / len(y)
    ll_null = np.sum(np.log(null_proba[y]))
    
    # Calculate McFadden's pseudo R²
    pseudo_r2 = 1 - (ll_model / ll_null)
    
    return pseudo_r2

def evaluate_model(X, y):
    pipeline = create_pipeline()
    
    # Use LeaveOneOut cross-validation
    loo = LeaveOneOut()
    predictions = []
    true_values = []
    probabilities = []
    
    print("\nPerforming Leave-One-Out Cross-validation...")
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        
        predictions.extend(pred)
        probabilities.extend(prob)
        true_values.extend(y_test)
    
    # Calculate metrics
    accuracy = accuracy_score(true_values, predictions)
    r2 = r2_score(true_values, probabilities)  # Traditional R²
    
    # Fit final model on all data for pseudo-R²
    pipeline.fit(X, y)
    pseudo_r2 = calculate_pseudo_r2(pipeline, X, y)
    
    # Calculate correlation
    correlation, p_value = pearsonr(true_values, probabilities)
    
    print("\nModel Performance Metrics:")
    print(f"Leave-One-Out Cross-validation Accuracy: {accuracy:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"McFadden's Pseudo R²: {pseudo_r2:.4f}")
    print(f"Correlation coefficient: {correlation:.4f} (p-value: {p_value:.4f})")
    print("\nClassification Report:")
    print(classification_report(true_values, predictions))
    
    # Create prediction vs actual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, probabilities, alpha=0.5)
    plt.plot([1, 2], [1, 2], 'r--')
    plt.xlabel('Actual Version')
    plt.ylabel('Predicted Probability')
    plt.title('Actual vs Predicted Values')
    plt.savefig('prediction_quality.png')
    plt.close()
    
    return pipeline

def analyze_feature_importance(model, feature_names):
    # Get coefficients from the logistic regression
    coefficients = model.named_steps['classifier'].coef_[0]
    
    # Create importance DataFrame
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coef': abs(coefficients)
    })
    importance = importance.sort_values('abs_coef', ascending=False)
    
    # Plot top 20 most important features
    plt.figure(figsize=(12, 6))
    top_20 = importance.head(20)
    sns.barplot(data=top_20, x='coefficient', y='feature')
    plt.title('Top 20 Most Influential Benchmarks')
    plt.tight_layout()
    plt.savefig('influential_benchmarks.png')
    plt.close()
    
    return importance

def main():
    # Read data
    df = pd.read_csv('regression_analysis/regression_data.csv')
    
    # Remove columns with all NaN values
    df = df.dropna(axis=1, how='all')
    
    print("Dataset shape:", df.shape)
    print("Number of samples per version:")
    print(df['version'].value_counts())
    
    # Prepare data
    X, y, feature_names = prepare_data(df)
    
    # Train and evaluate model
    model = evaluate_model(X, y)
    
    # Analyze feature importance
    importance = analyze_feature_importance(model, feature_names)
    
    print("\nTop 10 Most Influential Benchmarks:")
    print(importance.head(10)[['feature', 'coefficient']].to_string())
    
    # Save detailed results
    importance.to_csv('influential_benchmarks.csv', index=False)

if __name__ == "__main__":
    main() 