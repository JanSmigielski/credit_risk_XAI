import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from pathlib import Path

CLEAN_DIR = Path("data/cleaned")
PROCESSED_DIR = Path("data/processed")

data = pd.read_csv(CLEAN_DIR / 'data_cleaned.csv')
data = data.drop(columns=['Unnamed: 0', 'KRS'])

# Separating numerical and categorical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Excluding the target variable from the lists
numerical_features = [col for col in numerical_features if col != 'class']
categorical_features = [col for col in categorical_features if col != 'class']

data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Training a Random Forest model to compute feature importance
X = data_encoded.drop('into_default', axis=1)
y = data_encoded['into_default']
rf = RandomForestClassifier(random_state=37)
rf.fit(X, y)

# Feature importance
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

important_features = list(feature_importances.head(15).index)
important_features.append('into_default') 
important_features.append('Wiek firmy (lata)') 

final_dataset = data_encoded[important_features]

# Obliczamy macierz korelacji
correlation_matrix = final_dataset.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Finding highly correlated variables 
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

# Dropping correlated features 
final_dataset = final_dataset.drop(columns=to_drop)

train, test = train_test_split(final_dataset, test_size=0.2, random_state=37, stratify=final_dataset[['into_default']])

final_dataset.to_csv(PROCESSED_DIR / 'modeling_dataset.csv')
train.to_csv(PROCESSED_DIR /'train.csv')
test.to_csv(PROCESSED_DIR /'test.csv')