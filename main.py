# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# 1) Explore Data Analysis (EDA)
# Load the dataset
url = "https://data.cityofnewyork.us/api/views/c3uy-2p5r/rows.csv"
data = pd.read_csv(url)

# Display dataset structure
print(data.head())
print(data.info())
print(data.describe())

#-------------------------------------------------------------------------------------------------------------------------------------------

# 2) Data Cleaning and Preprocessing
# a. Handling missing values: Drop rows with missing 'Geo Join ID' or 'Data Value'
data = data.dropna(subset=['Geo Join ID', 'Data Value'])

# b. Converting 'Start_Date' to datetime format
data['Start_Date'] = pd.to_datetime(data['Start_Date'], format='%m/%d/%Y')

# c. Remove irrelevant columns
# Drop the 'Message' and 'Indicator ID' columns as these don't contribute to the model
data = data.drop(columns=['Message', 'Indicator ID'])

# X: Select all columns except 'Data Value' (target)
X = data.drop(columns=['Data Value'])  
y = data['Data Value']

print(X.head())
print(y.head())

#-------------------------------------------------------------------------------------------------------------------------------------------

# 3) Data Visualization

# a. Distribution of the target variable (Data Value)
plt.figure(figsize=(10, 6))
sns.histplot(data['Data Value'], bins=30, kde=True)
plt.title('Distribution of Data Value')
plt.xlabel('Data Value')
plt.ylabel('Frequency')
plt.show()

# b. Time series analysis of Data Value over Start_Date
plt.figure(figsize=(12, 6))
sns.lineplot(x='Start_Date', y='Data Value', data=data)
plt.title('Data Value Over Time')
plt.xlabel('Start Date')
plt.ylabel('Data Value')
plt.xticks(rotation=45)
plt.show()

# c. Bar plot to compare Data Value across different Geo Place Names
plt.figure(figsize=(12, 6))
geo_means = data.groupby('Geo Place Name')['Data Value'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=geo_means.values, y=geo_means.index)
plt.title('Top 10 Geo Place Names by Average Data Value')
plt.xlabel('Average Data Value')
plt.ylabel('Geo Place Name')
plt.show()

# d. Scatter plot for Geo Join ID vs Data Value
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Geo Join ID', y='Data Value', data=data)
plt.title('Geo Join ID vs Data Value')
plt.xlabel('Geo Join ID')
plt.ylabel('Data Value')
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------

# 4) Model Development

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_preprocessed, y_train)
    y_pred = model.predict(X_test_preprocessed)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} R-squared: {r2:.4f}')

# Hyperparameter Tuning for Gradient Boosting Regressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=models['Gradient Boosting Regressor'], param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train_preprocessed, y_train)

best_gb_model = grid_search.best_estimator_
y_pred_best_gb = best_gb_model.predict(X_test_preprocessed)
r2_best_gb = r2_score(y_test, y_pred_best_gb)
print(f'Best Gradient Boosting Regressor R-squared: {r2_best_gb:.4f}')

#-------------------------------------------------------------------------------------------------------------------------------------------

# 5) Model Evaluation

# Print all R-squared scores
for name, model in models.items():
    model.fit(X_train_preprocessed, y_train)
    y_pred = model.predict(X_test_preprocessed)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} R-squared: {r2:.4f}')

print(f'Best Gradient Boosting Regressor R-squared: {r2_best_gb:.4f}')

