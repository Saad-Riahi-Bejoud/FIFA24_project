# FIFA 24 Player Performance Analysis Project

## Description

The FIFA Player Performance Analysis project aims to analyze the performance, attributes, and valuation of soccer players based on FIFA player statistics data. The project involves preprocessing the data, conducting player performance analysis, club analysis, exploring player valuation trends, and developing a machine learning model to predict player valuation.

## Project Overview

### Milestones and Steps

1. **Data Preprocessing**
   - Handle missing values, duplicates, and outliers in the dataset.
   - Encode categorical variables and scale/normalize numerical features.
   - Prepare the dataset for analysis.

2. **Player Performance Analysis**
   - Conduct descriptive statistics to summarize player attributes.
   - Visualize player performance distributions and correlations.
   - Explore the relationships between attributes and player performance.

3. **Club Analysis**
   - Group player statistics by clubs and analyze club-wise performance.
   - Visualize club-level statistics and identify top-performing clubs.

4. **Player Valuation Trends**
   - Analyze player valuation trends over time using historical data.

5. **Machine Learning Model Development**
   - Define features and target variable for predicting player valuation.
   - Split the dataset into training and testing sets.
   - Train a machine learning model (e.g., Linear Regression) to predict player valuation.
   - Evaluate model performance using metrics such as Mean Squared Error (MSE) and R-squared.

### Codes

#### Data Preprocessing

```python
# Data preprocessing steps
import pandas as pd
import numpy as np

# Load the FIFA player statistics dataset
player_stats_df = pd.read_csv('Player_Stats.csv')

# Handle missing values
player_stats_df.fillna(player_stats_df.median(), inplace=True)

# Remove duplicates
player_stats_df.drop_duplicates(inplace=True)

# Handle outliers
# Remove outliers in the 'value' column using IQR method
Q1 = player_stats_df['value'].quantile(0.25)
Q3 = player_stats_df['value'].quantile(0.75)
IQR = Q3 - Q1
player_stats_df = player_stats_df[~((player_stats_df['value'] < (Q1 - 1.5 * IQR)) | (player_stats_df['value'] > (Q3 + 1.5 * IQR)))]

# Encode categorical variables
player_stats_df = pd.get_dummies(player_stats_df, columns=['country', 'club', 'att_position'])

# Scale/normalize numerical features
numerical_columns = player_stats_df.select_dtypes(include=['int64', 'float64']).columns
player_stats_df[numerical_columns] = (player_stats_df[numerical_columns] - player_stats_df[numerical_columns].min()) / (player_stats_df[numerical_columns].max() - player_stats_df[numerical_columns].min())


Player Performance Analysis

# Compute descriptive statistics for player attributes.
player_stats_descriptive = player_stats_df.describe()
print(player_stats_descriptive)
# Analyze the distribution of a specific attribute
attribute_distribution = player_stats_df['ball_control'].value_counts()
print(attribute_distribution)
# Exploration of attribute-performance relationships
correlation_matrix = player_stats_df[['ball_control', 'dribbling']].corr()
print(correlation_matrix)
# Visualizing attributes
import seaborn as sns
import matplotlib.pyplot as plt 

sns.histplot(player_stats_df['ball_control'])
plt.title('Distribution of Ball Control Attribute')
plt.xlabel('Ball Control')
plt.ylabel('Frequency')
plt.show()
# Investigate correlations between different player attributes.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

numeric_columns = player_stats_df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = player_stats_df[numeric_columns].corr()
if correlation_matrix.isnull().values.any():
    correlation_matrix.fillna(0, inplace=True) 
if not np.isfinite(correlation_matrix).all().all():
    correlation_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    correlation_matrix.fillna(0, inplace=True) 

# Plot clustered heatmap with improved visualization
plt.figure(figsize=(14, 12)) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Player Attributes')
plt.show()

Positional Insights

# Explore player distribution across clubs
player_distribution = player_stats_df['club'].value_counts()
print(player_distribution)
# Analyze attribute distributions based on player positions
import seaborn as sns
import matplotlib.pyplot as plt
# Set the figure size
plt.figure(figsize=(12, 8))
# Create a Violin Plot
sns.violinplot(x='att_position', y='ball_control', data=player_stats_df)
# Set the title and labels
plt.title('Distribution of Ball Control by Player Position')
plt.xlabel('Player Position')
plt.ylabel('Ball Control')
# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
# Show the plot
plt.show()

Club Analysis

# Statistics for players grouped by their respective clubs
numeric_columns = player_stats_df.select_dtypes(include=['int64', 'float64']).columns
club_statistics = player_stats_df.groupby('club')[numeric_columns].mean()
print(club_statistics)
# Statistics for players grouped by their respective positions
numeric_columns = player_stats_df.select_dtypes(include=[np.number]).columns
if not numeric_columns.empty:
    player_stats_df[numeric_columns] = player_stats_df[numeric_columns].fillna(player_stats_df[numeric_columns].median())
    position_insights = player_stats_df.groupby('att_position')[numeric_columns].mean()
    print(position_insights)
else:
    print("No numeric columns found to perform aggregation.")
# Visualize overall rating for each club-level statistics
import matplotlib.pyplot as plt

club_avg_overall = player_stats_df.groupby('club')['value'].mean().sort_values(ascending=False)
# Plotting
plt.figure(figsize=(12, 8))
club_avg_overall[:20].plot(kind='bar', color='skyblue')
plt.title('Average Overall Rating by Club (Top 20)')
plt.xlabel('Club')
plt.ylabel('Average Overall Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

Player Valuation Trends

# Analyze player valuation trends over time
import pandas as pd
import matplotlib.pyplot as plt

print(player_stats_df.head())
player_mean_valuation = player_stats_df['value'].mean()

# Visualize player valuation trends over time
plt.figure(figsize=(10, 6))
plt.plot(player_stats_df.index, player_stats_df['value'], marker='o', color='b')
plt.title('Player Valuation Trends Over Time')
plt.xlabel('Player Index')
plt.ylabel('Valuation')
plt.axhline(y=player_mean_valuation, color='r', linestyle='--', label='Mean Valuation')
plt.legend()
plt.grid(True)
plt.show()

Machine Learning Model Development

# Define features and target variable
selected_features = ['age', 'ball_control', 'dribbling', 'marking', 'slide_tackle', 'stand_tackle', 'aggression', 'reactions', 'interceptions', 'vision', 'composure', 'crossing', 'short_pass', 'long_pass', 'acceleration', 'stamina', 'strength', 'balance', 'sprint_speed', 'agility', 'jumping', 'heading', 'shot_power', 'finishing', 'long_shots', 'curve', 'fk_acc', 'penalties', 'volleys']
X = player_stats_df[selected_features]
y = player_stats_df['value']
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate model performance (MSE, R-squared)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


Objectives
Analyze player performance and valuation trends.
Identify key attributes influencing player valuation.
Develop a predictive model to estimate player valuation.
Provide insights for clubs, scouts, and fantasy soccer enthusiasts. 
