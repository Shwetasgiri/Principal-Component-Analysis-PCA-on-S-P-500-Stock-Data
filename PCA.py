#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('stockdata.csv')


# In[2]:


df.head()


# In[3]:


# Task 2a: Identify and sort unique stock names
# We select the 'Name' column, get the unique values, and convert it to a list.
all_names = sorted(df['Name'].unique())

# Task 2b: Count how many unique names there are
num_names = len(all_names)
print(f"Task 2b: There are {num_names} unique stock names.")

# Task 2c: List the first and last 5 names
print("\nTask 2c: First 5 names:")
print(all_names[:5])

print("\nTask 2c: Last 5 names:")
print(all_names[-5:])


# In[6]:


# --- Task 3 

# First, it's crucial to convert the 'date' column to a proper datetime format.
df['date'] = pd.to_datetime(df['date'])

# Group the data by each stock's name and find its first ('min') and last ('max') trading date.
stock_date_ranges = df.groupby('Name')['date'].agg(['min', 'max'])

# Define the start and end dates of our analysis window.
start_date = pd.to_datetime('2019-11-01')
end_date = pd.to_datetime('2022-10-31')

# Identify the names of stocks that should be removed.
names_to_remove = stock_date_ranges[
    (stock_date_ranges['min'] > start_date) | (stock_date_ranges['max'] < end_date)
].index

# (3b) Print the names that were removed.
print("Task 3b: The following names were removed:")
print(names_to_remove.tolist())

# (3a) Create a new DataFrame keeping only the stocks that are NOT in the removal list.
df_filtered = df[~df['Name'].isin(names_to_remove)]

# Get a sorted list of the final stock names we will use.
remaining_names = sorted(df_filtered['Name'].unique())

# (3c) Print the count of remaining names.
print(f"\nTask 3c: There are {len(remaining_names)} names left after filtering.")


# In[7]:


# Task 4: Find the set of dates common to all remaining names.

# Get the set of dates for the very first stock in our list.
# We'll use this as our starting point.
common_dates = set(df_filtered[df_filtered['Name'] == remaining_names[0]]['date'])

# Loop through the rest of the stocks and find the dates they all share.
for name in remaining_names[1:]:
    # Get the dates for the current stock.
    stock_dates = set(df_filtered[df_filtered['Name'] == name]['date'])
    # Update our common_dates set by keeping only the dates that are also in this stock's dates.
    common_dates.intersection_update(stock_dates)

# (4b) Now, filter these common dates to be within our 3-year window.
# We already have start_date and end_date from the previous task.
final_dates = [
    date for date in sorted(list(common_dates)) 
    if start_date <= date <= end_date
]

# (4c) Count how many dates are left.
print(f"Task 4c: There are {len(final_dates)} common trading dates left.")

# (4d) List the first and last 5 dates.
# We format them as strings for cleaner printing.
print("\nTask 4d: The first 5 dates are:")
print([d.strftime('%Y-%m-%d') for d in final_dates[:5]])

print("\nTask 4d: The last 5 dates are:")
print([d.strftime('%Y-%m-%d') for d in final_dates[-5:]])


# In[8]:


# Task 5: Build a new pandas dataframe with a column for each name and a row for each date.

# First, we'll create a final dataframe that only contains the rows for our selected stocks and dates.
# This isn't strictly necessary but is good practice.
final_df = df_filtered[df_filtered['date'].isin(final_dates)]

# Now, pivot this final dataframe to get the desired structure.
# index='date': makes the 'date' column the new row labels.
# columns='Name': makes the unique values in the 'Name' column the new column headers.
# values='close': fills the table with the corresponding 'close' prices.
close_prices_df = final_df.pivot(index='date', columns='Name', values='close')

# (5b) Call the python print function for your dataframe and show the result.
# The full dataframe is very large (756 rows x 481 columns), so we'll show a snapshot.
print("Task 5b: Pivoted DataFrame of Close Prices (showing top 5 rows and columns):")
print(close_prices_df.iloc[:5, :5])


# In[9]:


# Task 6: Create a dataframe containing daily returns.

# pandas has a built-in function, pct_change(), that calculates the percentage change
# between the current and a prior element. This is exactly the return formula.
returns_df = close_prices_df.pct_change()

# The first row of this new dataframe will be all 'NaN' (Not a Number)
# because there's no previous day to calculate a return from. We must drop it.
returns_df = returns_df.dropna()

# (6b) Call the Python print function for your dataframe and show the result.
# Again, we'll show a snapshot of the top-left corner.
print("Task 6b: DataFrame of Daily Returns (showing top 5 rows and columns):")
print(returns_df.iloc[:5, :5])


# In[10]:


# Task 7: Calculate the principal components of the returns.

# Import the PCA class from the scikit-learn library.
from sklearn.decomposition import PCA

# Create an instance of the PCA model.
# By not specifying the number of components, it will find all of them (481 in our case).
pca = PCA()

# Fit the model to our returns data. This is where the PCA calculation happens.
# The model learns the principal components from the data.
pca.fit(returns_df)

# (7b) Print the top five PCs.
# The PCs are stored in the '.components_' attribute of the fitted pca object.
# They are already sorted by importance (eigenvalue), so the first row is PC1, the second is PC2, etc.
# We'll create a new DataFrame to display them nicely with labels.
top_five_pcs = pd.DataFrame(
    pca.components_[:5], 
    columns=returns_df.columns, 
    index=[f'PC{i+1}' for i in range(5)]
)

print("Task 7b: The top five Principal Components (showing weights for the first 5 stocks):")
print(top_five_pcs.iloc[:, :5])


# In[11]:


# Task 8: Analyse the explained variance of the principal components.
import matplotlib.pyplot as plt
import numpy as np

# (8a) Extract the explained variance ratios from the fitted PCA object.
# This is an array where each element is the percentage of variance explained by that component.
explained_variance_ratios = pca.explained_variance_ratio_

# (8b) What percentage of variance is explained by the first principal component?
pc1_variance = explained_variance_ratios[0] * 100
print(f"Task 8b: The first principal component explains {pc1_variance:.2f}% of the variance.")

# (8c) Plot the first 20 explained variance ratios.
plt.figure(figsize=(12, 6))
# We plot component number (1 to 20) on the x-axis
component_numbers = np.arange(1, 21)
plt.plot(component_numbers, explained_variance_ratios[:20], marker='o', linestyle='--')
plt.title('Explained Variance by Principal Component (First 20)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(component_numbers) # Ensure we have integer ticks for component numbers
plt.grid(True)

# (8d) Identify an elbow and mark it on the plot.
# The "elbow" is where the drop in variance slows significantly. Visually, it's around component 3 or 4.
elbow_point = 3 
plt.axvline(x=elbow_point, color='r', linestyle=':', label=f'Elbow at PC{elbow_point}')
plt.legend()

plt.show()


# In[12]:


# --- Task 9: Cumulative Variance Analysis ---
print("--- Task 9: Cumulative Variance Analysis ---")

# (9a) Calculate the cumulative variance ratios using numpy.cumsum.
# This takes our list of variances [0.32, 0.05, 0.03, ...] 
# and turns it into a running total [0.32, 0.37, 0.40, ...]
cumulative_variance = np.cumsum(explained_variance_ratios)

# (9b) Plot all these cumulative variance ratios.
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='-')
plt.title('Task 9: Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.yticks(np.arange(0, 1.1, 0.1)) # Make y-axis clearer

# (9c) Mark on your plot the component for which the ratio is >= 95%.
# Find the index of the first value that is >= 0.95. Add 1 because index is 0-based.
components_for_95_variance = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Task 9c: {components_for_95_variance} components are needed to explain at least 95% of the variance.")

# Add lines to the plot to make this clear.
plt.axhline(y=0.95, color='r', linestyle=':', label='95% Threshold')
plt.axvline(x=components_for_95_variance, color='g', linestyle=':', label=f'{components_for_95_variance} components for 95% variance')
plt.legend()
plt.show()


# In[14]:


# --- Task 10: PCA on Normalised Data ---
print("--- Task 10: PCA on Normalised Returns ---")

# FIX: Import StandardScaler from the scikit-learn library
from sklearn.preprocessing import StandardScaler

# (10a) Normalise the returns dataframe.
scaler = StandardScaler()
normalized_returns_df = pd.DataFrame(scaler.fit_transform(returns_df), columns=returns_df.columns)

# (10b) Repeat Task 7 for this new dataframe.
pca_normalized = PCA()
pca_normalized.fit(normalized_returns_df)
top_five_pcs_normalized = pd.DataFrame(
    pca_normalized.components_[:5],
    columns=normalized_returns_df.columns,
    index=[f'PC{i+1}' for i in range(5)]
)
print("Task 10b: Top 5 PCs on Normalised Data (weights for first 5 stocks):")
print(top_five_pcs_normalized.iloc[:, :5])
print("\n")

# (10c) Repeat Task 8 for the new dataframe.
explained_variance_normalized = pca_normalized.explained_variance_ratio_
pc1_variance_normalized = explained_variance_normalized[0] * 100
print(f"Task 10c: The first PC (normalised) explains {pc1_variance_normalized:.2f}% of the variance.")

# Plot for 10c
plt.figure(figsize=(12, 6))
component_numbers = np.arange(1, 21) # Define component numbers for the x-axis
plt.plot(component_numbers, explained_variance_normalized[:20], marker='o', linestyle='--')
plt.title('Task 10c: Explained Variance by PC (Normalised Data)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(component_numbers)
plt.grid(True)
elbow_point_norm = 2
plt.axvline(x=elbow_point_norm, color='r', linestyle=':', label=f'Elbow at PC{elbow_point_norm}')
plt.legend()
plt.show()
print("\n")

# (10d) Repeat Task 9 for the new dataframe.
cumulative_variance_normalized = np.cumsum(explained_variance_normalized)
components_for_95_variance_normalized = np.argmax(cumulative_variance_normalized >= 0.95) + 1
print(f"Task 10d: {components_for_95_variance_normalized} components are needed for 95% variance on normalised data.")

# Plot for 10d
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, len(cumulative_variance_normalized) + 1), cumulative_variance_normalized, marker='.', linestyle='-')
plt.title('Task 10d: Cumulative Explained Variance (Normalised Data)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axhline(y=0.95, color='r', linestyle=':', label='95% Threshold')
plt.axvline(x=components_for_95_variance_normalized, color='g', linestyle=':', label=f'{components_for_95_variance_normalized} components for 95% variance')
plt.legend()
plt.show()


# In[ ]:




