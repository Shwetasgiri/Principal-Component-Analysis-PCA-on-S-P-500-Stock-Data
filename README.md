# Principal-Component-Analysis-PCA-on-S-P-500-Stock-Data
Project Overview
This project performs a detailed analysis of historical stock data for companies in the S&P 500 index over a three-year period (November 2019 - October 2022). The core of the analysis is the application of Principal Component Analysis (PCA), a powerful dimensionality reduction technique, to uncover the primary factors driving stock market returns.

The script demonstrates a complete data analysis workflow, from initial data loading and cleaning to advanced statistical modeling and visualization.

Key Findings
Dominant Market Factor: The analysis successfully identified a primary component (PC1) that explains a substantial portion of the market's daily variance, representing the overall market trend or "systematic risk."

Dimensionality Reduction: It was shown that the complex movements of nearly 500 individual stocks can be effectively represented by a much smaller number of principal components.

Impact of Normalisation: The project highlights the critical importance of standardizing data before applying PCA. The results differed between raw and normalized returns, demonstrating how normalization leads to a more robust interpretation of the underlying market structure.

This plot from the analysis shows the cumulative variance explained by the principal components on the normalized data.

How to Run This Project
1. Prerequisites
Ensure you have Python 3.7+ and the necessary libraries installed. You can install them using pip:

pip install pandas numpy scikit-learn matplotlib

2. Data File
Place the stockdata.csv file in the same directory as the PCA.py script.

3. Execute the Script
Run the script from your terminal:

python PCA.py

The script will print the results of each task to the console and save the four analysis plots as .png files in the same directory.

Files in This Repository
PCA.py: The main Python script containing all the code for data cleaning, analysis, and visualization.

Principle_Component_Analysis.md: The detailed project report (converted from the original PDF) that explains the methodology and findings for each step of the analysis.

task8_plot.png: Plot of Explained Variance for raw returns.

task9_plot.png: Plot of Cumulative Explained Variance for raw returns.

task10c_plot.png: Plot of Explained Variance for normalized returns.

task10d_plot.png: Plot of Cumulative Explained Variance for normalized returns.
