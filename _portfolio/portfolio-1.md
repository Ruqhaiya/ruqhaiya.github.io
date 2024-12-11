---
title: "Predicting-Electricity-Load-in-New-Hampshire"
collection: portfolio
---
The primary objective of this project is to develop a predictive model that accurately forecasts the hourly electricity load for the New Hampshire zone. This forecast will cover the upcoming month, providing crucial insight into the fund's trading activities. To achieve this, we will analyze historical electricity load data from 2020 to 2022, focusing exclusively on New Hampshire.
 
## Code Repository

### Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Visualization](#visualization)
- [Getting Started](#getting-started)
   - [Prerequisite](#prerequisite)
   - [Installation](#installation)

- [How To Use](how-to-use)
- [Versioning](versioning)



Here is the structured content based on the Word file you uploaded. I will include relevant sections, summaries, and descriptions of diagrams as requested.



### Overview
The project aims to develop a predictive model to accurately forecast the **hourly electricity load for New Hampshire** for an upcoming month. This is critical for insights into energy trading and operational decisions. The study uses data from 2020 to 2022 and employs various machine learning techniques like linear regression, KNN, and Random Forest.



### Features
1. Data Integration:
   - Combines datasets from 2020 to 2022.
   - Data cleaning and transformation to improve quality.
2. Exploratory Data Analysis (EDA):
   - Comprehensive analysis to identify trends and correlations.
3. Feature Engineering:
   - Interaction terms like dry bulb and dew point.
   - Transformation of variables for linearity.
4. Modeling Techniques:
   - Linear regression, KNN, and Random Forest.
   - Custom transformations for non-linear relationships.
5. Performance Metrics:
   - Evaluates models using Mean Square Error (MSE).



### Visualization
1. Scatter Plots:
   - Visualize relationships between load and variables like dry bulb and dew point.
   - Example: Figure 9 shows the relationship between load and dry bulb.

2. Time Series Plots:
   - Load patterns over time.
   - Example: Figure 2 displays hourly load trends in New Hampshire.

3. Residual Plots:
   - Assess model performance.
   - Example: Figures 21 and 22 highlight residuals for Linear Regression and Random Forest models.

4. Boxplots:
   - Monthly and hourly load distributions.
   - Example: Figure 15 demonstrates monthly load patterns.



### Getting Started
1. Objective: Predict the hourly electricity load for New Hampshire for one month.
2. Data Sources:
   - "2020 SMD Hourly Data"
   - "2021 SMD Hourly Data"
   - "2022 SMD Hourly Data"
3. Approach:
   - EDA, feature engineering, and predictive modeling.



### Prerequisite
1. Hardware:
   - Minimum 8 GB RAM.
   - Storage for dataset files.
2. Software:
   - Python with libraries like Pandas, NumPy, and scikit-learn.
   - Google Colab or similar Python environment.



### Installation
1. Clone the Repository:
   ```bash
   git clone https://github.com/your-repo/energy-forecasting.git
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Scripts:
   ```bash
   python main.py
   ```



### How To Use
1. Load Data:
   - Place dataset files in the `data/` directory.
2. Run EDA:
   - Execute `eda.py` to generate visualizations and insights.
3. Train Model:
   - Use `train_model.py` to train predictive models.
4. Evaluate Performance:
   - Outputs include residual plots and MSE metrics.



### Versioning
1. Current Version: 1.0
2. Release Notes:
   - Implemented Random Forest model.
   - Enhanced feature engineering.
   - Added visualizations for analysis.



### Diagrams (Descriptions Included)

#### Figure 2: Hourly Load in New Hampshire in Time Series
![image](https://github.com/user-attachments/assets/21b694a9-cb9b-420e-a4cb-d3a4562e7e3f)

A time series plot showing hourly electricity load trends for New Hampshire over three years. Clear seasonal patterns are visible, with peaks during summer and winter months.

#### Figure 15: Boxplot of Monthly Load
![image](https://github.com/user-attachments/assets/6c09dbc5-880d-4055-b6cc-13aa0ba52182)

A boxplot showcasing monthly variations in electricity load. Peaks correspond to summer (air conditioning demand) and winter (heating demand).

#### Figure 21: Residual Plot for Linear Regression
![image](https://github.com/user-attachments/assets/9549952e-2cd0-4ad7-a31c-2ca894bc3fe9)

Displays the residuals (differences between actual and predicted values) for a linear regression model with 16 features. Points are symmetrically distributed around zero.

#### Figure 22: Residual Plot for Random Forest
![image](https://github.com/user-attachments/assets/6a6d81b8-20b3-47bf-b4ba-3a59ca180588)

Highlights the Random Forest model’s residuals, showing improved performance compared to linear regression.






 
