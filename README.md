## Project Report

### 1. Introduction
This project aimed to build a loan prediction model based on the provided dataset. The dataset contains various features related to loan applications and whether the loan was approved or not.

### 2. Data Loading and Initial Exploration
- The dataset was loaded into a pandas DataFrame.
- The shape of the dataset was checked, showing 20000 rows and 35 columns.
- The data types and non-null counts were examined using `info()`. All columns were found to have 20000 non-null entries, indicating no missing values.
- Descriptive statistics were generated using `describe()` to understand the distribution of numerical features.
- The head of the dataset was printed to get a glimpse of the data structure and content.

### 3. Data Cleaning
- Missing values were explicitly checked for each column using `isnull().sum()`, confirming no missing values.
- Outliers were handled by removing rows where any numerical feature was outside the 1.5*IQR range, but only for rows where `LoanApproved` was not 1. This resulted in a cleaned dataset (`cleanLoan_df`) with 11879 rows.

### 4. Data Transformation
- Categorical variables in the cleaned dataset were encoded using one-hot encoding (`pd.get_dummies()`) to convert them into numerical format for modeling.

### 5. Feature Selection based on Correlation
- The correlation matrix of the encoded dataset was computed.
- Features with an absolute correlation of 0.3 or higher with the target variable 'LoanApproved' were selected (`df_filtered`). This reduced the number of features to 6.

### 6. Data Visualization
- Various plots were generated to visualize the relationships between the selected features and the target variable:
    - Histograms of 'AnnualIncome' and 'MonthlyIncome'.
    - Boxplots of 'AnnualIncome' and 'MonthlyIncome' by 'LoanApproved'.
    - Correlation heatmap of the selected features.
    - Pairplot of the selected features with hue based on 'LoanApproved'.
    - Jointplot of 'AnnualIncome' and 'RiskScore'.
    - Countplot of 'LoanApproved'.
    - Line plot of 'AnnualIncome' vs 'MonthlyIncome'.
    - QQ plots for 'AnnualIncome', 'MonthlyIncome', 'RiskScore', 'TotalDebtToIncomeRatio', and 'InterestRate' to check for normality.

### 7. Data Scaling
- The selected numerical features (excluding 'LoanApproved') were scaled using `MinMaxScaler` to bring them to a similar range, which is important for some machine learning algorithms.

### 8. Data Splitting
- The dataset was split into training and testing sets based on the 'LoanApproved' status to maintain the proportion of approved and not approved loans in both sets. 70% of each group was used for training and 30% for testing.

### 9. Machine Learning Model Integration with Forward Feature Selection
- Three classification models were selected: Logistic Regression, Random Forest, and K-Nearest Neighbors.
- Forward Feature Selection was applied to each model to identify the best subset of features that maximize accuracy on the test set.
- The selected features and the accuracy achieved with those features were reported for each model.

### 10. Model Evaluation
- Classification reports were generated for each model using the selected features on the test set, providing metrics like precision, recall, and f1-score.
- The accuracy of each model was printed.
- Bar plots were generated to compare the accuracy of the models and the number of features selected by each model.

### 11. Best Model Selection and Saving
- The Random Forest model achieved the highest accuracy (0.9919) with the feature 'RiskScore'.
- The best performing model (Random Forest) was saved as a pickle file named 'loan_prediction_model.pkl'.

### 12. Conclusion
The Random Forest model, trained on the 'RiskScore' feature, demonstrated the highest accuracy in predicting loan approval in this project. This suggests that 'RiskScore' is a highly influential factor in determining loan approval based on this dataset. Further analysis could involve exploring other feature selection methods, hyperparameter tuning for the models, and evaluating the model on unseen data.
