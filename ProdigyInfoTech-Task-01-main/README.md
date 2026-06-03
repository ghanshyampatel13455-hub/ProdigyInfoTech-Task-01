**Prodigy Infotech - Mechine Learning Internship**  
**Task 01 : House Prices - Advanced Regression Techniques**

### Project Overview
This project focuses on building a predictive model to estimate residential property prices in Ames, Iowa, using the famous Ames Housing Dataset. The goal is to develop a robust regression model that accurately predicts `SalePrice` based on various structural, location, and quality features of houses.

By working on this end-to-end project, I gained hands-on experience in real-world data science workflows — from data exploration and preprocessing to model training, evaluation, and deployment-ready predictions.

### Skills Developed
- **Data Loading & Exploratory Data Analysis (EDA)**: Understanding dataset structure, handling missing values, and visualizing feature distributions and relationships.
- **Feature Engineering & Preprocessing**: Handling numerical and categorical variables, imputing missing data (mean for numeric, mode for categorical).
- **Machine Learning Modeling**: Implementing Linear Regression as a baseline model using scikit-learn.
- **Model Evaluation**: Using metrics like MAE, MSE, and R² Score; visualizing predictions vs actual values and residual analysis.
- **Prediction & Submission**: Generating predictions on the test set and creating a submission file in the required format.
- **Data Visualization**: Using Matplotlib and Seaborn for pair plots, scatter plots, residual plots, and distribution analysis.

### Tools & Technologies Used
- **Python**
- **Pandas** & **NumPy** (Data manipulation)
- **Matplotlib** & **Seaborn** (Visualization)
- **Scikit-learn** (Model training & evaluation)
- **Jupyter Notebook** / VS Code

### Project Workflow
1. **Data Loading**  
   Loaded `train.csv` and `test.csv` files containing 80+ features describing house attributes.

2. **Data Preprocessing**  
   - Identified and filled missing values appropriately.
   - Selected key predictive features: `GrLivArea`, `BedroomAbvGr`, `FullBath`, `HalfBath`, `TotRmsAbvGrd`.

3. **Exploratory Data Analysis**  
   - Created pair plots to understand feature relationships.
   - Analyzed distributions and correlations with `SalePrice`.

4. **Model Training**  
   - Split training data into train/validation sets.
   - Trained a **Linear Regression** model.

5. **Model Evaluation**  
   - Evaluated on validation set using MAE, MSE, and R².
   - Plotted Actual vs Predicted values and Residual plots to assess model performance.

6. **Prediction & Submission**  
   - Generated predictions on the test dataset.
   - Saved results to `submission.csv` in the required Kaggle format.

### Key Results
- **Example Prediction**: A house with 2000 sq ft living area, 3 bedrooms, 2 full baths, 1 half bath, and 7 total rooms above grade was predicted at **$240,896**.

- Strong linear relationship observed between `GrLivArea` and `SalePrice`.
- Residual analysis shows reasonable model fit for a baseline linear model.

### Future Improvements
- Advanced feature engineering (e.g., total square footage, age of house, neighborhood encoding).
- Trying ensemble models (Random Forest, XGBoost, Gradient Boosting).
- Hyperparameter tuning and cross-validation.
- Handling outliers and incorporating more categorical features.

---

**Repository Structure**
```
Prodigy_House_Price_Prediction/
├── train.csv
├── test.csv
├── task1.ipynb          # Main Jupyter Notebook
├── submission.csv       # Model predictions
├── data_description.txt # Feature descriptions
└── README.md
```
