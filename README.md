# 🏠 House Price Predictor — Linear Regression

> **Prodigy InfoTech Machine Learning Internship — Task 1**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=flat&logo=pandas)](https://pandas.pydata.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20House%20Prices-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 📌 Project Overview

A complete end-to-end supervised machine learning pipeline that predicts residential house sale prices based on **square footage**, **number of bedrooms**, and **number of bathrooms** using the Kaggle House Prices dataset.

The project includes:
- Full ML pipeline (data → features → model → evaluation → prediction)
- 7 engineered features from Kaggle's raw columns
- Live interactive web dashboard with real-time price prediction sliders
- Annotated source code with step-by-step explanations

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| R² Score | **0.902** |
| RMSE | **$19,947** |
| MAE | **$15,896** |
| MAPE | **5.82%** |
| CV R² (5-fold) | **0.884 ± 0.011** |
| Train size | 1,168 houses |
| Test size | 292 houses |

> The model explains **90.2%** of house price variance with an average prediction error of ~$20k on a ~$293k average price — roughly **6.8% error rate**.

---

## 📁 Project Structure

```
house-price-predictor/
│
├── house_price_prediction.py   # Main ML pipeline (training + evaluation)
├── house_price_website.html    # Interactive web dashboard
├── README.md                   # This file
│
├── data/
│   └── train.csv               # Kaggle dataset (download separately)
│
└── outputs/
    ├── results.png             # Actual vs Predicted + Residuals plot
    └── house_price_model.pkl   # Saved model (generated after training)
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/house-price-predictor.git
cd house-price-predictor
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 3. Download the dataset
- Go to [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- Download `train.csv`
- Place it in the `data/` folder

### 4. Run the model
```bash
python house_price_prediction.py
```

### 5. Open the web dashboard
Simply open `house_price_website.html` in any browser — no server needed.

---

## 🔧 Features Used

| Column (Kaggle name) | Description | Type |
|---|---|---|
| `GrLivArea` | Above-grade living area (sq ft) | Raw |
| `BedroomAbvGr` | Bedrooms above grade | Raw |
| `TotalBath` | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath | Engineered |
| `TotRmsAbvGrd` | Total rooms above grade | Raw |
| `SqftPerRoom` | GrLivArea ÷ TotRmsAbvGrd | Engineered |
| `SqftPerBed` | GrLivArea ÷ BedroomAbvGr | Engineered |
| `BedBathRatio` | BedroomAbvGr ÷ TotalBath | Engineered |

---

## 🧠 Model Pipeline

```
Raw Data (train.csv)
        ↓
Feature Engineering (7 features)
        ↓
Train/Test Split (80% / 20%)
        ↓
StandardScaler Normalization
        ↓
LinearRegression (OLS)  ←── model.fit(X_train, y_train)
        ↓
Evaluation (R², RMSE, MAE, 5-fold CV)
        ↓
Predict new houses
```

---

## 📊 Feature Importance (Scaled Coefficients)

| Feature | Coefficient | Importance |
|---|---|---|
| GrLivArea | +$59,017 | 70.2% |
| TotalBath | +$7,991 | 9.5% |
| BedBathRatio | −$5,410 | 6.4% |
| TotRmsAbvGrd | +$5,367 | 6.4% |
| BedroomAbvGr | +$5,031 | 6.0% |
| SqftPerBed | −$795 | 0.9% |
| SqftPerRoom | +$418 | 0.5% |

> **Square footage dominates** at 70.2% — consistent with real-world housing market behavior.

---

## 💻 Code Example — Predict a New House

```python
import pickle
import pandas as pd

# Load saved model
with open('house_price_model.pkl', 'rb') as f:
    saved = pickle.load(f)
    model  = saved['model']
    scaler = saved['scaler']

# Define a new house
new_house = pd.DataFrame([{
    'GrLivArea':    1800,   # 1800 sq ft
    'BedroomAbvGr': 3,      # 3 bedrooms
    'TotalBath':    2.5,    # 2 full + 1 half bath
    'TotRmsAbvGrd': 7,      # 7 total rooms
    'SqftPerRoom':  1800/7,
    'SqftPerBed':   1800/3,
    'BedBathRatio': 3/2.5,
}])

# Scale and predict
price = model.predict(scaler.transform(new_house))[0]
print(f"Predicted Price: ${price:,.0f}")
# → Predicted Price: $327,842
```

---

## 📈 Visualizations

The web dashboard (`house_price_website.html`) includes:

- **Actual vs Predicted scatter plot** — how close are predictions to reality?
- **Price vs Square Footage line chart** — linear trend with all other features fixed
- **Residuals histogram** — bell-shaped curve centred at 0 = good model
- **5-Fold CV bar chart** — consistent R² across all folds = no overfitting
- **Feature importance bar chart** — which features matter most?
- **Live price predictor** — drag sliders to predict any house instantly

---

## 📚 What I Learned

- How to read and apply a real-world dataset description (`data_description.txt`)
- Feature engineering — turning raw columns into meaningful model inputs
- Why `StandardScaler` is critical before fitting linear models
- The difference between R², RMSE, MAE, and MAPE — and when to use each
- Cross-validation to confirm a model generalises, not just memorises
- Saving and loading models with `pickle` for deployment
- Building a standalone interactive web dashboard from a trained Python model

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| scikit-learn | ML model, scaler, metrics, CV |
| Matplotlib / Seaborn | Charts and plots |
| HTML / CSS / JavaScript | Web dashboard |
| Chart.js | Interactive browser charts |

---

## 📄 Dataset

**Kaggle House Prices: Advanced Regression Techniques**
- 1,460 training samples
- 79 features (we use 4 raw + 3 engineered)
- Target: `SalePrice` (residential home sale price in Ames, Iowa)
- Link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

---

## 👤 Author

**Your Name**
B.Tech AI & DS — 7th Semester
Machine Learning Intern @ Prodigy InfoTech

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/yourusername)

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If this project helped you, please give it a star on GitHub!
