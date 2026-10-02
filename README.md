# 🛒 Smart E-Commerce Intelligence System

**@fetch.ai.rcpit**

> An interactive, beginner-friendly Machine Learning workshop built on a synthetic e-commerce customer dataset. Two business questions, four model families, one end-to-end supervised ML workflow.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Colab](https://img.shields.io/badge/Google-Colab-yellow) ![Data](https://img.shields.io/badge/Data-Synthetic-lightgrey)

---

## 📌 Overview

This project answers two business questions for an online store:

| # | Business question | ML task | Target column |
|---|-------------------|---------|---------------|
| 1 | **How much will a customer spend?** | Regression | `Annual_Spending` (₹) |
| 2 | **Will a customer churn?** | Classification | `Churn` (0 = stay, 1 = churn) |

Models covered:

1. **Linear Regression** – predict annual spending
2. **Logistic Regression** – predict churn
3. **Decision Tree** (Regressor + Classifier) – rule-based predictions
4. **Random Forest** (Regressor + Classifier) – ensemble learning

> ⚠️ **Important:** The dataset is **synthetic and educational**. Results are for learning only and must not be treated as real business evidence.

---

## 📂 Repository Structure

```
.
├── README.md
├── SMART_E-COMMERCE_INTELLIGENCE_SYSTEM_-__ML_Project_.ipynb   # Main Colab notebook
├── ecommerce_customers_200_clean.csv                            # Dataset (200 customers, 18 columns)
├── ecommerce_data_dictionary.csv                                # Column definitions
└── docs/
    ├── SPYTHON_-_Rise_to_the_Machine_Learning__ML_Part_.pdf     # ML session slides (Day 1)
    └── git-cheat-sheet-education.pdf                            # Git quick reference
```

---

## 📊 Dataset

- **Rows:** 200 customers  **Columns:** 18  **Duplicates:** 0  **Unique IDs:** 200
- **Class balance (Churn):** 145 No Churn / 55 Churn

| Column | Type | Meaning |
|--------|------|---------|
| `Customer_ID` | Identifier | Unique ID – *not used as a feature* |
| `Age` | Numeric | Age in years |
| `Gender` | Categorical | Male / Female / Other |
| `City_Tier` | Categorical | Tier 1 / 2 / 3 |
| `Income` | Numeric | Approx. monthly income (INR) |
| `Tenure_Months` | Numeric | Months since joining |
| `Visits_Per_Month` | Numeric | Avg. monthly website/app visits |
| `Products_Bought` | Numeric | Products purchased in the observation period |
| `Avg_Order_Value` | Numeric | Average order value (INR) |
| `Discount_Usage_Rate` | Numeric | Share of orders using a discount (0–1) |
| `Cart_Abandonment_Rate` | Numeric | Share of carts abandoned (0–1) |
| `Reviews_Given` | Numeric | Reviews submitted |
| `Support_Tickets` | Numeric | Support requests |
| `Days_Since_Last_Purchase` | Numeric | Days since last purchase |
| `Membership_Type` | Categorical | None / Basic / Premium |
| `Preferred_Category` | Categorical | Most frequently purchased category |
| `Annual_Spending` | **Target** | Regression target (INR) |
| `Churn` | **Target** | Classification target (0/1) |

Full definitions: [`ecommerce_data_dictionary.csv`](ecommerce_data_dictionary.csv)

---

## 🔬 Workflow

```
Business Question → Dataset → Features & Target → Train/Test Split
        → Model → Prediction → Evaluation → Interpretation
```

1. Import libraries (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn)
2. Load and inspect the data (shape, dtypes, missing values, duplicates, summary stats)
3. Visual exploration (spending distribution, income vs spending, churn counts, spending by membership)
4. **Part A – Regression:** 80/20 split → Linear Regression, Decision Tree, Random Forest → MAE / RMSE / R² → residual plot
5. **Part B – Classification:** stratified 80/20 split → Logistic Regression, Decision Tree, Random Forest → Accuracy / Precision / Recall / F1 / ROC-AUC → confusion matrices → threshold analysis → feature importance
6. Interactive prediction on a single customer + student mini-challenges

**Preprocessing:** `OneHotEncoder` for categorical columns, `StandardScaler` for numeric columns (linear/logistic models), combined with `ColumnTransformer` + `Pipeline`. 160 train rows / 40 test rows, `random_state=42`.

---

## 📈 Results (test set, 40 customers)

### Regression – `Annual_Spending`

| Model | MAE (₹) | RMSE (₹) | R² |
|-------|--------:|---------:|---:|
| **Linear Regression** | **9,330** | **12,030** | **0.462** |
| Decision Tree (depth 5) | 12,053 | 15,783 | 0.074 |
| Random Forest (200 trees, depth 8) | 10,672 | 13,136 | 0.359 |

### Classification – `Churn`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| Logistic Regression | 0.725 | 0.500 | 0.273 | 0.353 | 0.702 |
| Decision Tree (depth 5) | 0.625 | 0.250 | 0.182 | 0.211 | 0.597 |
| **Random Forest** | **0.750** | **0.667** | 0.182 | 0.286 | **0.749** |

**Threshold effect (Logistic Regression):** lowering the churn threshold from 0.50 to 0.30 raises recall from 0.273 to 0.727 while precision stays at 0.500.

**Top Random Forest features for churn:** `Days_Since_Last_Purchase` (~0.18), `Cart_Abandonment_Rate` (~0.10), `Avg_Order_Value` (~0.09), `Visits_Per_Month` (~0.09).

### Takeaways

- Simple Linear Regression beat the tree-based models on this small dataset.
- With only 40 test rows and 11 churners, **accuracy alone is misleading** – look at recall, F1 and ROC-AUC.
- Feature importance shows association, **not causation**.

---

## 🚀 Getting Started

### Option 1 – Google Colab (recommended)

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run the import cells.
3. When prompted, upload `ecommerce_customers_200_clean.csv` (and optionally the data dictionary).
4. Run all remaining cells top to bottom.

### Option 2 – Local

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

> The notebook uses `google.colab.files.upload()`. For local runs, replace that cell with `df = pd.read_csv("ecommerce_customers_200_clean.csv", keep_default_na=False)`.

---

## ⚠️ Known Issue – `"None"` read as missing value

`Membership_Type` contains the legitimate category **`"None"`** (55 customers with no membership). By default, `pd.read_csv` treats the text `"None"` as `NaN`, so the final verification cell reports **55 missing values** and `df.dropna()` then silently deletes those 55 customers (200 → 145 rows).

**Fix:** load with

```python
df = pd.read_csv("ecommerce_customers_200_clean.csv", keep_default_na=False)
```

and update the final assertions to `df.shape == (200, 18)` with no `dropna()`.

---

## 🧰 Git Workflow (quick reference)

```bash
git init                         # start a repo
git add .                        # stage changes
git commit -m "Add ML notebook"  # snapshot
git branch feature/readme        # new branch
git checkout feature/readme      # switch branch
git merge feature/readme         # merge into current branch
git remote add origin <url>      # connect remote
git push origin main             # upload
git pull                         # fetch + merge
```

More commands in [`docs/git-cheat-sheet-education.pdf`](docs/git-cheat-sheet-education.pdf).

---

## ✅ Good ML Practice Used Here

- Never use the target as an input feature
- Keep the test set separate from training
- Avoid data leakage (preprocessing inside a `Pipeline`)
- Understand each metric before trusting it
- Remember: synthetic data ≠ real-world evidence

---

## 🧪 Student Mini-Challenges

1. **Regression:** change `Visits_Per_Month`, `Avg_Order_Value`, `Products_Bought` – watch predicted spending.
2. **Classification:** raise `Days_Since_Last_Purchase`, `Cart_Abandonment_Rate`, `Support_Tickets` – watch churn probability.
3. **Comparison:** why can Linear Regression behave differently from a Decision Tree? Why does Random Forest use many trees?
4. **Business:** which traits relate to higher spending/churn *in this dataset* – without claiming causation?

---

## 🙌 Credits

- **@fetch.ai.rcpit** – workshop / project
- Course material: *SPYTHON – Rise to the Machine Learning*
- Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Google Colab

---

<p align="center">Made with ❤️ for learning ML · <b>@fetch.ai.rcpit</b></p>
