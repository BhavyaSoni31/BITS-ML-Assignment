# Shopper Intent Lab — Predicting Purchases from E-commerce Sessions

## a. Problem statement

An online retailer logs every browsing session: which page types were visited, how long the visitor
stayed, Google-Analytics engagement signals and some context (month, browser, region, traffic source).
Only about 1 in 6 sessions ends in a sale. The goal is to build a **binary classifier** that predicts,
from these session attributes, whether a session will end in a **purchase (`Revenue = True`)** or not.
Five classical models (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) are trained on the
same data and compared on six metrics. They are then served through an interactive Streamlit app
that scores an uploaded test CSV with the saved models. The app never re-trains a model.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository — *Online Shoppers Purchasing Intention Dataset* (id 468), Sakar et al., 2018 |
| Link | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances | 12,330 sessions (12,205 after removing 125 exact duplicate rows) |
| Input features | 17 (10 numeric + 7 categorical) |
| Target | `Revenue` — `True` (purchase) / `False` (no purchase) |
| Class balance | 15.6 % purchase vs 84.4 % no purchase (imbalanced) |
| Missing values | None |

**Features**

| Group | Columns | Meaning |
|---|---|---|
| Page counts | `Administrative`, `Informational`, `ProductRelated` | Number of pages of each type visited in the session |
| Page durations | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | Seconds spent on each page type |
| Google Analytics | `BounceRates`, `ExitRates`, `PageValues` | Average bounce rate, exit rate and page value of the visited pages |
| Calendar | `SpecialDay`, `Month`, `Weekend` | Closeness to a special day (e.g. Valentine's), month of visit, weekend flag |
| Visitor context | `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` | Integer-coded technical / geographic info, and New / Returning / Other visitor |

**Preprocessing (identical for all five models, stored inside each saved pipeline)**

1. Removed exact duplicate rows. Used a stratified 80/20 split → 9,764 training / 2,441 hold-out rows.
   The hold-out split is saved as [`test_data.csv`](test_data.csv), with features and target.
2. Added four engineered behaviour features: `TotalPages`, `TotalDuration`,
   `SecondsPerProductPage` and `HasPageValue`.
3. Numeric block: median imputation, then a Yeo-Johnson power transform with standardisation.
   This tames the very skewed duration and page-value columns.
4. Categorical block (`Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`,
   `Weekend`): one-hot encoding. Rare levels (< 15 rows) are grouped, and unseen levels are tolerated.
5. Hyper-parameters were tuned by a 5-fold stratified grid search that maximises **MCC**, because
   MCC is robust to class imbalance. Each SVM gets probability outputs through sigmoid
   calibration, which is needed for AUC.

## c. GitHub Repository Link

**Repository:** `https://github.com/BhavyaSoni31/BITS-ML-Assignment/`

**Live Streamlit app:** `https://bits-ml-assignment-mgh8ntayb9ftq3prlrwahq.streamlit.app/`

```
BITS-ML-Assignment/
├── app.py                         # Streamlit front-end (loads saved models, no retraining)
├── requirements.txt
├── README.md
├── test_data.csv                  # hold-out test split (17 features + Revenue)
├── data/
│   └── online_shoppers_intention.csv   # raw UCI data used for training
├── model/
│   ├── shopper_toolkit.py         # feature schema, feature-engineering transformer, metric helper
│   ├── train_models.py            # tuning + training + evaluation script
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── leaderboard.csv            # hold-out metrics of all models
│   └── training_report.json       # best hyper-parameters, CV scores, schema
└── .streamlit/config.toml         # custom theme
```

**Reproduce**

```bash
pip install -r requirements.txt
python -m model.train_models      # re-creates model/*.joblib, leaderboard.csv, test_data.csv
streamlit run app.py
```

## d. Models used

All metrics below were computed on the same untouched hold-out set (`test_data.csv`, 2,441 sessions).
The decision threshold is 0.5. Precision, Recall and F1 refer to the positive class (*Purchase*).
AUC is the ROC-AUC of the predicted purchase probability.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8783 | 0.9162 | 0.5783 | **0.8220** | **0.6789** | **0.6209** |
| Decision Tree | **0.9033** | 0.9295 | 0.7267 | 0.6126 | 0.6648 | 0.6118 |
| KNN | 0.9005 | **0.9296** | 0.7038 | 0.6283 | 0.6639 | 0.6070 |
| Naive Bayes | 0.8689 | 0.8802 | 0.5576 | 0.7853 | 0.6522 | 0.5870 |
| Support Vector Machine | 0.9017 | 0.8930 | **0.7536** | 0.5524 | 0.6375 | 0.5915 |

Tuned hyper-parameters (from the grid search):

| Model | Best parameters | CV MCC |
|---|---|---|
| Logistic Regression | `C=0.003`, `class_weight='balanced'` | 0.5970 |
| Decision Tree | `criterion='gini'`, `max_depth=7`, `min_samples_leaf=60` | 0.5938 |
| KNN | `n_neighbors=75`, `weights='distance'`, `p=2` (Euclidean) | 0.5903 |
| Naive Bayes (Gaussian) | `var_smoothing=1.0` | 0.5608 |
| SVM (RBF kernel) | `C=1.0`, `gamma='scale'` | 0.6142 |

> A model that always predicts "No Purchase" already scores 0.844 accuracy on this data. Accuracy alone
> is therefore misleading here, so the comparison relies mainly on **MCC, F1 and AUC**.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best MCC (0.621) and best F1 (0.679), although it is the simplest model. After the Yeo-Johnson transform, the purchase signal is close to linear, and it is driven mostly by `PageValues` and `ExitRates`. A tiny `C` (strong L2 regularisation) generalised best. `class_weight='balanced'` moves the boundary towards the minority class. The model therefore catches 82 % of real buyers (highest recall), but pays for it with lower precision (0.578) and the second-lowest accuracy. This is a good fit when missing a buyer costs more than a false alarm, for example in retargeting campaigns. |
| Decision Tree | Highest accuracy (0.903) and an AUC (0.930) that matches KNN. The depth-7 tree with at least 60 samples per leaf is small enough to avoid over-fitting, and you can read its rules directly. The root split is on `PageValues`, which is the dataset's dominant signal (54 leaves in total). It is more conservative than Logistic Regression: precision 0.727, recall 0.613. Its probabilities are coarse, piece-wise constant leaf frequencies, yet the ranking quality is still strong. |
| KNN | Top AUC (0.9296) and very close to the tree on every other metric. Many neighbours (k = 75) with distance weighting were needed, because buyers are a sparse minority and small neighbourhoods were noisy. Its predictions depend on the scaled feature space, so the power transform was essential. Drawbacks: the saved model is the largest (it stores the whole training set), and each prediction searches that stored set, so it is lazy at inference time. The neighbourhood vote also never fully trades precision for recall, so recall stays at 0.628. |
| Naive Bayes | Weakest overall: lowest accuracy (0.869), AUC (0.880) and MCC (0.587). The independence assumption is clearly violated, because page counts, durations, the engineered totals and bounce/exit rates are strongly correlated. The model therefore double-counts evidence and is over-confident. Grid search chose very heavy variance smoothing (1.0) to soften this. Like LR it leans towards recall (0.785) but has the lowest precision (0.558). Its advantages are near-instant training and a tiny model. |
| Support Vector Machine | Best cross-validation MCC (0.614) and the highest hold-out precision (0.754): when it predicts a purchase, it is usually right. On the hold-out set it is the most conservative model, with the lowest recall (0.552). Its AUC (0.893) is lower than LR, DT and KNN, so its score ranking is less reliable away from the default threshold. The gap between CV MCC and hold-out MCC (0.614 → 0.592) points to some variance. It was also by far the slowest model to tune, calibrate and score. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It has the best MCC (0.621) and F1 (0.679), which are the most informative metrics for this imbalanced problem, and it is the cheapest and most interpretable model. Decision Tree and KNN are close runners-up: they win on accuracy and AUC, but they detect fewer buyers. If false alarms are expensive, the precision-oriented SVM or Decision Tree would be better choices. The app's cut-off slider lets you explore this trade-off for every model. |

## Streamlit app features

- **CSV upload** of test data (sidebar). The bundled `test_data.csv` can be downloaded or used directly.
  Column validation flags missing features. The target may be `True/False`, `1/0` or `yes/no`.
- **Model selection dropdown** covering all five saved classifiers.
- **Evaluation metrics**: all six metrics shown as tiles, next to the values recorded at training time.
- **Confusion matrix** (counts and row percentages) and a **classification report**.
- Extra views: an ROC overlay of all models, a side-by-side comparison table and chart of the five models
  on the uploaded data, a row-level predictions table with CSV download, an adjustable decision cut-off,
  and the tuning notes.
- Models are only loaded with `joblib` and used for `predict` / `predict_proba`. Nothing is re-trained in the app.
