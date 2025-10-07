# AER 850 Intro to Machine Learning Project 1
# Ferdos Baikeliaji
# 501050870
# Submission date: Oct 6th 2025


# Step 1: Imports & Setup

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook", palette="Set2")
SEED_VALUE = 37
rnd = np.random.default_rng(SEED_VALUE)

# To make an Result_dir

RESULT_DIR = "."
os.makedirs(RESULT_DIR, exist_ok=True)


# Step 2: Data Loading & Inspection

df_src = pd.read_csv("Project 1 Data.csv")

print(df_src.head(17))
print(df_src.shape)

print("\nUnique counts by column:")
print(df_src.nunique())

# To detect label column

target_col = None
label_candidates = [c for c in df_src.columns
                    if str(c).strip().lower() in {"step", "label", "class", "target", "y"}]
if not label_candidates:
    auto_num_cols = [c for c in df_src.select_dtypes(include=[np.number]).columns
                     if 1 < df_src[c].nunique() <= min(50, max(3, df_src.shape[0] // 20))]
    target_col = auto_num_cols[-1] if auto_num_cols else df_src.columns[-1]
else:
    target_col = label_candidates[0]

print(f"\nAuto-detected label column: {target_col!r}")

# To identify numeric input features

numeric_fields = df_src.select_dtypes(include=[np.number]).columns.tolist()
feature_fields = [c for c in numeric_fields if c != target_col]


# STEP 3: VISUALIZATION

# For Histograms - "Green is the color of the plots"

df_src.hist(
    bins=30,
    figsize=(15, 10),
    grid=True,
    edgecolor="black",
    linewidth=1.2,
    color="green"
)
plt.suptitle("Histograms (All Numeric Columns)")
plt.tight_layout()
plt.show()

# For Histograms - "Green is preffered for the color of the plots"

sel_cols = feature_fields + ([target_col] if target_col in numeric_fields else [])
if sel_cols:
    df_src[sel_cols].hist(
        figsize=(12, 7),
        bins=20,
        edgecolor="black",
        color="green"
    )
    plt.suptitle("Histograms (Selected Numeric Columns)")
    plt.tight_layout()
    plt.show()

# For Custom scatter + boxplots if X,Y,Z available

xyz_ok = all(col in df_src.columns for col in ["X", "Y", "Z"]) and (target_col in df_src.columns)
if xyz_ok:

# "Green is preffered for the color of the plots"

    plt.figure(figsize=(10, 5))
    plt.scatter(df_src["X"], df_src["Z"], c=df_src[target_col], cmap="Greens",
                edgecolors="black", linewidths=0.5, s=35, alpha=0.9)
    plt.title("X vs Z Colored by Label Class")
    plt.xlabel("X"); plt.ylabel("Z")
    plt.colorbar(label=f"{target_col} Class")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(df_src["Y"], df_src["Z"], c=df_src[target_col], cmap="Greens",
                edgecolors="black", linewidths=0.5, s=35, alpha=0.9)
    plt.title("Y vs Z Colored by Label Class")
    plt.xlabel("Y"); plt.ylabel("Z")
    plt.colorbar(label=f"{target_col} Class")
    plt.tight_layout()
    plt.show()

# "Green is preffered for the color of the plots"

    for axis_var in ["X", "Y", "Z"]:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=target_col, y=axis_var, data=df_src,
                    palette=["#66bb66"],  # solid green fill
                    boxprops=dict(edgecolor="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    medianprops=dict(color="black", linewidth=1.5))
        plt.title(f"Boxplot of {axis_var} by {target_col} Class")
        plt.tight_layout()
        plt.show()
else:
    print("\nColumns 'X', 'Y', 'Z', or label not found. Skipping custom scatter and boxplots.")

# For Pearson heatmap - "Green is preffered for the color of the plots"
corr_map = df_src.corr(numeric_only=True, method="pearson")
plt.figure(figsize=(6, 4))
sns.heatmap(corr_map, annot=True, cmap="Greens", vmin=-1, vmax=1,
            linecolor="white", linewidths=0.4)
plt.title("Pearson Correlation Heatmap (numeric only)")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "correlation_heatmap_numeric.png"), dpi=300, bbox_inches="tight")
plt.show()


# Step 4: Preprocessing + Splits

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from scipy.stats import loguniform
import joblib
import numpy as np
import pandas as pd
import os

# Set Target as categorical multiclass

label_aliases = {"step", "label", "class", "target", "y"}
target_col = "Step" if "Step" in df_src.columns else next(
    (c for c in df_src.columns if str(c).strip().lower() in label_aliases), None
)
if target_col is None:
    raise ValueError("Could not identify the target column. Please set `target_col` explicitly.")

# To build a clean feature list

feat_cols = [c for c in df_src.columns
             if c != target_col and pd.api.types.is_numeric_dtype(df_src[c])]
if not feat_cols:
    raise ValueError("No numeric feature columns found after excluding the target.")
print("Using features:", feat_cols)

# Build X, Y

X_mat = df_src[feat_cols].copy()
y_series_orig = df_src[target_col].copy()

# To treat *every distinct* label value as a class; produce int codes 0..K-1 and readable names

y_categ = pd.Categorical(y_series_orig.astype(str))
y_enc = y_categ.codes
class_names = y_categ.categories.astype(str)

# Making sure CV folds not exceed the min class count

class_counts = np.bincount(y_enc)
min_per_class = class_counts.min()
n_splits = max(2, min(5, int(min_per_class)))
print(f"Detected {len(class_names)} classes; counts={class_counts.tolist()} | Using StratifiedKFold(n_splits={n_splits})")

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED_VALUE)

# To satisfy the rain or test split 
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_mat, y_enc, test_size=0.2, random_state=SEED_VALUE, stratify=y_enc
)

# To preprocessing pipeline (median impute + standardize

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

pre = ColumnTransformer([
    ("num", num_pipe, feat_cols)
], remainder="drop")


# Building Decision Tree — GridSearchCV

dt_pipe = Pipeline([
    ("pre", pre),
    ("clf", DecisionTreeClassifier(random_state=SEED_VALUE))
])
dt_grid = {
    "clf__criterion": ["gini", "entropy", "log_loss"],
    "clf__max_depth": [None, 5, 10, 20, 40],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4]
}
dt_search = GridSearchCV(dt_pipe, dt_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)

# Building Random Forest — GridSearchCV
rf_pipe = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(random_state=SEED_VALUE))
])
rf_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20, 40],
    "clf__min_samples_split": [2, 5, 10],
    "clf__max_features": ["sqrt", "log2", None]
}
rf_search = GridSearchCV(rf_pipe, rf_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)

# Building SVM — GridSearchCV
svm_pipe = Pipeline([
    ("pre", pre),
    ("clf", SVC(probability=True, random_state=SEED_VALUE))
])
svm_grid = {
    "clf__kernel": ["rbf", "linear"],
    "clf__C": [0.1, 1, 3, 10, 30, 100],
    "clf__gamma": ["scale", "auto"]
}
svm_search = GridSearchCV(svm_pipe, svm_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)

# Building Logistic Regression — RandomizedSearchCV
lr_pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=2000, random_state=SEED_VALUE))
])
lr_distributions = {
    "clf__C": loguniform(1e-3, 1e2),
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "saga"]
}
lr_search = RandomizedSearchCV(
    lr_pipe, lr_distributions, n_iter=40, scoring="f1_macro",
    cv=cv, n_jobs=-1, random_state=SEED_VALUE, refit=True
)

searches = {
    "DecisionTree": dt_search,
    "RandomForest": rf_search,
    "SVM": svm_search,
    "LogReg(RandSearch)": lr_search
}

# Now fitting all the searches
search_results = {}
for name, search in searches.items():
    search.fit(X_train_raw, y_train)
    search_results[name] = search
    print(f"{name}: best f1_macro (CV) = {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")


# STEP 5: Confusion Matrix (For LR RandomizedSearchCV only)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

lr_key = "LogReg(RandSearch)"
if lr_key not in search_results:
    raise KeyError(f"Expected '{lr_key}' in search_results. Found: {list(search_results.keys())}")

lr_best = search_results[lr_key].best_estimator_
y_pred_lr = lr_best.predict(X_test_raw)
cm_lr = confusion_matrix(y_test, y_pred_lr)

disp = ConfusionMatrixDisplay(cm_lr, display_labels=[str(c) for c in class_names])
disp.plot(cmap="Greens", values_format="d", xticks_rotation=45)
plt.title("Confusion Matrix — Logistic Regression (RandomizedSearchCV)")
plt.tight_layout()
plt.show()


# STEP 6: Stacked Model (For LR + Decision Tree)

dt_key = "DecisionTree"
if dt_key not in search_results:
    raise KeyError(f"Expected '{dt_key}' in search_results. Found: {list(search_results.keys())}")

dt_best = search_results[dt_key].best_estimator_

estimators_for_stack = [
    ("LR", lr_best),
    ("DT", dt_best)
]

stack_clf = StackingClassifier(
    estimators=estimators_for_stack,
    final_estimator=LogisticRegression(max_iter=2000, random_state=SEED_VALUE),
    passthrough=False, n_jobs=-1
)
stack_clf.fit(X_train_raw, y_train)

y_pred_stack = stack_clf.predict(X_test_raw)
cm_stack = confusion_matrix(y_test, y_pred_stack)

disp = ConfusionMatrixDisplay(cm_stack, display_labels=[str(c) for c in class_names])
disp.plot(cmap="Greens", values_format="d", xticks_rotation=45)
plt.title("Confusion Matrix — Stacked (LR + Decision Tree)")
plt.tight_layout()
plt.show()


# STEP 7: Package stacked model & test on provided coordinates

import joblib
import numpy as np
import pandas as pd
import os

# Saving the stacked model and class mapping

model_path  = os.path.join(RESULT_DIR, "best_model_Stacked_LR_DT.joblib")
classes_path = os.path.join(RESULT_DIR, "class_names.npy")
joblib.dump(stack_clf, model_path)
np.save(classes_path, np.array(class_names, dtype=str))
print(f"Saved model → {model_path}")
print(f"Saved class names → {classes_path}")

# Building a prediction DataFrame with the SAME feature columns or ordering as training

def make_feature_df(coords_array, feature_cols):
    base_cols = ["X", "Y", "Z"]
    dfp = pd.DataFrame(coords_array, columns=base_cols[:coords_array.shape[1]])
    return dfp.reindex(columns=feature_cols, fill_value=np.nan)

# Provided coordinate triples

to_predict = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.0,   3.0625, 1.93],
    [9.4,   3.0,    1.8],
    [9.4,   3.0,    1.3],
], dtype=float)

X_new = make_feature_df(to_predict, feat_cols)

# The print shows

pred_codes = stack_clf.predict(X_new)
pred_labels = [str(class_names[i]) for i in pred_codes]

print("\nPredictions for provided coordinates (Stacked LR + DT):")
for coords, lab in zip(to_predict.tolist(), pred_labels):
    print(f"  {coords}  →  step: {lab}")
