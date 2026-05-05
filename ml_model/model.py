import pandas as pd
import numpy as np

y = df_merged["MedInc"].copy()
X = df_merged.drop(columns=["MedInc"]).copy()

groups = df_merged["msa_name"]  # for group-aware splitting/CV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

CAT_FEATURES = ["msa_name"]
NUM_FEATURES = [c for c in X.columns if c not in CAT_FEATURES]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, NUM_FEATURES),
        ("cat", categorical_pipe, CAT_FEATURES),
    ],
    remainder="drop",
)

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def make_pipeline(model_name: str):
    models = {
        "ridge": Ridge(random_state=42),
        "lasso": Lasso(random_state=42),
        "rf": RandomForestRegressor(random_state=42, n_estimators=300),
        "gbr": GradientBoostingRegressor(random_state=42),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model_name={model_name}. Choose from {list(models.keys())}")

    return Pipeline([
        ("preprocess", preprocess),
        ("model", models[model_name]),
    ])

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def train_eval(pipe, X, y, groups, test_size=0.2, random_state=42):
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": root_mean_squared_error(y_test, pred),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_msas": X_train["msa_name"].nunique(),
        "test_msas": X_test["msa_name"].nunique(),
    }

pipe_ridge = make_pipeline("ridge")
metrics_ridge = train_eval(pipe_ridge, X, y, groups)
metrics_ridge

results = []
for name in ["ridge", "lasso", "gbr", "rf"]:
    pipe = make_pipeline(name)
    m = train_eval(pipe, X, y, groups)
    m["model"] = name
    results.append(m)

df_results = pd.DataFrame(results).sort_values("r2", ascending=False)
df_results

from sklearn.model_selection import GroupKFold, cross_validate
import pandas as pd

cv = GroupKFold(n_splits=5)

def group_cv(pipe, X, y, groups):
    scores = cross_validate(
        pipe, X, y,
        cv=cv,
        groups=groups,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
        n_jobs=-1,
        return_train_score=False
    )
    return {
        "r2_mean": scores["test_r2"].mean(),
        "r2_std": scores["test_r2"].std(),
        "mae_mean": (-scores["test_mae"]).mean(),
        "mae_std": (-scores["test_mae"]).std(),
        "rmse_mean": (-scores["test_rmse"]).mean(),
        "rmse_std": (-scores["test_rmse"]).std(),
    }

models_to_check = ["gbr", "rf"]  # focus on the top two
rows = []
for name in models_to_check:
    pipe = make_pipeline(name)
    m = group_cv(pipe, X, y, groups)
    m["model"] = name
    rows.append(m)

df_cv = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
df_cv

final_model_name = "ridge"  # change to chosen best
final_pipe = make_pipeline(final_model_name)

final_pipe.fit(X, y)
print("Trained final model:", final_model_name)

# Streamlit usage:
# import joblib
# model = joblib.load("medinc_model.pkl")
# pred = model.predict(input_df)
import joblib

joblib.dump(final_pipe, "medinc_model.pkl")
print("Saved: medinc_model.pkl")

# export the file
from google.colab import files
files.download("medinc_model.pkl")
###