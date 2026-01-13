# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

# %%
train = pd.read_csv(
    "/home/wwj/src/Deep-Learning-Projects/Predicting_Student_Test_Scores/data/train.csv"
)
test = pd.read_csv(
    "/home/wwj/src/Deep-Learning-Projects/Predicting_Student_Test_Scores/data/test.csv"
)

# %%
train.head()

# %%
ID_COL = "id"
TARGET = "exam_score"


# %%
X = train.drop(columns = [ID_COL, TARGET])
y = train[TARGET]
X_test = test.drop(columns = [ID_COL])

# %%
cat_cols = X.select_dtypes(include = ['object']).columns.tolist()
num_cols = X.select_dtypes(exclude =['object']).columns.tolist()

# %%
print("Categorical: ", cat_cols)
print("Numerical: ", num_cols)

# %%
for col in X.columns:
    X[col + "_is_null"] = X[col].isnull().astype(int)
    X_test[col+"_is_null"] = X_test[col].isnull().astype(int)

# %%
X.head()

# %%
for col in cat_cols:
    freq = X[col].value_counts()
    X[col + "_freq"] = X[col].map(freq)
    X_test[col+"_freq"] = X_test[col].map(freq)

# %%
X.head()

# %%
X = X.drop(columns=cat_cols)
X_test = X_test.drop(columns=cat_cols)

# %%
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1,
    # "device": "gpu"
}

# %%
kf = KFold(n_splits=3, shuffle=True, random_state=42)

oof = np.zeros(len(X))
pred_test = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=5000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(200)]
    )
    
    oof[val_idx] = model.predict(X_val)
    pred_test += model.predict(X_test) / kf.n_splits


# %%
rmse = mean_squared_error(y, oof)
print("CV RMSE:" , rmse)


# %%
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: pred_test
})

submission.to_csv("submission.csv", index=False)


# %%
