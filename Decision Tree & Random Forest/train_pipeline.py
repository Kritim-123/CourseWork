import pandas as pd
from RandomForest import RandomForest

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


X_train = train.drop(columns=["isFraud"])
y_train = train["isFraud"].astype(int)
X_test = test.drop(columns=["TransactionID"])

# One-hot encode
## Changes categorical variables to numerical
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)


# Changing to numpy arrays
X_train_np = X_train_enc.values
y_train_np = y_train.values
X_test_np = X_test_enc.values

print("line 25")

# Train Random Forest
rf = RandomForest(
    n_estimators=5,
    criterion="entropy",
    max_depth=5,
    min_sample_split=10,
    alpha=0.9,
    max_features="sqrt"
)
print("line 34")
rf.fit(X_train_np, y_train_np)
print("line 37")

# Predicting values
y_pred = rf.predict(X_test_np, threshold=0.3)

# Save
submission = pd.DataFrame({
    "TransactionID": test["TransactionID"],
    "isFraud": y_pred
})



submission.to_csv("submission.csv", index=False)
print(submission["isFraud"].value_counts())
