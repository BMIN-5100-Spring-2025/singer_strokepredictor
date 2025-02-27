from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import joblib
import matplotlib.pyplot as plt
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.preprocessing import StandardScaler
from scipy import special
import os


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(" Time taken: %i minutes and %s seconds." % (tmin, round(tsec, 2)))



base_directory = os.path.dirname(os.path.dirname(__file__))
input_directory = os.getenv('INPUT_DIR', os.path.join(base_directory, 'data/input/'))
output_directory = os.getenv('OUTPUT_DIR', os.path.join(base_directory, 'data/output/'))


DATA_TRAIN_PATH = os.path.join(input_directory, "train.csv")
DATA_TEST_PATH = os.path.join(input_directory, "test.csv")


def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train_loader = pd.read_csv(path_train)
    train = train_loader.drop(["stroke", "id"], axis=1)
    features = train.columns.tolist()
    print("\n Train dataset shape:", train.shape)
    train_labels = train_loader["stroke"].values
    train_ids = train_loader["id"].values

    test_loader = pd.read_csv(path_test)
    test = test_loader[features]
    print(" Whole test dataset shape:", test.shape)
    test_ids = test_loader["id"].values

    return train, train_labels, train_ids, features, test, test_ids

def get_top_risk_factors(features, coefficients, top_n=3):
    """
    Outputs the top N risk factors based on Lasso coefficients.
    """
    feature_importance = pd.DataFrame({"Feature": features, "Coefficient": coefficients})
    feature_importance["AbsCoefficient"] = feature_importance["Coefficient"].abs()
    feature_importance = feature_importance.sort_values(by="AbsCoefficient", ascending=False)
    return feature_importance.head(top_n)

if __name__ == "__main__":

    # We will do repeated K-fold cross-validation
    folds = 10
    repeats = 10
    # For historic reasons I use one of these 3 seeds
    seeds = [6772, 6659, 7622]

    # Load data set and target values
    start_time = timer(None)
    print("\n# Reading and Processing Data")
    X_train, y, train_ids, features, X_test, test_ids = load_data()

    all_cols = features
    cols_cat = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    num_features = [col for col in all_cols if col not in cols_cat]

### PROCESSING DATA

print("\n Encoding categorical variables ...")
# sigma=0.05 injects a bit of noise so we don't overfit
ce = LeaveOneOutEncoder(cols=cols_cat, random_state=2022, sigma=0.05, verbose=1)
X_train = ce.fit_transform(X_train, y)
X_test = ce.transform(X_test)
print("\n Train Set Matrix Dimensions: %d x %d" % (X_train.shape[0], X_train.shape[1]))
print(" Test Set Matrix Dimensions: %d x %d" % (X_test.shape[0], X_test.shape[1]))

print(
    " Potential NaN or Inf values in train data:  ",
    np.isnan(X_train[features].values).any(),
    "  ",
    np.isinf(X_train[features].values).any(),
)
print(
    " Potential NaN or Inf values in test data:  ",
    np.isnan(X_test[features].values).any(),
    "  ",
    np.isinf(X_test[features].values).any(),
)
timer(start_time)

scaler = StandardScaler()
scaler.fit(X_train[features].values)
X_train[features] = scaler.transform(X_train[features].values)
joblib.dump(scaler, "StandardScaler_Lasso-01-v1.joblib")
# scaler = joblib.load('StandardScaler_Lasso-01-v1.joblib')
X_test[features] = scaler.transform(X_test[features].values)

### CV SET UP

rkf_grid = list(
    RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seeds[0]).split(
        X_train, y
    )
)
start_time = timer(None)
# Run Lasso cross-validation to determine coefficient alpha and the intercept
# Doing repeated CV to test many different folds and avoid overfitting
print("\n Running Lasso:")
model_llcv = LassoCV(
    precompute="auto",
    fit_intercept=True,
    max_iter=1000,
    verbose=False,
    eps=1e-04,
    cv=rkf_grid,
    n_alphas=1000,
    n_jobs=8,
)
model_llcv.fit(X_train, y)
joblib.dump(model_llcv, "Stroke_Lasso-01-v1.joblib")
#    model_llcv = joblib.load('Stroke_Lasso-01-v1.joblib')
print(" Best alpha value: %.10f" % model_llcv.alpha_)
print(" Intercept: %.10f" % model_llcv.intercept_)
print(" LassoCV score: %.10f" % model_llcv.score(X_train, y))
# Output primary risk factors
print("\n### PRIMARY RISK FACTORS ###")
top_risks = get_top_risk_factors(features, model_llcv.coef_, top_n=3)
print(top_risks)

# Save the top risk factors as a CSV file for the web app
os.makedirs(output_directory, exist_ok=True)
top_risks.to_csv(os.path.join(output_directory, "top_risk_factors.csv"), index=False, float_format="%.6f")
### CHECKING SCORES
RMSE_nocv = np.sqrt(mean_squared_error(y, model_llcv.predict(X_train)))
AUC_nocv = roc_auc_score(y, model_llcv.predict(X_train))
print("\n Non cross-validated LassoCV RMSE: %.6f" % RMSE_nocv)
print(" Non cross-validated AUC: %.6f" % AUC_nocv)


### RUN ON MORE FOLDS

cv_sum = 0
cv_sum_auc = 0
pred = []
fpred = []
avreal = y
avpred = np.zeros(X_train.shape[0])
avpred_count = np.zeros(X_train.shape[0])
idpred = train_ids

train_time = timer(None)
repeats = 10
rkf_grid = list(
    RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seeds[0]).split(
        X_train, y
    )
)

for i, (train_index, val_index) in enumerate(rkf_grid):
    fold_time = timer(None)
    print("\n Fold %02d" % (i + 1))
    Xtrain, Xval = X_train.loc[train_index], X_train.loc[val_index]
    ytrain, yval = y[train_index], y[val_index]

    scores_val = model_llcv.predict(Xval)
    corr_val = np.sqrt(mean_squared_error(yval, scores_val))
    corr_val_auc = roc_auc_score(yval, scores_val)
    print(" Fold %02d RMSE: %.6f" % ((i + 1), corr_val))
    print(" Fold %02d AUC: %.6f" % ((i + 1), corr_val_auc))
    y_pred = model_llcv.predict(X_test)
    timer(fold_time)

    avpred[val_index] += scores_val
    avpred_count[val_index] += 1
    if i > 0:
        fpred = pred + y_pred
    else:
        fpred = y_pred
    pred = fpred
    cv_sum = cv_sum + corr_val
    cv_sum_auc = cv_sum_auc + corr_val_auc

### TEST SET PREDICTIONS
# Make predictions for the test dataset
test_predictions = special.expit(model_llcv.predict(X_test))
result = pd.DataFrame({"id": test_ids, "stroke_probability": test_predictions})

# Calculate predicted number of stroke cases
expected_strokes = (test_predictions >= 0.5).sum()  # Assuming a threshold of 0.5
summary_text = f"We predict approximately {expected_strokes}/{len(test_predictions)} people in the test dataset may experience a stroke."
print(f"\n### PREDICTED NUMBER OF STROKES ###")
print(summary_text)

# Save predictions to a CSV file for the web app
result.to_csv(os.path.join(output_directory, "stroke_predictions.csv"), index=False, float_format="%.6f")
with open(os.path.join(output_directory, "prediction_summary.txt"), "w") as f:
        f.write(summary_text)

print("\Test prediction and prediction summary saved to 'prediction_summary.txt'.")


timer(train_time)

cv_score = cv_sum / (folds * repeats)
cv_score_auc = cv_sum_auc / (folds * repeats)
avpred = avpred / avpred_count
oof_corr = np.sqrt(mean_squared_error(avreal, avpred))
oof_corr_auc = roc_auc_score(avreal, avpred)
print("\n Average RMSE: %.6f" % cv_score)
print(" Out-of-fold RMSE: %.6f" % oof_corr)
print(" Average AUC: %.6f" % cv_score_auc)
print(" Out-of-fold AUC: %.6f" % oof_corr_auc)
score = str(round(oof_corr_auc, 6)).replace(".", "")
mpred = pred / (folds * repeats)

now = datetime.now()
# Not really necessary, but applying sigmoid function here so all predictions map to [0,1] range
oof_result = pd.DataFrame(avreal, columns=["stroke"])
oof_result["prediction"] = special.expit(avpred)
oof_result["id"] = idpred
oof_result = oof_result[["id", "stroke", "prediction"]]
sub_file = (
    "train_OOF_10_by_10x-Lasso_"
    + score
    + "_"
    + str(now.strftime("%Y-%m-%d-%H-%M"))
    + ".csv"
)


# Not really necessary, but applying sigmoid function here so all predictions map to [0,1] range
result = pd.DataFrame(special.expit(mpred), columns=["stroke"])
result["id"] = test_ids
result = result[["id", "stroke"]]
print("\n First 10 lines of your prediction:")
print(result.head(10))

