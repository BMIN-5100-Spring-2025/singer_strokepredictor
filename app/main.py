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
import boto3
import json
import base64
import logging
import sys

# Environment setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
s3 = boto3.client("s3")


def download_files_from_s3(bucket, prefix, input_directory):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        return
    for obj in response["Contents"]:
        key = obj["Key"]
        if not key.endswith("/"):  # Skip directory keys
            local_filename = os.path.join(input_directory, os.path.basename(key))
            s3.download_file(bucket, key, local_filename)

       
def upload_files_to_s3(bucket, prefix, output_directory):
    for filename in os.listdir(output_directory):
        local_path = os.path.join(output_directory, filename)
        s3_key = f"{prefix}output/{filename}" if prefix else f"output/{filename}"
        s3.upload_file(local_path, bucket, s3_key)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(" Time taken: %i minutes and %s seconds." % (tmin, round(tsec, 2)))

# Load environment variables
session_id = os.getenv('SESSION_ID')
logger.info(f"session: {session_id}")


# parameters = os.getenv('PARAMETERS')
# 
# if parameters and parameters.strip() != "":
#     logger.info(f"Raw PARAMETERS env var: {parameters}")
#     parameters = json.loads(base64.b64decode(parameters).decode('utf-8'))
# else:
#     parameters = None
# logger.info(f"parameters: {parameters}")

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
RUN_ENV = os.getenv("RUN_ENV", "local")

base_directory = os.path.dirname(os.path.dirname(__file__))
input_directory = os.getenv('INPUT_DIR', os.path.join(base_directory, 'data/input/'))
output_directory = os.getenv('OUTPUT_DIR', os.path.join(base_directory, 'data/output/'))

prefix = f"{session_id}/" if session_id else ""


if RUN_ENV == "fargate":
    input_directory = "/tmp/input/"
    output_directory = "/tmp/output/"
    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    logger.info(f"Downloading files from s3://{S3_BUCKET}/{prefix}input/")
    download_files_from_s3(S3_BUCKET, f"{prefix}input/", input_directory)

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
    feature_importance = pd.DataFrame({"Feature": features, "Coefficient": coefficients})
    feature_importance["AbsCoefficient"] = feature_importance["Coefficient"].abs()
    feature_importance = feature_importance.sort_values(by="AbsCoefficient", ascending=False)
    return feature_importance.head(top_n)

if __name__ == "__main__":
    logger.info("Initiating LassoRegression")

    folds = 10
    repeats = 10
    seeds = [6772, 6659, 7622]

    start_time = timer(None)
    print("\n# Reading and Processing Data")
    X_train, y, train_ids, features, X_test, test_ids = load_data()

    all_cols = features
    cols_cat = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

    print("\n Encoding categorical variables ...")
    ce = LeaveOneOutEncoder(cols=cols_cat, random_state=2022, sigma=0.05, verbose=1)
    X_train = ce.fit_transform(X_train, y)
    X_test = ce.transform(X_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "StandardScaler_Lasso-01-v1.joblib")

    rkf_grid = list(RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seeds[0]).split(X_train, y))

    start_time = timer(None)
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

    top_risks = get_top_risk_factors(features, model_llcv.coef_, top_n=3)
    os.makedirs(output_directory, exist_ok=True)
    top_risks.to_csv(os.path.join(output_directory, "top_risk_factors.csv"), index=False, float_format="%.6f")

    RMSE_nocv = np.sqrt(mean_squared_error(y, model_llcv.predict(X_train)))
    AUC_nocv = roc_auc_score(y, model_llcv.predict(X_train))
    print(f"RMSE: {RMSE_nocv:.6f}, AUC: {AUC_nocv:.6f}")

    pred_path = os.path.join(output_directory, "stroke_predictions.csv")
    summary_path = os.path.join(output_directory, "prediction_summary.txt")

    result = pd.DataFrame({
        "id": test_ids,
        "stroke_probability": special.expit(model_llcv.predict(X_test))
    })
    result.to_csv(pred_path, index=False, float_format="%.6f")

    with open(summary_path, "w") as f:
        f.write(f"Predicted strokes: {(result['stroke_probability'] >= 0.5).sum()}")

    if RUN_ENV == "fargate":
        logger.info(f"Uploading files back to s3://{S3_BUCKET}/{prefix}output/")
        upload_files_to_s3(S3_BUCKET, prefix, output_directory)

    timer(start_time)
