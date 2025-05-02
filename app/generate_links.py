from flask import Flask, request, jsonify
import boto3
import os

app = Flask(__name__)
s3 = boto3.client('s3')

# Set your bucket name
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "singer-strokepredictor")

@app.route("/generate-download-links", methods=["POST"])
def generate_download_links():
    data = request.json
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    filenames = ["stroke_predictions.csv", "prediction_summary.txt", "top_risk_factors.csv"]
    links = {}

    for name in filenames:
        key = f"{session_id}/output/{name}"
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': S3_BUCKET, 'Key': key},
            ExpiresIn=3600  # Link valid for 1 hour
        )
        links[name] = url

    return jsonify(links)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
