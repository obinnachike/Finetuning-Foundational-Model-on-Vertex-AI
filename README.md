# Finetuning Foundational Model on Vertex AI

![Vertex AI Banner](https://storage.googleapis.com/cloud-ai-blog.appspot.com/vertex-ai-banner.png)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/obinnachike/Finetuning-Foundational-Model-on-Vertex-AI/blob/main/vertex_llm_finetuning.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/_Hugging_Face-Model-yellow.svg)](https://huggingface.co/)
[![Google Cloud](https://img.shields.io/badge/Powered_by-Google_Cloud-blue?logo=google-cloud\&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Python](https://img.shields.io/badge/Built_with-Python_3.12-green?logo=python\&logoColor=white)](https://www.python.org/)

---

##  Project Overview

This project demonstrates **how to fine-tune a foundational large language model (LLM)** using **Google Cloud Vertex AI** services.

The notebook walks through the full process — from **loading BBC fulltext data** to **training a fine-tuned model** and **evaluating outputs** compared with the base model.

---

## Objective

> This lab shows how to tune a foundational model on new unseen data and you will use the following Google Cloud products:

* Vertex AI Pipelines
* Vertex AI Evaluation Services
* Vertex AI Model Registry
* Vertex AI Endpoints

### Use Case

Using Generative AI, we generate a suitable **TITLE** for a news **BODY** from *BBC Fulltext Data* (BigQuery Public Dataset: `bigquery-public-data.bbc_news.fulltext`).
We fine-tune `text-bison@002` into a new fine-tuned model **bbc-news-summary-tuned**, and compare the result with the base model.

---

##  Environment Setup

```bash
!pip install google-cloud-bigquery pandas
!pip install -U google-cloud-aiplatform
```

Libraries used include:

* `google-cloud-bigquery`
* `google-cloud-aiplatform`
* `pandas`
* `vertexai`
* `google-cloud-storage`
* `json`, `os`, `warnings`, `sys`

---

##  Authentication & Data Loading

```python
from google.colab import auth
auth.authenticate_user()

from google.cloud import bigquery
project_id = "phonic-hydra-474723-k1"
client = bigquery.Client(project=project_id)

query = """
SELECT title, body
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE title IS NOT NULL AND body IS NOT NULL
LIMIT 100
"""
df = client.query(query).to_dataframe()
df.head()
```

---

## Data Preprocessing

The dataset is converted into **JSONL** format for training.

```python
import json
with open("stackoverflow_gemini_correct.jsonl", "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        input_text = str(row.get("body", "")).strip()
        output_text = str(row.get("title", "")).strip()

        if not input_text or not output_text:
            continue

        example = {
            "contents": [
                {"role": "user", "parts": [{"text": input_text}]},
                {"role": "model", "parts": [{"text": output_text}]}
            ]
        }
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
```

---

## Uploading to Google Cloud Storage

```python
from google.cloud import storage

BUCKET_NAME = "my-ver-bucket"
DESTINATION_BLOB_NAME = "TRAININGS1.jsonl"
SOURCE_FILE_NAME = "stackoverflow_gemini_correct.jsonl"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(DESTINATION_BLOB_NAME)
blob.upload_from_filename(SOURCE_FILE_NAME)

print(f"Uploaded to gs://{BUCKET_NAME}/{DESTINATION_BLOB_NAME}")
```

Output:

```
Uploaded to gs://my-ver-bucket/TRAININGS1.jsonl
```

---

## Model Fine-Tuning on Vertex AI

```python
from vertexai.tuning import sft

sft_tuning_job = sft.train(
    source_model="gemini-2.0-flash-001",
    train_dataset="gs://my-ver-bucket/TRAININGS1.jsonl",
)
```

Monitor the tuning process via:

```
https://console.cloud.google.com/vertex-ai/generative/language/locations/us-central1/tuning
```

Polling for completion:

```python
import time
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()
```

Outputs:

```
projects/88233143849/locations/us-central1/models/6484720569018220544@1
projects/88233143849/locations/us-central1/endpoints/881292654522925056
Job state: 4
```

---

##  Predict with the Fine-Tuned Model

```python
from vertexai.generative_models import GenerativeModel

content = "Summarize this text to generate a title: \n Ever noticed how plane seats appear to be getting smaller..."
sft_tuning_job = sft.SupervisedTuningJob("projects/.../tuningJobs/7145409907783630848")
tuned_model = GenerativeModel(sft_tuning_job.tuned_model_endpoint_name)
response = tuned_model.generate_content(content)

print(response.text)
```

Output:

```
Passenger safety on planes
```

---

##  Deleting a Tuned Model

```python
from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=REGION)

models = aiplatform.Model.list()
model = aiplatform.Model(MODEL_ID)
model.delete()
```

---

##  Key Features

✅ Uses **Vertex AI Tuning API (SFT)** for supervised model fine-tuning
✅ Integrates **BigQuery Public Datasets**
✅ Automatically uploads data to **Google Cloud Storage**
✅ Deploys and queries tuned model endpoints
✅ Supports **Gemini**, **Bison**, and other foundation models

---

## Tech Stack

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Google Vertex AI | Model training, tuning, and deployment |
| BigQuery         | Data sourcing                          |
| GCS              | Dataset storage                        |
| Python (Colab)   | Development environment                |
| Hugging Face     | Model hosting (optional)               |

---

##  References

* [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
* [BigQuery Public Datasets](https://cloud.google.com/bigquery/public-data)
* [Hugging Face Model Hub](https://huggingface.co/models)

---

##  Author

**Chiejina Chike Obinna**
*Researcher • AI/ML Engineer • Educator*
🔗 [GitHub Profile](https://github.com/obinnachike) | [LinkedIn](https://linkedin.com/in/chiejina-chike-obinna)

---
