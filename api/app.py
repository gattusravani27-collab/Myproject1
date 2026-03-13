from fastapi import FastAPI
import requests
import os

app = FastAPI()

ENDPOINT_URL = "https://adb-4495317092823078.18.azuredatabricks.net/serving-endpoints/rag_chatbot4/invocations"

TOKEN = os.getenv("DATABRICKS_TOKEN")

if not TOKEN:
    raise ValueError("DATABRICKS_TOKEN environment variable is not set.")

@app.get("/")
def home():
    return {"message": "RAG chatbot API running"}

@app.post("/chat")
def chat(question: str, session_id: str = "user1"):

    if not question.strip():
        return {"error": "Question cannot be empty"}

    if len(question) > 500:
        return {"error": "Question too long"}

    payload = {
        "dataframe_split": {
            "columns": ["question", "session_id"],
            "data": [[question, session_id]]
        }
    }

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            ENDPOINT_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
    except Exception as e:
        return {
            "error": "Failed to connect to Databricks endpoint",
            "details": str(e)
        }

    if response.status_code != 200:
        return {
            "error": "Databricks endpoint error",
            "status_code": response.status_code,
            "message": response.text
        }

    return response.json()