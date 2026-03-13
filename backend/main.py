from fastapi import FastAPI
from backend.Rag_pipeline import rag_pipeline

app = FastAPI()

@app.get("/")
def health():
    return {"status": "RAG chatbot running"}

@app.post("/chat")
def chat(question: str):
    response = rag_pipeline(question)
    return {"answer": response}