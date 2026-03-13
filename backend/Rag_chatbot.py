# Databricks notebook source
# MAGIC %pip install \
# MAGIC langchain \
# MAGIC langchain-community \
# MAGIC langchain-huggingface \
# MAGIC sentence-transformers \
# MAGIC faiss-cpu \
# MAGIC mlflow

# COMMAND ----------

# MAGIC %pip install langchain-openai openai

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model.save("/tmp/embedding_model")

# COMMAND ----------

import pandas as pd
import mlflow.pyfunc


class RAGChatbotModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):

        import os
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        # Load API key from Databricks secrets (only in notebook)
        try:
            import dbutils
            os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(
                scope="rag-secrets",
                key="llmkey"
            )
        except:
            pass

        # Load artifacts
        faiss_path = context.artifacts["faiss_store"]
        embedding_path = context.artifacts["embedding_model"]

        # Embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_path
        )

        # Load FAISS
        self.vectorstore = FAISS.load_local(
            faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 15}
        )

        # Initialize LLM only if API key exists
        import os
        if os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
            )
        else:
            self.llm = None


    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:

        question = model_input["question"].iloc[0]

        docs = self.retriever.invoke(question)

        context_text = "\n\n".join([doc.page_content for doc in docs[:3]])

        sources = []
        for doc in docs:
            if "source" in doc.metadata:
                sources.append(doc.metadata["source"])

        prompt = f"""
You are a document question answering assistant.

Answer ONLY using the provided context.

If the answer is not present in the context say:
"The information is not available in the provided documents."

Context:
{context_text}

Question:
{question}

Answer:
"""

        # If LLM not initialized (during MLflow logging)
        if self.llm is None:
            answer = "LLM not initialized during model logging."
        else:
            response = self.llm.invoke(prompt)
            answer = response.content

        return pd.DataFrame({
            "answer": [answer],
            "sources": [", ".join(set(sources))]
        })

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd
import mlflow

input_example = pd.DataFrame({
    "question": ["What is India's G20 presidency?"],
    "session_id": ["user1"]
})

input_schema = Schema([
    ColSpec("string", "question"),
    ColSpec("string", "session_id")
])

output_schema = Schema([
    ColSpec("string", "answer"),
    ColSpec("string", "sources")
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run():

    mlflow.pyfunc.log_model(
        name="rag_chatbot_model",
        python_model=RAGChatbotModel(),

        artifacts={
            "faiss_store": "/Volumes/sravani/sujan/faiss_store",
            "embedding_model": "/tmp/embedding_model"
        },

        signature=signature,  # remove input_example to avoid validation issues

        pip_requirements=[
            "mlflow",
            "faiss-cpu",
            "numpy",
            "sentence-transformers",
            "langchain",
            "langchain-community",
            "langchain-huggingface",
            "langchain-openai",
            "openai"
        ],

        registered_model_name="rag_chatbot_model"
    )