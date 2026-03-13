# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog sravani;
# MAGIC use schema sujan;

# COMMAND ----------

# MAGIC %pip install transformers sentence-transformers mlflow torch
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

import mlflow
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as npa 

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
class HFEmbeddingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):                               # Weights are loading from the model we are using
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)          #to compute embeddings
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):          # vector embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def predict(self, context, model_input):                        # predict - to create custom model
        texts = model_input["text"].tolist()
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return pd.DataFrame(embeddings.numpy().tolist())

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="hf_embedder",
        python_model=HFEmbeddingModel(),
        input_example=pd.DataFrame({"text": ["hello world"]}),
    )
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/hf_embedder"

# COMMAND ----------

run_id = run.info.run_id
model_uri = f"runs:/{run_id}/hf_embedder"

result = mlflow.register_model(
    model_uri=model_uri,
    name="hf_embedding_model"
)

print(f"Model registered as: {result.name}, Version: {result.version}")