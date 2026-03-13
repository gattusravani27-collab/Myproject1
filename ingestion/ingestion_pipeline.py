# Databricks notebook source
# MAGIC %pip install langchain langchain-community faiss-cpu pypdf requests

# COMMAND ----------

# DBTITLE 1,Imports
import os
import json
import hashlib
import requests
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings

# COMMAND ----------

# DBTITLE 1,Paths
PDF_FOLDER = "/Volumes/sravani/sujan/doc1"
FAISS_PATH = "/Volumes/sravani/sujan/faiss_store"

TEXTS_PATH = f"{FAISS_PATH}/texts.json"
HASH_PATH = f"{FAISS_PATH}/hashes.json"
PDF_HASH_PATH = f"{FAISS_PATH}/pdf_hashes.json"

DATABRICKS_TOKEN = dbutils.secrets.get(scope="ragscope", key="DATABRICKSTOKEN")
EMBEDDING_ENDPOINT = dbutils.secrets.get(scope="ragscope", key="EMBEDDINGENDPT")

# COMMAND ----------

# DBTITLE 1,Load Secrets
DATABRICKS_TOKEN = dbutils.secrets.get(scope="ragscope", key="DATABRICKSTOKEN")
EMBEDDING_ENDPOINT = dbutils.secrets.get(scope="ragscope", key="EMBEDDINGENDPT")



# COMMAND ----------

# DBTITLE 1,Helper Functions
class DatabricksEndpointEmbeddings(Embeddings):

    def __init__(self, endpoint, token):
        self.url = endpoint
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def _dict_to_list(self, vec_dict):
        return [float(v) for k, v in sorted(vec_dict.items(), key=lambda x: int(x[0]))]

    def embed_documents(self, texts: List[str]):

        response = requests.post(
            self.url,
            headers=self.headers,
            json={"inputs": texts},
            timeout=120
        )

        response.raise_for_status()
        preds = response.json()["predictions"]

        if isinstance(preds[0], dict):
            return [self._dict_to_list(p) for p in preds]

        return preds

    def embed_query(self, text: str):
        return self.embed_documents([text])[0]

# COMMAND ----------

# DBTITLE 1,Chunk hash function
embedding_model = DatabricksEndpointEmbeddings(
    EMBEDDING_ENDPOINT,
    DATABRICKS_TOKEN
)

# COMMAND ----------

# DBTITLE 1,Helper function
def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()

#hash pdf
def hash_file(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()  

#Batch Generator
def batch_list(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
    


# COMMAND ----------

# DBTITLE 1,Load Hash Metadata
def load_json(path):

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

hashes = load_json(HASH_PATH)
pdf_hashes = load_json(PDF_HASH_PATH)

# COMMAND ----------

# DBTITLE 1,Load PDFs (Detect Changed PDFs)
documents = []

for file in os.listdir(PDF_FOLDER):

    if not file.endswith(".pdf"):
        continue

    path = f"{PDF_FOLDER}/{file}"

    pdf_hash = hash_file(path)

    if pdf_hashes.get(file) == pdf_hash:
        print("Skipping unchanged PDF:", file)
        continue

    print("Processing PDF:", file)

    loader = PyPDFLoader(path)
    docs = loader.load()

    for d in docs:
        d.metadata["source"] = file

    documents.extend(docs)

    pdf_hashes[file] = pdf_hash

print("Documents loaded:", len(documents))

# COMMAND ----------

# DBTITLE 1,Chunk Text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))
print("Total chunks:", chunks)

# COMMAND ----------

# DBTITLE 1,Remove Duplicate Chunks (CRITICAL STEP)
new_chunks = []
new_metadata = []

for chunk in chunks:

    text = chunk.page_content.strip()

    chunk_hash = hash_text(text)

    if chunk_hash in hashes:
        continue

    hashes[chunk_hash] = True

    new_chunks.append(text)

    new_metadata.append(chunk.metadata)

print("New unique chunks:", len(new_chunks))

# COMMAND ----------

# DBTITLE 1,Load or Create FAISS
if os.path.exists(f"{FAISS_PATH}/index.faiss"):

    print("Loading existing FAISS index")

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

else:

    print("Creating new FAISS index")

    if new_chunks:
        vectorstore = FAISS.from_texts(new_chunks, embedding_model)
        new_chunks = []
    else:
        raise Exception("No index exists and no new chunks available")

# COMMAND ----------

# DBTITLE 1,Embed Only New Chunks
if new_chunks:

    print("Embedding new chunks:", len(new_chunks))

    for batch in batch_list(new_chunks, 32):

        vectorstore.add_texts(batch)

else:

    print("No new chunks to embed")

# COMMAND ----------

# DBTITLE 1,Save FAISS
vectorstore.save_local(FAISS_PATH)

print("FAISS index saved")

# COMMAND ----------

# DBTITLE 1,Save Hash Metadata
with open(HASH_PATH, "w") as f:
    json.dump(hashes, f)

with open(PDF_HASH_PATH, "w") as f:
    json.dump(pdf_hashes, f)

print("Hash metadata saved")

# COMMAND ----------

# DBTITLE 1,Final Validation
print("\nFINAL VALIDATION")

faiss_vectors = vectorstore.index.ntotal
docstore_docs = len(vectorstore.docstore._dict)

print("FAISS vectors:", faiss_vectors)
print("Docstore docs:", docstore_docs)

assert faiss_vectors == docstore_docs, \
    "FAISS and docstore mismatch — RAG will break"

print("FAISS index healthy")

# COMMAND ----------

if len(new_chunks) > 0:
    print("New documents detected. Updating FAISS and redeploying endpoint.")

    dbutils.notebook.run(
        "/Workspace/Myproject1/backend/Rag_chatbot",
        timeout_seconds=0
    )

else:
    print("No new documents. Skipping FAISS update and endpoint deployment.")
