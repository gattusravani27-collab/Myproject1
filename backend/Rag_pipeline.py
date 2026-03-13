import os
import requests
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models import LLM
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import RedisChatMessageHistory



from dotenv import load_dotenv
import os

load_dotenv()

FOUNDRY_URL = os.getenv("FOUNDRY_URL")
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY")

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_URL = os.getenv("REDIS_URL")




from typing import List, Union
from langchain_core.embeddings import Embeddings
import requests


class DatabricksEndpointEmbeddings(Embeddings):
    """
    FINAL production-safe embedding wrapper for:
    - Databricks Serving Endpoint (HF/MLflow)
    - LangChain RAG chains
    - FAISS compatibility
    - Handles dict inputs from Runnable chains
    """

    def __init__(self, host: str, token: str, endpoint: str, timeout: int = 300):
        self.host = host
        self.token = token
        self.endpoint = endpoint
        self.timeout = timeout

        # USE FULL ENDPOINT DIRECTLY
        self.url = endpoint

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def _convert_dict_to_list(self, vec_dict):
        """Convert dict embeddings {"0": v, "1": v} → [v, v]"""
        return [float(v) for k, v in sorted(vec_dict.items(), key=lambda x: int(x[0]))]

    def _ensure_strings(self, texts: List[Union[str, dict]]) -> List[str]:
        """
        CRITICAL FIX:
        RAG chains sometimes pass dict like {"question": "..."}
        We must extract only the string.
        """
        clean_texts = []
        for t in texts:
            if isinstance(t, str):
                clean_texts.append(t)
            elif isinstance(t, dict):
                # Extract the actual question/text field
                if "question" in t:
                    clean_texts.append(str(t["question"]))
                elif "text" in t:
                    clean_texts.append(str(t["text"]))
                else:
                    clean_texts.append(str(t))
            else:
                clean_texts.append(str(t))
        return clean_texts

    def _call_endpoint(self, texts: List[Union[str, dict]]) -> List[List[float]]:
        # 🔥 FIX: sanitize inputs from RAG chain
        texts = self._ensure_strings(texts)

        payload = {
            "inputs": texts  # HF endpoint expects list[str]
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code != 200:
            print("❌ Status Code:", response.status_code)
            print("❌ Response:", response.text)

        response.raise_for_status()
        result = response.json()

        # Handle MLflow / HF outputs
        if isinstance(result, dict) and "predictions" in result:
            preds = result["predictions"]

            if isinstance(preds[0], dict):
                return [self._convert_dict_to_list(p) for p in preds]

            if isinstance(preds[0], list):
                return preds

        if isinstance(result, list):
            return result

        raise ValueError(f"Unexpected embedding response format: {result}")

    # Required by LangChain
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_endpoint(texts)

    # Required by LangChain (used during retrieval)
    def embed_query(self, text: str) -> List[float]:
        return self._call_endpoint([text])[0]




embedding_model = DatabricksEndpointEmbeddings(
    host=DATABRICKS_HOST,
    token=DATABRICKS_TOKEN,
    endpoint=EMBEDDING_ENDPOINT
)



FAISS_PATH = "/Volumes/sravani/sujan/faiss_store"

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
print("FAISS loaded successfully")



from langchain_core.language_models import LLM
from langchain_core.messages import BaseMessage
from pydantic import Field
from typing import Optional, List, Union
import requests


class FoundryChatLLM(LLM):
    url: str = Field(...)
    api_key: str = Field(...)
    temperature: float = 0.0
    timeout: int = 60

    @property
    def _llm_type(self) -> str:
        return "azure_foundry_chat"

    def _convert_to_text(self, prompt: Union[str, List[BaseMessage]]) -> str:
        """
        Convert LangChain messages (HumanMessage, AIMessage, ChatPromptValue)
        into plain string for Foundry API.
        """
        # Case 1: Already string
        if isinstance(prompt, str):
            return prompt

        # Case 2: List of messages (memory chain case)
        if isinstance(prompt, list):
            return "\n".join([msg.content for msg in prompt if hasattr(msg, "content")])

        # Case 3: ChatPromptValue object
        if hasattr(prompt, "to_messages"):
            messages = prompt.to_messages()
            return "\n".join([m.content for m in messages])

        # Fallback
        return str(prompt)

    def _call(self, prompt, stop: Optional[List[str]] = None) -> str:
        # 🔥 CRITICAL FIX: convert HumanMessage → text
        formatted_prompt = self._convert_to_text(prompt)

        payload = {
            "messages": [
                {"role": "user", "content": formatted_prompt}
            ],
            "temperature": self.temperature
        }

        response = requests.post(
            self.url,
            headers={
                "api-key": self.api_key,  # Correct for Azure Foundry
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]




llm = FoundryChatLLM(
    url=FOUNDRY_URL,
    api_key=FOUNDRY_API_KEY
)

# Quick sanity test
print(llm.invoke("Reply in one short sentence."))



from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful conversational AI assistant.
Use the retrieved context to answer the question.
Also use the chat history to maintain conversation continuity.
If the answer is not in context, say "I don't know"."""),
    
    ("placeholder", "{history}"),  # 🔥 THIS ENABLES MEMORY
    
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)




from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)



from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL
    )

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,                 # IMPORTANT: not prompt | llm
    get_redis_history,
    input_messages_key="question",
    history_messages_key="history",
)



response = rag_with_memory.invoke(
    {"question": "Tell about Indias G20 presidency"},
    config={"configurable": {"session_id": "user1"}}
)

print(response)



response = rag_with_memory.invoke(
    {"question": "what was my last question? "},
    config={"configurable": {"session_id": "user1"}}
)

print(response)


response = rag_with_memory.invoke(
    {"question": "what was my previous question? "},
    config={"configurable": {"session_id": "user1"}}
)

print(response)
