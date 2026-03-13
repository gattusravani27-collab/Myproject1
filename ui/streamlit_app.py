import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.title("RAG Chatbot")

question = st.text_input("Ask a question")

if st.button("Send"):

    if not question.strip():
        st.warning("Please enter a question")

    else:

        try:
            response = requests.post(
                API_URL,
                params={
                    "question": question,
                    "session_id": "user1"
                },
                timeout=30
            )

            result = response.json()

        except Exception as e:
            st.error("Failed to connect to API")
            st.write(e)
            st.stop()

        if "predictions" in result:

            answer = result["predictions"][0]["answer"]
            sources = result["predictions"][0].get("sources", "")

            st.write("### Answer")
            st.write(answer)

            st.write("### Sources")
            st.write(sources)

        else:
            st.error("Unexpected API response")
            st.write(result)