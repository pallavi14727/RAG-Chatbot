RAG-Based PDF Question Answering Chatbot

1. Project Overview

This project is an AI-powered PDF Question Answering Chatbot built using Retrieval-Augmented Generation (RAG).

It allows users to ask questions about PDF documents and receive relevant answers based on the information available in the uploaded PDF.

2. Objectives

- To extract information from PDF documents.
- To answer questions using Artificial Intelligence.
- To retrieve relevant information using vector search.
- To provide accurate and easy-to-understand answers.

3. Technologies Used

- Python
- LangChain
- Groq API
- Qdrant Vector Database
- FastEmbed Embeddings
- PyPDF
- Retrieval-Augmented Generation (RAG)

4. Working Process

1. Load the PDF document.
2. Extract text from the PDF.
3. Divide text into smaller chunks.
4. Generate embeddings for the text.
5. Store embeddings in Qdrant.
6. Retrieve relevant information based on the user's question.
7. Generate an answer using the Groq LLM.

5. Project Structure

RAG-Chatbot/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── DBMS_PYQ_Question_Bank_100_Pages (1).pdf

6. Installation

Clone the repository:

git clone https://github.com/pallavi14727/RAG-Chatbot.git

Open the project folder:

cd RAG-Chatbot

Install the required packages:

pip install -r requirements.txt

7. API Configuration

Create a ".env" file in the project folder.

Add your Groq API key:

GROQ_API_KEY=your_api_key

Do not upload the ".env" file to GitHub.

8. Run the Project

python main.py

9. Example Question

Question: What is DBMS? Explain its advantages.

The chatbot retrieves relevant information from the PDF and generates an answer using the language model.

10. Future Scope

- Support multiple PDF documents.
- Develop a web-based chatbot interface.
- Add chat history.
- Improve answer accuracy.
- Support different document formats.

11. Author

Pallavi Singh

B.Tech CSE (CC + AI)

12. Conclusion

This project demonstrates how Retrieval-Augmented Generation can be used to build an intelligent PDF-based Question Answering Chatbot using Python, LangChain, Groq and Qdrant.
