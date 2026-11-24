import os
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load API key
load_dotenv()

api_key = os.getenv("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")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

# 2. Load DBMS PDF
pdf_path = os.path.join(
    os.path.dirname(__file__),
    "DBMS_PYQ_Question_Bank_100_Pages (1).pdf"
)

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

print("Loading DBMS PDF...")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} pages.")

# 3. Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80
)

chunks = text_splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks.")

# 4. Create embeddings
print("Creating embeddings...")

embeddings = FastEmbedEmbeddings()

# 5. Create Qdrant vector store
client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="dbms_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="dbms_collection",
    embedding=embeddings
)

vectorstore.add_documents(chunks)

print("DBMS PDF indexing completed!")

# 6. Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.2,
    max_tokens=2500
)

# 7. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 8. Create DBMS prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert DBMS tutor. "
        "Answer questions using ONLY the provided PDF context. "
        "Explain answers in simple student-friendly language. "
        "If the answer is not present in the context, "
        "clearly say that the information is not available "
        "in the provided PDF.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}")
])

rag_chain = prompt | llm | StrOutputParser()

# 9. RAG Question Answering
def query_rag(question):

    print("\n" + "=" * 60)
    print("Question:", question)
    print("=" * 60)

    retrieved_docs = retriever.invoke(question)

    print("\nRetrieved PDF sections:")

    for i, doc in enumerate(retrieved_docs, start=1):
        page = doc.metadata.get("page", "Unknown")
        print(f"{i}. Page {page + 1}")

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    print("\nGenerating DBMS answer...")

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAnswer:")
    print(response)

    print("=" * 60)

    return response


# 10. Test chatbot
if __name__ == "__main__":

    test_question = "What is DBMS? Explain its advantages."

    query_rag(test_question)
    
