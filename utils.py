from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
import os
import time


# ✅ Extract text from a PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Ensure it doesn't break on empty pages
    return text


# ✅ Convert uploaded PDFs into documents
def create_docs(user_pdf_list, unique_id):
    docs = []
    for pdf_file in user_pdf_list:
        chunks = get_pdf_text(pdf_file)

        # Add metadata for retrieval
        docs.append(Document(
            page_content=chunks,
            metadata={
                "name": pdf_file.name,
                "unique_id": unique_id
            }
        ))

    return docs


# ✅ Create OpenAI embeddings instance
def create_embeddings_load_data():
    return OpenAIEmbeddings(model="text-embedding-ada-002")


# ✅ Initialize and connect to Pinecone
def init_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)


# ✅ Push documents to Pinecone
def push_to_pinecone(pinecone_index_name, embeddings, docs):
    init_pinecone()
    Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)


# ✅ Connect to an existing Pinecone index
def get_pinecone_index():
    init_pinecone()
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    return pinecone.Index(pinecone_index_name)


# ✅ Fetch similar resumes from Pinecone
def similar_docs(query, k):
    embeddings = create_embeddings_load_data()
    query_embedding = embeddings.embed_query(query)

    index = get_pinecone_index()
    results = index.query(query_embedding, top_k=int(k), include_metadata=True)

    return results


# ✅ Summarize document using OpenAI
def get_summary(current_doc):
    from langchain.llms import OpenAI

    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary
