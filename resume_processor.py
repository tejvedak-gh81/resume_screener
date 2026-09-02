"""Agent for resume screening
RESUME SCREENING IS DONE AND STORE RESUME IN VECTOR DATABASES AND CAN BE QUERID FURTHER TO GET RELEVANT
RESUMES
"""

import os
import time
from dotenv import load_dotenv

# loaders and splitters
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# Vector Stores
from langchain_community.vectorstores import Chroma


# self query & schema tools

from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

# Gemini : Chat + embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# loading environment variables API keys
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# EMBEDDINGS: gOOGLE TEXT EMBEDDINGS 004
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embeddings-004")

# Chat llm Gemini-3.6-flash

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash",
                             temperature=0.2,
                             requests_options={"timeout": 300})

# ------------------Functions----------------------#
# Load resume in different format


def load_resume(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx") or file_path.endswith(".doc"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

# Analyze the resume using gemini (minimal change :llm now gemini)


def analyze_resume(docs, job_description):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    full_analysis = ""

# ADDED: Create a chain that automatically parses the LLM output into a clean string
    chain = llm | StrOutputParser()

    for chunk in chunks:
        prompt = f"""
Compare this resume with the job description and provide detailed analysis. Give:
1.Suitability score(0-100)
2.Skills Matched
3.Experinece Relevance
4.Education Evaluation
5.Strenghts
6.Weaknesses
7.Final Recommendation

Job Descritpion: {job_description}
Resume: {chunk.page_content}
"""
        result = chain.invoke(prompt)  # Gemini Call

        # 2. Append the extracted text string to full_analysis
        full_analysis += result + "\n\n"

        time.sleep(1)
    return full_analysis

# store text chunks in chromaDb vector store( embeddings now Google)


def store_to_vectorestore(text_chunks, persist_dictionary="chroma_store"):
    texts = [chunk.page_content for chunk in text_chunks]
    metadatas = [{"source": f"resume_chunk_{i}"} for i in range(len(texts))]

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,  # Google embeddings
        metadatas=metadatas,
        persist_directory=persist_dictionary
    )
    vectordb.persist()
    return vectordb


def run_self_query(query, persist_dictionary="chroma_store"):
    """ use selfquery retriever to interpret and fetch relevant chunks from the vector store based on user query  (llm now Gemini)
        rUN SELF QUERY FUNCTION TO FETCH RELEVANT CHUNKS FROM VECTOR STORE BASED ON USER QUERY
    """
    vectorstore = Chroma(
        persist_directory=persist_dictionary,
        embedding_function=embeddings
    )
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The source of the chunk(where the chunk is extracted from)",
            type="string"
        )
    ]

    document_content_description = "This reperesents a chunk of a resume"

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        search_type="mmr"
    )

    return retriever.get_relevant_documents(query)

# ---------------end of resume processor.py-----------------#
