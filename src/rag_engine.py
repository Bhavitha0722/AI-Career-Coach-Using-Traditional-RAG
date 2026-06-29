import os
#import shutil
#from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

#DB_dir="career_coach_chroma_db"

#-----------------DATA INJECTION PIPELINE-----------------

#Step-1:---model loaded-----
def get_llm(model: str="llama-3.1-8b-instant",temperature:float=0):
    try:
        api_key = os.getenv("GROQ_API_KEY")
    except:
        print("Model not loaded")

    return ChatGroq(
    model=model,
    temperature=temperature,
    api_key=os.getenv("GROQ_API_KEY")
)

#step-2:----loading Embeddings--
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

#step-3:---loading text files and creating Document objects-----
def load_text_file(file_path:str,source_name:str,doc_type:str) -> List[Document]: # this is to load the pdf 
    path=Path(file_path) #loading the file from the path
    text=path.read_text(encoding="utf-8", errors="ignore")

    return[Document(page_content=text, metadata={'source':source_name,
                                                 "doc_type":doc_type})]

# split the documents into chunks ,the file reads the files.utils and convert into document format for embeddings, splittings 
def create_documents(resume_text:str,jd_text:str) -> List[Document]: # this is to load the text format of resume and job description and create Document objects
    return[Document(page_content=resume_text, 
                    metadata={'source':"uploaded_resume",
                              "doc_type":"resume"}),
            Document(page_content=jd_text, 
                    metadata={'source':"uploaded_job_description",
                              "doc_type":"job_description"})]


def split_documents(docs: List[Document], chunk_size:int=1000, 
                    chunk_overlap:int=150)->List[Document]:
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap,
                                            
    )
    return splitter.split_documents(docs)

#step-4:---create Embeddings and build vectorstore-----
def build_vectorstore(chunks: List[Document]):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="career_coach_rag"
    )
    return vectorstore


#------------------DATA RETRIEVAL PIPELINE-----------------

#---context which means retrieving the relevent information based on the user query --------
def retrieve_context(vectorstore,query:str,k:int=3):
    retriever=vectorstore.as_retriever(search_kwargs={'k':k})
    docs=retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs

def run_career_coach(vectorstore,resume_text:str,jd_text:str,question:str):
    llm=get_llm()
    retrieval_query=f"""Resume content and JD content relevant to the 
    coaching question"""
    context,source_docs=retrieve_context(vectorstore,retrieval_query,k=3)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert AI Career Coach for students, freshers and working professionals.
    Use ONLY the given context from the resume and job description.
    Do not invent skills, experience or job requirements.

CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear, practical answer with these sections when relevant:
1. Current Match Summary
2. Strengths
3. Missing Skills / Gaps
4. Recommended Improvements
5. Suggested Projects
6. Interview Preparation Tips

Keep the answer simple, actionable and beginner-friendly.
""")

    chain=prompt | llm | StrOutputParser()
    answer=chain.invoke({"context":context, "question":question})
    return answer,source_docs

def generate_complete_report(vectorstore, resume_text: str, jd_text: str):
    question = """
    Analyze this resume against this job description. Provide ATS-style score, skill match, missing skills,
    resume improvement suggestions, project suggestions, and interview questions.
    """
    return run_career_coach(vectorstore, resume_text, jd_text, question)

