from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from PyPDF2 import PdfReader

from dotenv import load_dotenv




app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google API setup
load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please set it as an environment variable.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file (UploadFile): The uploaded PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    try:
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")



def create_quiz_from_text(text: str) -> dict:
    """
    Generate a quiz based on the provided text.

    Args:
        text (str): The input text to generate the quiz from.

    Returns:
        dict: Quiz in the specified JSON format.
    """
    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents([Document(page_content=text)])

    # Create a vectorstore for document retrieval
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()

    # Define the quiz generation prompt
    prompt = PromptTemplate(
        template="""
        Create multiple-choice questions with four options, indicating the correct option in each question
        in a JSON format with the following structure:

        {{
          "question_1": {{"question":"the question", "choices":["A", "B", "C", "D"], "correct_choice":"B"}},
          "question_2": {{"question":"the question", "choices":["A", "B", "C", "D"], "correct_choice":"A"}}
              ...
        }}

        Add questions as you can that only cover all info in the context.

        Remove * from the output and make the questions and choices consistent and accurate.

        Generate the Q&A based only on this context: \n\n{context}\n
        """,
    )

    # Chain for generating questions
    rag_chain = (
        {"context": retriever}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate the quiz
    response = rag_chain.invoke("")
    return response

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file and generate a quiz from its content.

    Args:
        file (UploadFile): The uploaded PDF file.

    Returns:
        dict: Generated quiz.
    """
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(file)
        
        # Generate the quiz from extracted text
        quiz = create_quiz_from_text(text)
        return {"quiz": quiz}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
