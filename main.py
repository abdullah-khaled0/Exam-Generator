from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from pydantic import BaseModel
import os

from fastapi import FastAPI, HTTPException




app = FastAPI()


# Input model for query
class QueryRequest(BaseModel):
    query: str

# Google API setup
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please set it as an environment variable.")


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")


class TextRequest(BaseModel):
    text: str


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

@app.post("/create-quiz")
async def create_quiz(request: TextRequest):
    try:
        text = request.text
        quiz = create_quiz_from_text(text)
        return {"quiz": quiz}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))