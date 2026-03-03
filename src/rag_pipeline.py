import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.config import settings
from src.vector_store import VectorStoreManager


# -------------------------------------------------------------------
# Structured Output Definitions
# -------------------------------------------------------------------
class SourceNode(BaseModel):
    page: int = Field(description="The page number where the information was found.")
    text: str = Field(description="A short, direct quote from the text that supports the answer.")

class RAGResponse(BaseModel):
    answer: str = Field(description="The detailed answer to the user's question.")
    confidence: str = Field(description="High, Medium, or Low based on how explicitly the document answers the query.")
    sources: List[SourceNode] = Field(description="List of supporting evidence citations.")


# -------------------------------------------------------------------
# Anti-Hallucination Prompt Template
# -------------------------------------------------------------------
# This strict prompt ensures the LLM does not use outside knowledge.
SYSTEM_PROMPT = """
You are an expert financial and corporate analyst assistant.
Your ONLY task is to answer questions strictly based on the provided document excerpts from the Swiggy Annual Report.

CRITICAL INSTRUCTIONS:
1. You MUST NOT use any outside knowledge or prior training data. 
2. If the answer is NOT explicitly contained within the provided context, you MUST respond exactly with:
   "The information is not available in the provided document."
3. If the context contains the answer, provide a clear, professional response.
4. You must cite your sources accurately using the page numbers provided in the context metadata.
5. Provide a short, direct quote for each source you cite to prove where you found the information.

CONTEXT EXCERPTS:
{context}

USER QUESTION:
{question}

{format_instructions}
"""


class RAGPipeline:
    """
    Orchestrates the retrieval and generation phases.
    Uses LangChain Expression Language (LCEL) for a clean, modular pipeline.
    """
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        # Initialize the retriever only if the index exists, otherwise handles gracefully later
        try:
            self.retriever = self.vector_store_manager.get_retriever()
        except RuntimeError:
            self.retriever = None
            
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.0  # Zero temperature for maximum determinism and factual adherence
        )
        
        # Set up structured output parser based on our Pydantic model
        self.output_parser = PydanticOutputParser(pydantic_object=RAGResponse)
        
        self.prompt = ChatPromptTemplate.from_template(
            template=SYSTEM_PROMPT,
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Formats retrieved documents into a single string, injecting metadata 
        so the LLM knows which page each chunk came from.
        """
        formatted_chunks = []
        for i, doc in enumerate(docs):
            page_num = doc.metadata.get("page", "Unknown")
            # Clear delimiter helps the LLM distinguish separate chunks
            formatted_chunks.append(f"--- CHUNK {i+1} (Page: {page_num}) ---\n{doc.page_content}")
            
        return "\n\n".join(formatted_chunks)

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline: Retrieve -> Format -> Generate -> Parse.
        
        Returns:
            A dictionary matching the RAGResponse structure.
        """
        if not self.retriever:
            # Attempt to lazily initialize in case index was created after pipeline instantiation
            try:
                self.retriever = self.vector_store_manager.get_retriever()
            except RuntimeError as e:
                 raise RuntimeError("Vector index not found. Please ingest the document first.") from e
        
        # 1. Retrieve relevant chunks
        docs = self.retriever.invoke(question)
        
        # If no documents are returned (highly unlikely with FAISS but possible with strict score thresholds)
        if not docs:
           return {
               "answer": "The information is not available in the provided document.",
               "confidence": "Low",
               "sources": []
           }
           
        formatted_context = self._format_docs(docs)

        # 2. Construct the pipeline using LCEL
        chain = self.prompt | self.llm | self.output_parser
        
        # 3. Execute the chain
        try:
            response_obj: RAGResponse = chain.invoke({
                "context": formatted_context,
                "question": question
            })
            
            # Additional validation: if the LLM followed the strict instruction for missing info,
            # clear the sources to ensure consistency.
            if "not available in the provided document" in response_obj.answer.lower():
                response_obj.sources = []
                response_obj.confidence = "Low"
                
            return response_obj.model_dump()
            
        except Exception as e:
            # Provide a safe fallback if generation or parsing fails
            print(f"Error during generation: {e}")
            return {
                "answer": "An error occurred while generating the response. Please try rephrasing the question.",
                "confidence": "Low",
                "sources": [],
                "error": str(e)
            }


if __name__ == "__main__":
    from dotenv import load_dotenv
    import pprint
    
    load_dotenv()
    pipeline = RAGPipeline()
    res = pipeline.generate_answer("What was the total revenue for the year?")
    pprint.pprint(res)
