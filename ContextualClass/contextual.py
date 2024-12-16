import hashlib
import os
import getpass
import torch
from typing import List, Tuple
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_community.vectorstores import Chroma

from dotenv import dotenv_values
config = dotenv_values(".env")

os.environ["OPENAI_API_KEY"] = config["openai_api"]



def init_model():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Training will run on CPU.")
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id="scb10x/llama-3-typhoon-v1.5x-8b-instruct",
        device_map="auto",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            return_full_text=False,
        ),
        model_kwargs={"quantization_config": quantization_config},
    )

    chat_model = ChatHuggingFace(llm=llm)

    return chat_model

class ContextualRetrieval:
    """
    A class that implements the Contextual Retrieval system.
    """

    def __init__(self):
        """
        Initialize the ContextualRetrieval system.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
            chunk_size=512, 
            chunk_overlap=300,
            length_function=len,
            add_start_index=True
        )
        self.embeddings = OpenAIEmbeddings()
        self.llm = init_model()
    
    def process_document(self, document: str) -> Tuple[List[Document], List[Document]]:
        """
        Process a document by splitting it into chunks and generating context for each chunk.
        """
        chunks = self.text_splitter.create_documents([document])
        contextualized_chunks = self._generate_contextualized_chunks(document, chunks)
        return chunks, contextualized_chunks
    
    def _generate_contextualized_chunks(self, document: str, chunks: List[Document]) -> List[Document]:
        """
        Generate contextualized versions of the given chunks.
        """
        contextualized_chunks = []
        for chunk in chunks:
            context = self._generate_context(document, chunk.page_content)
            contextualized_content = f"{context}\n\n{chunk.page_content}"
            contextualized_chunks.append(Document(page_content=contextualized_content, metadata=chunk.metadata))
        return contextualized_chunks
    
    def _generate_context(self, document: str, chunk: str) -> str:
        """
        Generate context for a specific chunk using the language model.
        """
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specializing in answering questions, particularly for Chitralada Technology Institute (CDTI). Your task is to provide brief, relevant context for a chunk of text from CDTI's document.
        Here is the document:
        <document>
        {document}
        </document>

        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>

        Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
        1. Identify the main topic or focus of the chunk (e.g., event details, academic program, administrative procedure, key announcement).
        2. Mention any relevant time frames, dates, or recurring schedules (e.g., semester schedule, specific event date, monthly update).
        3. Highlight how this information relates to the broader context of the document (e.g., its importance to students, faculty, or stakeholders in the institution).
        4. Include any critical details or numbers, such as dates, deadlines, room numbers, or participant names, that help clarify the context.
        5. Do not use phrases like "This chunk discusses" or "This section provides". Instead, directly state the context.

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.

        Context:
        """)
        messages = prompt.format_messages(document=document, chunk=chunk)
        response = self.llm.invoke(messages)
        return response.content
    
    def create_vectoDB(self, chunks: List[Document]) -> Chroma:
        """
        Create a vector DB for the given chunks
        """
        data = f"./Data/langchain"
        vectordb = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(), persist_directory=data)
        vectordb.persist()

    def create_bm25_index(self, chunks: List[Document]) -> BM25Okapi:
        """
        Create a BM25 index for the given chunks.
        """
        tokenized_chunks = [chunk.page_content.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        prompt = ChatPromptTemplate.from_template("""
        โปรดตอบคำถามโดยให้ชัดเจนและชัดเจนตามข้อมูลต่อไปนี้, หากข้อมูลไม่เพียงพอต่อการตอบคำถาม โปรดแจ้งให้ทราบ

        Question: {query}

        Relevant information:
        {chunks}

        Answer:
        """)
        messages = prompt.format_messages(query=query, chunks="\n\n".join(relevant_chunks))
        response = self.llm.invoke(messages)
        return response.content