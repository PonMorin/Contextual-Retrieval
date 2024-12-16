from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from ContextualClass.contextual import ContextualRetrieval
import os

def load_document(tarPath):
    documents = ""
    for root, dirs, files in os.walk(tarPath): 
        for file in files:
            full_path = os.path.join(root, file) 
            print(full_path)
            if file.endswith('.txt'):
                loader = TextLoader(full_path, encoding="utf-8")
                documents = loader.load()
            elif file.endswith('.md'):
                with open(full_path, 'r', encoding='utf-8') as f:
                    documents = f.read()
            elif file.endswith('.pdf'):
                loader = PyPDFLoader(full_path) 
                documents = loader.load()
    return documents

tarPath = f"./Doc"
document = load_document(tarPath)

cr = ContextualRetrieval()

# Process the document
original_chunks, contextualized_chunks = cr.process_document(document)

print('----------------------------')

print(original_chunks[0])

print('----------------------------')

print(contextualized_chunks[0])