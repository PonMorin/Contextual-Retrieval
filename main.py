from langchain_community.document_loaders import TextLoader, PyPDFLoader
from ContextualClass.contextual import ContextualRetrieval
import os

def load_document(tarPath):
    documents = []
    for root, dirs, files in os.walk(tarPath): 
        for file in files:
            full_path = os.path.join(root, file) 
            print(full_path)
            if file.endswith('.txt'):
                loader = TextLoader(full_path, encoding="utf-8")
                documents = loader.load()
            elif file.endswith('.md'):
                loader = TextLoader(full_path, encoding="utf-8")
                documents.extend(loader.load())
                # with open(full_path, 'r', encoding='utf-8') as f:
                #     documents = f.read()
            elif file.endswith('.pdf'):
                loader = PyPDFLoader(full_path) 
                documents = loader.load()
    return documents

tarPath = f"./digital_doc"
document = load_document(tarPath)
# print(document)
print("Load Document Successful")

cr = ContextualRetrieval()

for doc in document:
    print('*'*50)
    
    print(f"Document: {doc.metadata}")

    # Process the document
    original_chunks, contextualized_chunks = cr.process_document(doc)

    print('-------------Original-------------')

    print(original_chunks[0])

    print('-------------Context--------------')

    print(contextualized_chunks[0])

    # print('-------------Original-------------')

    # print(original_chunks[3])

    # print('-------------Context--------------')

    # print(contextualized_chunks[3])

    # print('----------------------------')

    original_vectorstore = cr.create_vectoDB(original_chunks, "original")
    contextualized_vectorstore = cr.create_vectoDB(contextualized_chunks, "context")
    print(f"Save {doc.metadata} to VectorDB!!")


    print('*'*50)