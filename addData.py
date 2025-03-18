import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import dotenv_values
config = dotenv_values(".env")

os.environ["OPENAI_API_KEY"] = config["openai_api"]

context_data = "./doc_Data/context"

contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name=f"cdti_doc")



document_1 = Document(
    page_content="""
    ## บุคลากร คณบดีของคณะเทคโนโลยีดิจิทัล และ อาจารย์ประจำคณะ ของคณะเทคโนโลยีดิจิทัล
    คณบดีประจำคณะ หรือ `ชื่อของคณบดี`: รองศาสตราจารย์ ดร. วรา วราวิทย์
    อาจารย์ประจำคณะ: 
    1. รองศาสตราจารย์ ดร. สุมาลี อุณหวณิชย์
    2. รองศาสตราจารย์ ดร. ธีรศิลป์ ทุมวิภาต
    3. อาจารย์ ชยันต์ คงทองวัฒนา
    4. ผู้ช่วยศาสตราจารย์ ภาษิศร์ ณ รังษี
    5. อาจารย์ กฤษฎา พรหมสุทธิรักษ์
    6. อาจารย์ ณัฐวดี ศรีคชา
    7. ดร. ศรายุทธ ฉายสุริยะ
    8. ผู้ช่วยศาสตราจารย์ ดร. วรัญญู วงษ์เสรี
    9. ผู้ช่วยศาสตราจารย์ สิริทัต เตชะพะโลกุล
    10. ผู้ช่วยศาสตราจารย์ ดร. ดำรงค์ฤทธิ์ เศรษฐ์ศิริโชค
    """,
    metadata={"source": "อาจารย์ประจำคณะ"},
)

document_2 = Document(
    page_content="""
    ### คณบดีประจำคณะเทคโนโลยีดิจิทัล
    คณบดีประจำคณะ หรือ `ชื่อของคณบดี` หรือ คณบดีของคณะเทคโนโลยีดิจิทัล: รองศาสตราจารย์ ดร. วรา วราวิทย์
    """,
    metadata={"source": "คณบดีของคณะเทคโนโลยีดิจิทัล"},
)

# document_2 = Document(
#     page_content="""
#     # ค่าเทอมของคณะดิจิทัล
#     ## หลักสูตร 4 ปี
#     ### `ค่าเทอมของคณะ`
#     | สาขาวิชา | ปี 1 |  | ปี 2 |  | ปี 3 |  | ปี 4 |  |
#     |----------|------|------|------|------|------|------|------|------|
#     |          | เทอม 1 | เทอม 2 | เทอม 1 | เทอม 2 | เทอม 1 | เทอม 2 | เทอม 1 | เทอม 2 |
#     | สาขาวิศวกรรมคอมพิวเตอร์ | 31,500 | 31,500 | 31,500 | 31,500 | 31,500 | 31,500 | 31,500 | 31,500 |
#     | สาขาการออกแบบดิจิทัลและเทคโนโลยี | 31,500 | 31,500 | 31,500 | 31,500 | 31,500 | 31,500 | 31,500 | 31,500 |
#     """,
#     metadata={"source": "./digital_doc/general/ค่าเทอม.md"},
# )

# documents = [document_1, document_2]

# contextual_vector.delete(ids=["อาจารย์ประจำคณะ"])
# contextual_vector.delete(where={"source": "./digital_doc/general/ค่าเทอม.md"})
contextual_vector.add_documents(documents=[document_2], ids=["คณบดีของคณะเทคโนโลยีดิจิทัล"])

