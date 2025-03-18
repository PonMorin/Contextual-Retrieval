import os
import pandas as pd
from dotenv import dotenv_values
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from ContextualClass.contextual import ContextualRetrieval
from intent.intent import (initialize_model, predict_intent)
from intent.prompt_template import RAG_PROMPT_TEMPLATES, DYNAMIC_PROMPT_TEMPLATE

# Load API key from environment file
config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["openai_api"]

# Initialize the Contextual Retrieval system
cr = ContextualRetrieval()

# Define directories for vectorstores
og_data = "./doc_Data/original"
context_data = "./doc_Data/context"

# Load vectorstores
contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name="cdti_doc")
retriever = contextual_vector.as_retriever(search_kwargs={"k": 4})

# Define queries
# queries = [
#     "ชื่อปริญญาและสาขาวิชาของหลักสูตรนี้คืออะไร ผมอยากได้ทั้งชื่อภาษาไทยและภาษาอังกฤษ",
#     "ผมอยากทราบ วัน เวลาในการดำเนินการเรียนการสอน",
#     "จำนวนหน่วยกิตที่ต้องเรียนทั้งหมด ต้องเรียนทั้งหมดกี่หน่วยกิต",
#     "รายวิชาของกลุ่มวิชาสังคมศาสตร์ มีรายวิชาอะไรบ้าง",
#     "อยากทราบ แผนการศึกษา ของคณะและสาขา ปีที่ 3 ภาคการศึกษาที่ 2 ",
#     "ช่วยอธิบายรายวิชาของ วิชาโครงสร้างของระบบคอมพิวเตอร์ ให้หน่อย",
#     "อยากทราบขั้นตอน เกณฑ์การสำเร็จการศึกษา",
#     "อยากทราบวิธีการสมัครขอรับทุน",
#     "ทุนการศึกษามีกี่ประเภทและอธิบายแต่ละประเภทให้หน่อย",
#     "ผมอยากทราบคุณสมบัติและหลักเกณฑ์การขอรับทุนการศึกษา",
#     "รอบแฟ้มสะสมผลงานสามารถสมัครรับทุนได้มั้ย",
#     "หลักเกณฑ์การพิจารณาทุนส่งเสริมศักยภาพการศึกษามีอะไรบ้าง",
#     "หลักเกณฑ์การจ่ายทุนช่วยเหลือการศึกษาสำหรับผู้ขาดแคลนมีอะไรบ้าง",
#     "อยากทราบหลักเกณฑ์การจ่ายทุนพัฒนาและส่งเสริมศักยภาพผู้เรียน"
# ]

# queries = [
#     "อยากทราบวิธีการสมัครขอรับทุน",
#     "ทุนการศึกษามีกี่ประเภทและอธิบายแต่ละประเภทให้หน่อย",
#     "ผมอยากทราบคุณสมบัติและหลักเกณฑ์การขอรับทุนการศึกษา",
#     "รอบแฟ้มสะสมผลงานสามารถสมัครรับทุนได้มั้ย",
#     "หลักเกณฑ์การพิจารณาทุนส่งเสริมศักยภาพการศึกษามีอะไรบ้าง",
#     "หลักเกณฑ์การจ่ายทุนช่วยเหลือการศึกษาสำหรับผู้ขาดแคลนมีอะไรบ้าง",
#     "อยากทราบหลักเกณฑ์การจ่ายทุนพัฒนาและส่งเสริมศักยภาพผู้เรียน"
# ]

# df = pd.read_excel('/home/s6410301020/SeniorProject/Contextual-Retrieval/ข้อสอบทั่วไป.xlsx')
# queries = df["instruction"]




# Define directories for vectorstores
context_data = "./doc_Data/context"

# Initialize the Contextual Retrieval system
model = ContextualRetrieval()
tokenizer, intent_model = initialize_model()

def intention(prompt: str):
    try:        
        # Initialize the model when the application starts. Load the model and tokenizer
        # tokenizer, intent_model = initialize_model()

        intent = predict_intent(prompt, tokenizer, intent_model)
        
        return intent
    except Exception as e:
        print(e)
        return e


def chat_with_model(prompt):
    dynamic_doc = ["academic_calendar", "student_activities"]

    intent = intention(prompt=prompt)

    print("intent --->", intent)
    if intent in dynamic_doc:
        response = f"[{intent}] Classify ผิด"
    else:
        system_prompt = RAG_PROMPT_TEMPLATES[intent]
        contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name=f"cdti_doc")
        retriever = contextual_vector.as_retriever(search_kwargs={"k": 4})

        answer = model.generate_answer_api(prompt, system_prompt=system_prompt, retriever=retriever)

        # Include the selected dropdown option in the response
        response = f"[{intent}] {answer["answer"]}"


    return response, intent

model_name = "proideas/CDTI-intent-classification"



def main():

    queries = [
    # "สถาบันอยู่ตรงไหนหรอครับเดินทางไปยังไงได้บ้าง?",
    "คณบดีคณะคือใคร",
    # "จะติดรูปในใบสมัตรต้องใช้ขนาดเท่าไหร่",
    "วิชาบังคับก่อนหน้าของวิชาระบบปฎิบัติการคือวิชาอะไร",
    # "อยากทราบเงื่อนไขการรับเกียรตินิยม",
    # "ปี 4 เทอม 2 มีเรียนอะไรบ้าง",
    # "คณบดีชื่ออะไร",
    # "คณบดีของคณะเทคโนโลยีดิจิทัล",
    # "คณบดีของคณะคือใคร",
    # "อาจารย์ประจำคณะมีใครบ้างครับ",
    # "ค่าเทอมระดับปริญญาตรีของคณะเทคโนโลยีดิจิทัล",
    # "อยากทราบค่าเทอมของคณะเทคโนโลยีดิจิทัลครับ",
    # "ค่าเทอมของคณะดิจิทัล",
    # "อยากทราบค่าเทอมของคณะ",
    # "ทุนมีกี่ประเภท",
    # "ค่าเทอมของคณะเทคโนโลยีอุตสาหกรรม",
    "การรับเกียรตินิยมอันดับ 1 ต้องมีเงื่อนไขอย่างไรบ้่าง"
    ]
    
    # Create a new DataFrame to store results
    output_data = []

    # exam_dict = {"general":general_data} 
    for i in queries:

        # intent = predict_intent(instruction,tokenizer,model_intent)
        ai_result, pred_Intent = chat_with_model(i)  # Change selected_option as needed

        print(f"Answer: {pred_Intent}, {ai_result}, \n")

if __name__ == "__main__":
    main()