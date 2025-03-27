import os
import pandas as pd
import time
from dotenv import dotenv_values
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from ContextualClass.contextual import ContextualRetrieval
from intent.intent import (initialize_model, predict_intent)
from intent.prompt_template import RAG_PROMPT_TEMPLATES, DYNAMIC_PROMPT_TEMPLATE
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


def chat_with_model(prompt, exam_name):
    dynamic_doc = ["academic_calendar", "student_activities"]

    if exam_name == "final":
        intent = intention(prompt=prompt)
    elif exam_name == "course":
        intent = "course"
    elif exam_name == "scholarship":
        intent = "Scholarship"
    elif exam_name == "general":
        intent = "general_question"

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
    # Load exam data
    course_data = pd.read_excel('Exam/ExamQuestion.xlsx')
    scholar_data = pd.read_excel('Exam/ข้อสอบทุน.xlsx')
    general_data = pd.read_excel('Exam/ข้อสอบทั่วไป.xlsx')
    all_data = pd.read_excel('Exam/Test_final.xlsx')

    # Create a new DataFrame to store results
    output_data = []

    exam_dict = {"final": all_data}  #"course": course_data, "scholarship": scholar_data, "general":general_data,
    for exam_name in exam_dict:
        for idx, row in exam_dict[exam_name].iterrows():
            print("----->ข้อ:", idx+1)
            if exam_name == "final":
                instruction = row["question"]
                ref_answer = row["ref_answer"]
                ref_intent = row["intent"]
                    
                ai_result, pred_Intent = chat_with_model(instruction, exam_name=exam_name)  # Change selected_option as needed
                check_answer = model.generate_check_answer(ref_answer=ref_answer, llm_answer=ai_result)

                print(f"-------->{ai_result}, {check_answer}\n")
                output_data.append({"no": idx + 1, 
                                    "question": instruction, 
                                    'intent': ref_intent,
                                    'ref_answer': ref_answer, 
                                    "predict_Intent": pred_Intent, 
                                    "AI_answer": ai_result, 
                                    "Check_answer": check_answer.content})
                
            else:
                instruction = row["instruction"]
                ref_answer = row["ref_answer"]
                ai_result, pred_Intent = chat_with_model(instruction, exam_name=exam_name)  # Change selected_option as needed
                check_answer = model.generate_check_answer(ref_answer=ref_answer, llm_answer=ai_result)

                print(f"-------->{ai_result},\n {check_answer}\n")
                output_data.append({"no": idx + 1, 
                                    "question": instruction, 
                                    'ref_answer': ref_answer, 
                                    "AI_answer": ai_result, 
                                    "Check_answer": check_answer.content})
            
            time.sleep(1)
        output_df = pd.DataFrame(output_data)
        print(len(output_df))

        # Save results to a new Excel file
        output_df.to_excel(f"exam_result/eval_{exam_name}_exam.xlsx", index=False)
        output_data = []

if __name__ == "__main__":
    main()