import os
import pandas as pd
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
# def chat_with_model(prompt, selected_option):
#     # Load vectorstores
#     contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name=f"{selected_option}")
    
#     contextual_vector_results = contextual_vector.similarity_search(prompt, k=3)
#     contextual_vector_answer = model.generate_answer_api(prompt, [doc.page_content for doc in contextual_vector_results])
    
#     return contextual_vector_answer

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
    # intent = "general_question"
    # if intent == "general":
    #     intent = "general_question"
    # elif intent == "scholarship":
    #     intent = "Scholarship"

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
    # if intent in dynamic_doc:
    #     if intent == "student_activities":
    #         with open("/home/s6410301020/SeniorProject/FDT_cdti_Chat/doc/dynamic 1.md", "r") as file:
    #             contexts = file.read()
    #         print(contexts)

    #         system_prompt = DYNAMIC_PROMPT_TEMPLATE(intent=intent, context=contexts)
    #         answer, _ = model.generate_answer_api_dynamic_with_history(prompt, system_prompt=system_prompt)
    #         response = f"[{intent}] {answer.content}]"
    #     else:
    #         with open("/home/s6410301020/SeniorProject/FDT_cdti_Chat/doc/dynamic 2.md", "r") as file:
    #             contexts = file.read()
    #         print(contexts)

    #         system_prompt = DYNAMIC_PROMPT_TEMPLATE(intent=intent, context=contexts)
    #         answer, _ = model.generate_answer_api_dynamic_with_history(prompt, system_prompt=system_prompt)
    #         response = f"[{intent}] {answer.content}]"

    # else:
    #     system_prompt = RAG_PROMPT_TEMPLATES[intent]
    #     contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name=f"cdti_doc")
    #     retriever = contextual_vector.as_retriever(search_kwargs={"k": 4})

    #     answer, _ = model.generate_answer_api_with_history(prompt, retriever=retriever, system_prompt=system_prompt)

    #     # Include the selected dropdown option in the response
    #     response = f"[{intent}] {answer["answer"]}"


    return response, intent

model_name = "proideas/CDTI-intent-classification"



def main():

    tokenizer, model_intent = initialize_model()

    # Load course data
    # course_data = pd.read_excel('Exam/ExamQuestion.xlsx')
    # scholar_data = pd.read_excel('Exam/ข้อสอบทุน.xlsx')
    # general_data = pd.read_excel('Exam/ข้อสอบทั่วไป.xlsx')

    all_data = pd.read_excel('Exam/Test_final.xlsx')


    # Create a new DataFrame to store results
    output_data = []

    # exam_dict = {"general":general_data} 
    for idx, row in all_data.iterrows():
        instruction = row["question"]
        ref_answer = row["ref_answer"]
        ref_intent = row["intent"]

        # intent = predict_intent(instruction,tokenizer,model_intent)
            
        ai_result, pred_Intent = chat_with_model(instruction)  # Change selected_option as needed
        check_answer = model.generate_check_answer(ref_answer=ref_answer, llm_answer=ai_result)

        print(f"{ai_result}, {check_answer}\n")
        output_data.append({"no": idx + 1, 
                            "question": instruction, 
                            'intent': ref_intent,
                            'ref_answer': ref_answer, 
                            "predict_Intent": pred_Intent, 
                            "AI_answer": ai_result, 
                            "Check_answer": check_answer.content})

    # Convert list to DataFrame
    output_df = pd.DataFrame(output_data)
    print(len(output_df))

    # Save results to a new Excel file
    output_df.to_excel(f"exam_result/eval_Final_exam.xlsx", index=False)

if __name__ == "__main__":
    main()