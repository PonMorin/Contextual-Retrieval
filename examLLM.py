import os
import pandas as pd
from dotenv import dotenv_values
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from ContextualClass.contextual import ContextualRetrieval
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define directories for vectorstores
context_data = "./doc_Data/context"

# Initialize the Contextual Retrieval system
model = ContextualRetrieval()

def chat_with_model(prompt, selected_option):
    # Load vectorstores
    contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name=f"{selected_option}")
    
    contextual_vector_results = contextual_vector.similarity_search(prompt, k=3)
    contextual_vector_answer = model.generate_answer_api(prompt, [doc.page_content for doc in contextual_vector_results])
    
    return contextual_vector_answer

model_name = "proideas/CDTI-intent-classification"

# โหลดโมเดล
def load_intent_classifier():
    """
    Load a sequence classification model and tokenizer from Hugging Face Hub.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()  # Set model to evaluation mode
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model from Hugging Face Hub: {e}")

# โหลด label mapping
def predict_intent(prompt, tokenizer, model):
    """
    Predict the intent of a given prompt using a pretrained model.
    """
    intent_mapping = ["Scholarship", "academic_calendar", "course", "general_question", "student_activities"]
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()

    return intent_mapping[predicted_class]


def handle_intent(intent, prompt):
    """
    return intent and prompt to User.
    """
    if intent in ["academic_calendar", "student_activities"]:
        return f"{intent}\n{prompt}"
    elif intent in ["Scholarship", "course", "general_question"]:
        return f"{intent}\n{prompt}"
    

#สำหรับโหลดโมเดลเมื่อเริ่มแอป
def initialize_model():
    """
    Initialize the model when the application starts.
    Returns the tokenizer and model for global use.
    """
    return load_intent_classifier()

def main():

    tokenizer, model_intent = initialize_model()

    # Load course data
    course_data = pd.read_excel('Exam/ExamQuestion.xlsx')
    scholar_data = pd.read_excel('Exam/ข้อสอบทุน.xlsx')
    general_data = pd.read_excel('Exam/ข้อสอบทั่วไป.xlsx')

    # Randomly sample 34 rows from each dataset
    course_sample = course_data.sample(n=34, random_state=42)
    scholar_sample = scholar_data.sample(n=32, random_state=42)
    general_sample = general_data.sample(n=34, random_state=42)

    # Combine sampled data
    combined_data = pd.concat([course_sample, scholar_sample, general_sample])

    # Shuffle the combined data
    # shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create a new DataFrame to store results
    output_data = []

    for idx, row in combined_data.iterrows():
        instruction = row["instruction"]
        result = row["result"]

        intent = predict_intent(instruction,tokenizer,model_intent)
        if intent == "course":
            intent = "course"
        elif intent == "Scholarship":
            intent = "capital"
        elif intent == "general_question":
            intent = "general"
            
        ai_result = chat_with_model(instruction, selected_option=intent)  # Change selected_option as needed
        check_answer = model.check_answer(ai_result, result=result)
        output_data.append({"no": idx + 1, "instruction": instruction, 'result': result, "AI_result": ai_result, "Check_answer": check_answer, "Intent": intent})

    # Convert list to DataFrame
    output_df = pd.DataFrame(output_data)

    # Save results to a new Excel file
    output_df.to_excel("exam_result/AI_exam.xlsx", index=False)

if __name__ == "__main__":
    main()