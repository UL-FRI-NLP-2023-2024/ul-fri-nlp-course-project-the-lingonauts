from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


#1.IMPORTING MODELS, DATA================================================================================================
device = "cuda"
print("Setting device to " + device)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

dataset = load_dataset("tau/commonsense_qa", split="validation")
print(f"Dataset loaded with {len(dataset)} entries.")



#2.QUESTION FORMATTING================================================================================================
def decoded_answer(formatted_question):
    input_ids = tokenizer.encode(formatted_question, return_tensors="pt").to(device)
    model.to(device)  # Moving the model to the GPU.

    # Generating an answer from the model.
    generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)[0]

    return decoded



def find_similar_questions(question, val_set, n=3):
    questions = [item['question'] for item in val_set]

    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)

    similarities = cosine_similarity(question_vectors[-1], question_vectors[:-1]).flatten()

    top_indices = np.argsort(similarities)[(-n):]
    return [val_set[int(i)] for i in top_indices] 


def few_shot_example_formatted_question(question, choices, choice_labels, answer_key):
    return (
        f"[Example question] Given the question '{question}' and the following choices: " #{tokenizer.bos_token}
        + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices)) 
        + ", which one is correct? Answer only with one of the following A, B, C, D or E.[End of example question]" #{tokenizer.eos_token}
        + f". \n[Answer]{answer_key}[End of answer]" #{tokenizer.eos_token}
    )

def normally_formatted_question(question, choices, choice_labels):
    return (
        f"[Main question] Given the question '{question}' and the following choices: "
        + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices))
        + f", which one is correct? Answer only with one of the following A, B, C, D or E.[End of main question]" #{tokenizer.eos_token}
    )

def few_shot_formatted_question(question, choices, choice_labels, val_set):
    similar_questions = find_similar_questions(question, val_set)
    
    examples = ""
    for i, sq in enumerate(similar_questions):

        examples += f"\n\n{few_shot_example_formatted_question(sq['question'], sq['choices']['text'], sq['choices']['label'], sq['answerKey'])}\n"

    main_question = normally_formatted_question(question, choices, choice_labels)


    prompt = (
        #f"{tokenizer.bos_token}" +
        f"[Introduction] You will see examples and a main question. Please provide the answer to the main question based on these examples. Your response can only include one character: A, B, C, D or E. Put your answer to the main question between brackets that say [Main answer] at the beggining of sentence and [End of main answer] at the end of it.[End of introduction]"
        + f"{examples}"
        + f"\n\n{main_question}" #{tokenizer.eos_token}
        #+ "\nANSWER:"
    )
    return prompt




def extracted_letter(decoded_output):
    
    
    # Define the regex pattern to extract the main answer
    main_pattern = r'\[Main answer\](.*?)(?:\[End of main answer\]|</s>)'
    
    # Search for the main answer
    main_match = re.search(main_pattern, decoded_output)
    
    if main_match:
        # Extract and return the main answer
        return main_match.group(1).strip()
    else:
        # Define the regex pattern to extract all answers
        answer_pattern = r'\[Answer\](.*?)(?:\[End of answer\])'
        
        # Find all matches for answers
        all_answers = re.findall(answer_pattern, decoded_output)
        
        if all_answers:
            # Extract and return the last answer
            return all_answers[-1].strip()
        else:
            raise ValueError("\n\n\n\n\n" + "Target sentence not found in the decoded output: " + decoded_output + "\n\n\n\n\n")

with open("output_cs2.txt", "w") as file:
    file.write("Test entry\n")  
    count = 0  
    correct = 0
    for example in dataset:
        if count < 100:
            question = example['question']
            choices = example['choices']['text']
            choice_labels = example['choices']['label']
            answer_key = example['answerKey']

            formatted_question = few_shot_formatted_question(question, choices, choice_labels, dataset)
            decoded = decoded_answer(formatted_question)
            extracted = extracted_letter(decoded)


            file.write("\n\n\n")
            file.write(": " + question + "\n")
            file.write("Answer key: " + answer_key + "\n")
            file.write("Extracted output: " + extracted_letter(decoded) + "\n")
           
            print("\n\n\n")
            
            print("Question: " + question + "\n")
            print("Answer key: " + answer_key + "\n")
            print("Extracted output: " + extracted_letter(decoded) + "\n")
            #
            if answer_key in extracted:
                correct += 1
            count += 1 
            accuracy = correct/count
            print("Accuracy: ", accuracy)
            print("Count: ", count)
        else:
            break  
    print("Processed first 5 examples.")

