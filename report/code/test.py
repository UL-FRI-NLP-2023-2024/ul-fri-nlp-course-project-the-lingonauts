from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def few_shot_example_formatted_question(question, choices, choice_labels, answer_key):
    # Formats the question with choices and explicitly includes the correct answer.
    return (
        f"{tokenizer.bos_token}[Example] Given the question '{question}' and the following choices: "
        + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices))
        + f". [Answer] {answer_key}. {tokenizer.eos_token}"
    )

def normally_formatted_question(question, choices, choice_labels):
    # Formats the main question without revealing the answer.
    return (
        f"{tokenizer.bos_token}[Question] Given the question '{question}' and the following choices: "
        + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices))
        + ", which one is correct? Answer only with one of the following A, B, C, D or E, like in the examples above. Do not repeat prompt. {tokenizer.eos_token}"
    )

def few_shot_formatted_question(question, choices, choice_labels, val_set):
    similar_questions = find_similar_questions(question, val_set)
    
    # Prepare examples with explicit answers included.
    examples = ""
    for i, sq in enumerate(similar_questions):

        examples += f"\n\n{few_shot_example_formatted_question(sq['question'], sq['choices']['text'], sq['choices']['label'], sq['answerKey'])}\n"

    # Prepare the main question without the answer.
    main_question = normally_formatted_question(question, choices, choice_labels)

    # Combine examples and the main question into a single prompt.
    return examples + "\n\n" + main_question



def cot_formatted_question(question, choices, choice_labels):
    return (
        f"{tokenizer.bos_token}[INST] Given the question '{question}' and the following choices: "
        + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices))
        + ", which one is correct? Think step by step, but answer only with one of the following A, B, C, D or E. [/INST]{tokenizer.eos_token}"
    )













def find_similar_questions(question, val_set, n=3):
    questions = [item['question'] for item in val_set]
    questions.append(question)  # Add the target question

    # Vectorize the questions
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)

    # Compute similarities
    similarities = cosine_similarity(question_vectors[-1], question_vectors[:-1]).flatten()

    # Get indices of the top n similar questions and convert them to Python ints
    top_indices = np.argsort(similarities)[(-n-1):-1] # Get n+1 most similar questions (the last one is the target question)



    # Return the most similar questions (and their metadata)
    return [val_set[int(i)] for i in top_indices]  # Convert numpy int64 to Python int here


#def few_shot_formatted_question_stara(question, choices, choice_labels, val_set):
#    # Find 3 most similar questions
#    similar_questions = find_similar_questions(question, val_set)
#
#    examples = ""
#    for i, sq in enumerate(similar_questions):
#        examples += f"\n\nExample {i+1}: {normally_formatted_question(sq['question'], sq['choices']['text'], sq['choices']['label'], sq["answerKey"])}\n"
#
#    main_question = normally_formatted_question(question, choices, choice_labels)
#
#    return examples + main_question



#print(dataset[0]["question"])
#print(dataset[0]["answerKey"])
#print(find_similar_questions(dataset[0]["question"], dataset))
#print(few_shot_formatted_question(dataset[0]["question"], dataset[0]["choices"]["text"], dataset[0]["choices"]["label"], dataset))


#sys.exit()





#sys.exit()




def process(n_examples):
    count = 0
    for example in dataset:
        if count < n_examples:

            #razpakiramo podatke
            question = example['question']
            choices = example['choices']['text']
            choice_labels = example['choices']['label']
            answer_key = example['answerKey']

            formatted_question = normally_formatted_question(question, choices, choice_labels)

            print(decoded_answer(formatted_question))
            count += 1
        
#process(1)



#sys.exit()
# Opening a text file to write the outputs.
with open("output_cs2.txt", "w") as file:
    file.write("Test entry\n")  
    count = 0  
    for example in dataset:
        if count < 1:
            question = example['question']
            choices = example['choices']['text']
            choice_labels = example['choices']['label']
            answer_key = example['answerKey']

            formatted_question = few_shot_formatted_question(question, choices, choice_labels, dataset)
            decoded = decoded_answer(formatted_question)

            print("Formatted prompt")
            print("------------------------------------------------------------------------------------------------")
            print(formatted_question)
            print("------------------------------------------------------------------------------------------------")

            
            # Writing the processed information to the file for each question.
            #file.write("Question ID: " + example['id'] + "\n")
            #file.write("Question: " + question + "\n")
            #file.write("Choices: " + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices)) + "\n")
            #file.write("Correct Answer Key: " + answer_key + "\n")
            #file.write("Generated Output: " + decoded + "\n")
            #file.write("\n")
            print("------------------------------------------------------------------------------------------------")
            print("ANSWER")
            print(decoded)
            print("------------------------------------------------------------------------------------------------")
            count += 1 
        else:
            break  
    print("Processed first 5 examples.")
