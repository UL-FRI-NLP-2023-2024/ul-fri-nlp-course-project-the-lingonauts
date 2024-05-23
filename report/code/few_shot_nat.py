from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random


#1.IMPORTING MODELS, DATA================================================================================================
device = "cuda"
print("Setting device to " + device)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

dataset = load_dataset("allenai/common_gen", split="validation")

print(f"Dataset loaded with {len(dataset)} entries.")



#2.QUESTION FORMATTING================================================================================================
def decoded_answer(formatted_question):
    input_ids = tokenizer.encode(formatted_question, return_tensors="pt").to(device)
    model.to(device)  # Moving the model to the GPU.

    # Generating an answer from the model.
    generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)[0]

    return decoded



def find_similar_questions(concept_set_idx, dataset, n=3):
    # Filter the dataset to get examples with the same concept_set_idx
    similar_questions = [item for item in dataset if item['concept_set_idx'] == concept_set_idx]
    
    # Randomly select up to n examples if there are more than n
    if len(similar_questions) > n:
        similar_questions = random.sample(similar_questions, n)
    
    return similar_questions


def few_shot_example_formatted_question(concepts, target):
    return (
        f"[Example concepts] {', '.join(concepts)} [End of example concepts]" #{tokenizer.eos_token}
        + f". \n[Target sentence]{target}[End of target sentence]" #{tokenizer.eos_token}
    )

def normally_formatted_question(concepts):
    return (
        f"[Main concepts] {', '.join(concepts)} [End of main concepts]" #{tokenizer.eos_token}
    )

def few_shot_formatted_question(concept_set_idx, concepts, dataset):
    similar_questions = find_similar_questions(concept_set_idx, dataset, n=2)
    
    examples = ""
    for i, sq in enumerate(similar_questions):

        examples += f"\n\n{few_shot_example_formatted_question(concepts, target)}\n"

    main_question = normally_formatted_question(concepts)


    prompt = (
        #f"{tokenizer.bos_token}" +
        f"[Introduction] You will see examples of target sentences constructed using specific concepts. Please provide one natural language sentence using the given concepts. [End of introduction]"
        + f"{examples}"
        + f"\n\n{main_question}" #{tokenizer.eos_token}
        #+ "\nANSWER:"
    )
    return prompt


with open("few_shot_commongen.txt", "w") as file:
    file.write("Test entry\n")  
    #count = 0  
    for example in dataset:
        #if count < 3:

        concept_set_idx = example['concept_set_idx']
        concepts = example['concepts']
        target = example['target']



        formatted_question = few_shot_formatted_question(concept_set_idx, concepts, dataset)
        decoded = decoded_answer(formatted_question)

        #print("Formatted prompt")
        #print("------------------------------------------------------------------------------------------------")
        #print(formatted_question)
        #print("------------------------------------------------------------------------------------------------")
        #print("------------------------------------------------------------------------------------------------")
        #print("ANSWER")
        #print(decoded)
        #print("------------------------------------------------------------------------------------------------")
        #count += 1 
        file.write("Concepts set ID: " + str(example['concept_set_idx']) + "\n")
        file.write("Concepts: " + str(example['concepts']) + "\n")
        file.write("Target: " + example['target'] + "\n")
        file.write("Generated Output: " + decoded + "\n")
#        else:
#            break  
#    print("Processed first 5 examples.")
