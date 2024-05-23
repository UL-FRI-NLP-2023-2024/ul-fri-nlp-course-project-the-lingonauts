from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda" # the device to load the model onto
print("Setting device to " + device)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


# Load the dataset
dataset = load_dataset("tau/commonsense_qa", split="validation")
print(f"Dataset loaded with {len(dataset)} entries.")

with open("output_zeroCoT.txt", "w") as file:
    file.write("Test entry" "\n")
    for example in dataset:
        question = example['question']
        choices = example['choices']['text']
        choice_labels = example['choices']['label']
        answer_key = example['answerKey']
        print("Answer Key:", example['answerKey'])


        # Format the question with instruction tokens for zero shot CoT
        formatted_question = (
            f"{tokenizer.bos_token}[INST] Given the question '{question}' and the following choices: "
            + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices))
            + f", which one is correct?  Answer only with one of the following A, B, C, D or E. Let's think step by step. The final output should be formatted as: 'Correct answer letter: <letter>', where <letter> is A,B,C,D or E. Answer with one letter only in the required format. Do not include any additional information. [/INST]{tokenizer.eos_token}"
        )
        print("Formated prompt")
        print("------------------------------------------------------------------------------------------------")
        print(formatted_question)
        print("------------------------------------------------------------------------------------------------")
        input_ids = tokenizer.encode(formatted_question, return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)[0]

        file.write("Question ID: " + example['id'] + "\n")
        file.write("Question: " + question + "\n")
        file.write("Choices: " + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices)) + "\n")
        file.write("Correct Answer Key: " + answer_key + "\n")
        file.write("Generated Output: " + decoded + "\n")
        file.write("\n")
        print("------------------------------------------------------------------------------------------------")
        print("ANSWER")
        print(decoded)
        print("------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------")
        
print("Processed all examples.")