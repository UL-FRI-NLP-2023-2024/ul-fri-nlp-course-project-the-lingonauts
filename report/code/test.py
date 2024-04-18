from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda" # the device to load the model onto
print("Setting device to " + device)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


# Load the dataset
dataset = load_dataset("tau/commonsense_qa", split="test")
print(f"Dataset loaded with {len(dataset)} entries.")

# Commonsense QA example
#example = {
# 'id': '075e483d21c29a511267ef62bedc0461',
# 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
# 'question_concept': 'punishing',
# 'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
#  'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
# 'answerKey': 'A'
#}



# Format the question with instruction tokens
#formatted_question = (
#    f"{tokenizer.bos_token}[INST] Given the question '{example['question']}' and the following choices: "
#    f"A: ignore, B: enforce, C: authoritarian, D: yell at, E: avoid, which one is correct? [/INST]{tokenizer.eos_token}"
#)

with open("output_cs.txt", "w") as file:
    file.write("Test entry" "\n")
    count = 0
    for example in dataset:
        if count < 5:
            question = example['question']
            choices = example['choices']['text']
            choice_labels = example['choices']['label']
            answer_key = example['answerKey']
            print("Answer Key:", example['answerKey'])


            # Format the question with instruction tokens
            formatted_question = (
                f"{tokenizer.bos_token}[INST] Given the question '{question}' and the following choices: "
                + ", ".join(f"{label}: {text}" for label, text in zip(choice_labels, choices))
                + ", which one is correct? Answer only with one of the following A, B, C, D or E. [/INST]{tokenizer.eos_token}"
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
            count += 1
        else:
            break
print("Processed first 5 examples.")
#out_txt = "\n".join(decoded)
#with open("generated_text.txt", "w") as file:
#    file.write(out_txt)
#print("Output")
#print(decoded[0])