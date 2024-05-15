from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda" # the device to load the model onto
print("Setting device to " + device)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


# Load the dataset
dataset = load_dataset("allenai/common_gen", split="validation")
print(f"Dataset loaded with {len(dataset)} entries.")
#print(dataset[0])

with open("ps_commmongen.txt", "w") as file:
# Example of formatting prompts for a few examples
    counter = 0
    for example in dataset:  # Process just a few examples for demonstration
        if counter < 3:
            concepts = ", ".join(example['concepts'])
            prompt = f"Generate only one sentence using all of the following concepts: {concepts}. Let's first understand the concepts and devise a plan how to create a sentence. Then, let's carry out the plan to solve the problem step by step."
            

            # This is the prompt you'd send to the model
            #print(prompt)

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            model.to(device)

            generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)[0]
            print(decoded)

            file.write("Concepts set ID: " + str(example['concept_set_idx']) + "\n")
            file.write("Concepts: " + str(example['concepts']) + "\n")
            file.write("Target: " + example['target'] + "\n")
            file.write("Generated Output: " + decoded + "\n")

            counter += 1
        else:
            break