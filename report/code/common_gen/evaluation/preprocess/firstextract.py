import re


# Function to read and parse the file
def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    target_sentences = []
    generated_sentences = []

    # Pattern to match target sentences
    target_pattern = re.compile(r"Target: (.*?)\nGenerated Output:", re.DOTALL)
    # Pattern to match final sentences
    generated_pattern = re.compile(r"Final sentence: (.*?)</s>", re.DOTALL)

    # Find all matches for target sentences
    target_matches = target_pattern.findall(content)
    # Find all matches for final sentences
    generated_matches = generated_pattern.findall(content)

    # Check if the number of matches is equal
    if len(target_matches) != len(generated_matches):
        print("Warning: Mismatch between the number of target and final sentences.")

    # Add matched pairs to the lists
    for target, generated in zip(target_matches, generated_matches):
        target_sentences.append(target.strip())
        generated_sentences.append(generated.strip())

    return target_sentences, generated_sentences


# Paths to your input and output files
input_file_path = 'ps_commmongen.txt'
output_pairs_file_path = 'your_output_pairs_ps_file.txt'

# Parse the file to get the target and generated sentences
target_sentences, generated_sentences = parse_file(input_file_path)

# Write the sentence pairs to the new file and count the pairs
pair_count = 0
with open(output_pairs_file_path, 'w') as file:
    for target, generated in zip(target_sentences, generated_sentences):
        file.write(f"Target: {target}\n")
        file.write(f"Generated: {generated}\n")
        file.write("\n")
        pair_count += 1

print(f"Number of sentence pairs: {pair_count}")
print(f"Pairs of sentences have been written to {output_pairs_file_path}")
