import re
import csv


def extract_info_from_entry(entry):
    # Extract Question ID
    id_pattern = r'Question ID:\s*([a-f0-9]+)'
    question_id_match = re.search(id_pattern, entry)
    question_id = question_id_match.group(1) if question_id_match else None

    # Extract Correct Answer Key
    correct_answer_pattern = r'Correct Answer Key:\s*([A-E])'
    correct_answer_match = re.search(correct_answer_pattern, entry)
    correct_answer = correct_answer_match.group(1) if correct_answer_match else None

    # Extract Generated Output Answer
    answer_pattern = r'{tokenizer\.eos_token}\s*([A-E])'
    answer_match = re.search(answer_pattern, entry)
    if not answer_match:
        # Try to capture more descriptive answers if the simple pattern fails
        answer_pattern_extended = r'{tokenizer\.eos_token}\s*([A-E]):[^.]*\.\s*'
        answer_match = re.search(answer_pattern_extended, entry)
    answer = answer_match.group(1) if answer_match else 'None'

    return question_id, correct_answer, answer


def process_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split content into individual entries
    entries = content.split('Question ID:')[1:]  # Ignore the first split part before the first "Question ID:"
    entries = ['Question ID:' + entry for entry in entries]  # Add back "Question ID:" to each entry

    # Extract information from each entry
    results = []
    correct_count = 0
    total_count = 0
    for entry in entries:
        question_id, correct_answer, answer = extract_info_from_entry(entry)
        if question_id:
            results.append({
                'Question ID': question_id,
                'Correct Answer Key': correct_answer,
                'Generated Output Answer': answer
            })
            total_count += 1
            if answer == correct_answer:
                correct_count += 1

    # Calculate percentage of correct answers
    correct_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0

    return results, correct_percentage


def save_to_csv(results, csv_file_path):
    # Define CSV column headers
    headers = ['Question ID', 'Correct Answer Key', 'Generated Output Answer']

    # Write to CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=headers)
        csvwriter.writeheader()
        for result in results:
            csvwriter.writerow(result)


# Example usage
file_path = 'output_zeroCoT.txt'
csv_file_path = 'out.csv'

results, correct_percentage = process_text_file(file_path)

# Save results to CSV
save_to_csv(results, csv_file_path)

# Print the results and the percentage of correct answers
for result in results:
    print(
        f"Question ID: {result['Question ID']}, Correct Answer Key: {result['Correct Answer Key']}, Generated Output Answer: {result['Generated Output Answer']}")

print(f"Percentage of Correct Answers: {correct_percentage:.2f}%")
