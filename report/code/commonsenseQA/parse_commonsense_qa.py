import csv
import re

# Input and output file paths
input_file_path = 'outputs/output_ps.txt'
output_file_path = 'parsed_ps.csv'


# Regular expressions to match relevant sections
question_id_re = re.compile(r"Question ID:\s*(\w+)")
correct_answer_key_re = re.compile(r"Correct Answer Key:\s*([A-E])")
generated_output_re = re.compile(r"Generated Output\:.*?{tokenizer\.eos_token}(.*?)[^\"]</s>", re.DOTALL)
answer_key_re = re.compile(r"answer.*? is:?\s*[\"\(\']*([A-E])")
answer_would_be_re = re.compile(r"(?:answer (?:must|would) be \"?([A-E]))")
answer_re = re.compile(r"[A,a]nswer:\s*\"?([A-E])")
direct_answer_re = re.compile(r"\n\s?([A-E])[\:\.][\s\w]+")
i_would_select_re = re.compile(r"I would select answer \"?([A-E])")
option_re = re.compile(r"(?:\"?([A-E]).*? correct)")
last_line_re = re.compile(r"([A-E])[\:\.]|(?:option|choice)\s([A-E])")
none_of_the_above_re = re.compile(r"none of the above", re.IGNORECASE)

# Read input file content
with open(input_file_path, 'r') as infile:
    content = infile.read()

# Find all question blocks
question_blocks = content.split('Question ID: ')[1:]  # Skip the first split as it will be empty

# Open the output CSV file for writing
with open(output_file_path, 'w', newline='') as outfile:
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(['Question ID', 'Correct Answer Key', 'Answer Key'])
    
    # Process each question block
    for block in question_blocks:
        # Extract Question ID
        #question_id_match = question_id_re.search(block)
        #question_id = question_id_match.group(1) if question_id_match else "error"
        question_id = block.split()[0] if block.split() else "error"
        
        # Extract Correct Answer Key
        correct_answer_key_match = correct_answer_key_re.search(block)
        correct_answer_key = correct_answer_key_match.group(1) if correct_answer_key_match else "error"
        
        # Extract Generated Output
        generated_output_match = generated_output_re.search(block)
        if generated_output_match:
            generated_output = "\n" + generated_output_match.group(1)
            
            # Extract the final line that contains </s>
            first_line = generated_output.strip().splitlines()[0]
            final_line = generated_output.strip().splitlines()[-1]
            
            # Extract Answer Key from the final line
            answer_key_match = answer_key_re.search(generated_output)
            if answer_key_match:
                answer_key = answer_key_match.group(1)
            else:
                answer_match = answer_re.search(generated_output)
                if answer_match:
                    answer_key = answer_match.group(1)
                else:
                    direct_answer_match = direct_answer_re.search(generated_output)
                    if direct_answer_match:
                        answer_key = direct_answer_match.group(1)
                    else:
                        answer_would_be_match = answer_would_be_re.search(generated_output)
                        if answer_would_be_match:
                            answer_key = answer_would_be_match.group(1)
                        else:
                            i_would_select_match = i_would_select_re.search(final_line)
                            if i_would_select_match:
                                answer_key = i_would_select_match.group(1)
                            else:
                                last_line_match = last_line_re.search(final_line)
                                if last_line_match:
                                    if last_line_match.group(1):
                                        answer_key = last_line_match.group(1)
                                    else:
                                        answer_key = last_line_match.group(2)
                                else:
                                    none_of_the_above_match = none_of_the_above_re.search(final_line)
                                    if none_of_the_above_match:
                                        answer_key = "none"
                                    else:
                                        option_match = option_re.search(final_line)
                                        if option_match:
                                            answer_key = option_match.group(1)
                                        else:
                                            answer_key = generated_output
        else:
            answer_key = "error"
        
        # Write the extracted information to the CSV file
        csv_writer.writerow([question_id, correct_answer_key, answer_key])