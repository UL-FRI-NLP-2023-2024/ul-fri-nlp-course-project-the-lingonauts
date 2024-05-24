import re
import pandas as pd

# Load the text file
file_path = 'your_output_pairs_baseline_file.txt'
with open(file_path, 'r') as file:
    data = file.read()

# Extract Target and Final sentence using regular expressions
pattern = r'Target: (.*?)\n.*?Final sentence: (.*?)\n'
matches = re.findall(pattern, data, re.DOTALL)

# Create a DataFrame to store the results
df_results = pd.DataFrame(matches, columns=['Target', 'Final Sentence'])

# Count the number of pairs
num_pairs = len(df_results)
print(f'Total number of pairs: {num_pairs}')

# Save the DataFrame to a CSV file
output_file_path = 'target_final_sentences_baseline.csv'
df_results.to_csv(output_file_path, index=False)

print(f'Target and Final Sentences saved to {output_file_path}')
