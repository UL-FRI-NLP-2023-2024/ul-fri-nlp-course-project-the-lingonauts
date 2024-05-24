import pandas as pd
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
from datasets import load_metric

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Smoothing function for BLEU score
smooth = SmoothingFunction().method4


def compute_bleu(generated_sentence, reference_sentence):
    reference_tokens = [nltk.word_tokenize(reference_sentence)]
    generated_tokens = nltk.word_tokenize(generated_sentence)
    return nltk.translate.bleu_score.sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smooth)


def compute_rouge(generated_sentence, reference_sentence):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_sentence, generated_sentence)
    return scores['rouge1'].fmeasure  # You can choose which ROUGE score to return


def compute_meteor(generated_sentence, reference_sentence):
    reference_tokens = nltk.word_tokenize(reference_sentence)
    generated_tokens = nltk.word_tokenize(generated_sentence)
    return nltk.translate.meteor_score.meteor_score([reference_tokens], generated_tokens)


def compute_bertscore(generated_sentence, reference_sentence):
    P, R, F1 = bert_score.score([generated_sentence], [reference_sentence], lang='en')
    return F1.mean().item()


# Load the CSV file
input_file_path = 'target_final_sentences_baseline.csv'
print("Loading CSV file...")
df = pd.read_csv(input_file_path)
print("CSV file loaded successfully.")

# Calculate scores for each pair
results = []
for index, row in df.iterrows():
    print(f"Processing row {index + 1}/{len(df)}")
    reference = row['Target']
    hypothesis = row['Final Sentence']

    print("Calculating BLEU score...")
    bleu = compute_bleu(hypothesis, reference)

    print("Calculating ROUGE score...")
    rouge = compute_rouge(hypothesis, reference)

    print("Calculating METEOR score...")
    meteor = compute_meteor(hypothesis, reference)

    print("Calculating BERTScore...")
    bertscore = compute_bertscore(hypothesis, reference)

    results.append({
        'Target': reference,
        'Final Sentence': hypothesis,
        'BLEU': bleu,
        'ROUGE': rouge,
        'METEOR': meteor,
        'BERTScore': bertscore
    })

# Convert results to DataFrame
print("Converting results to DataFrame...")
results_df = pd.DataFrame(results)
print("Conversion successful.")

# Calculate average scores
print("Calculating average scores...")
average_scores = {
    'Average BLEU': results_df['BLEU'].mean(),
    'Average ROUGE': results_df['ROUGE'].mean(),
    'Average METEOR': results_df['METEOR'].mean(),
    'Average BERTScore': results_df['BERTScore'].mean()
}

# Display the results
print("Results DataFrame:")
print(results_df)
print("\nAverage Scores:")
for score_name, score_value in average_scores.items():
    print(f"{score_name}: {score_value}")

# Save the results to a new CSV file
output_file_path = 'scored_target_final_sentences_baseline.csv'
print("Saving results to CSV file...")
results_df.to_csv(output_file_path, index=False)
print(f"Scored sentences saved to {output_file_path}")
