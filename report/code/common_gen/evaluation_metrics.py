import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Ensure you have downloaded required resources
nltk.download('punkt')

# Smoothing function for BLEU score
from nltk.translate.bleu_score import SmoothingFunction
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
    P, R, F1 = bert_score([generated_sentence], [reference_sentence], lang='en')
    return F1.mean().item()