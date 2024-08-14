import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

def calculate_entropy(prob_dist):
    """Calculate the entropy of a probability distribution."""
    return -np.sum(prob_dist * np.log(prob_dist + 1e-9))  # Adding epsilon to avoid log(0)

def get_token_entropy(prompt, model, tokenizer, temperature=1.0):
    """Generate text from the model and calculate the entropy for each predicted token."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=True, temperature=temperature, output_scores=True, return_dict_in_generate=True, max_new_tokens=50)
    
    # The `scores` attribute contains the logits for the predicted tokens
    logits = outputs.scores
    
    entropies = []

    # Calculate the probability distribution and entropy for each generated token
    for logit in logits:
        probabilities = torch.softmax(logit, dim=-1).detach().numpy()
        entropy = calculate_entropy(probabilities)
        entropies.append(entropy)

    return entropies

def main():
    # Load the pre-trained GPT-2 model and tokenizer (GPT-3/4 access might require API)
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    prompt = "Once upon a time in a land far, far away"
    entropies = get_token_entropy(prompt, model, tokenizer, temperature=1.0)

    # Print the entropy of each predicted token
    for idx, entropy in enumerate(entropies):
        print(f"Generated Token {idx + 1}, Entropy: {entropy:.4f}")

    # Print the average entropy
    avg_entropy = np.mean(entropies)
    print(f"Average Entropy: {avg_entropy:.4f}")

if __name__ == "__main__":
    main()
