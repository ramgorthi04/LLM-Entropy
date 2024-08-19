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
    
    logits = outputs.scores
    generated_tokens = outputs.sequences[0][len(inputs['input_ids'][0]):]  

    entropies = []

    for i, logit in enumerate(logits):
        probabilities = torch.softmax(logit, dim=-1).detach().numpy()
        entropy = calculate_entropy(probabilities)
        entropies.append((generated_tokens[i].item(), entropy))

    return entropies

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    prompt = "Once upon a time in a land far, far away"
    token_entropies = get_token_entropy(prompt, model, tokenizer, temperature=1.0)

    for idx, (token_id, entropy) in enumerate(token_entropies):
        token = tokenizer.decode([token_id])
        print(f"Generated Token {idx + 1}: '{token}', Entropy: {entropy:.4f}")

    avg_entropy = np.mean([entropy for _, entropy in token_entropies])
    print(f"Average Entropy: {avg_entropy:.4f}")

if __name__ == "__main__":
    main()
