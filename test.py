from mlx_lm import load
import mlx.core as mx

model, tokenizer = load("mlx-community/OLMoE-1B-7B-0125-8bit")

prompt = "Question: I want to wash my car. The car wash is 50 meters away. Should I walk or drive?\nAnswer:"
tokens = mx.array(tokenizer.encode(prompt))
reference_answer = "drive"

walk_token = tokenizer.encode(" walk")[0]
drive_token = tokenizer.encode(" drive")[0]

logits = model(tokens[None, :])
next_token_logits = logits[0, -1]
probs = mx.softmax(next_token_logits, axis=-1)

walk_prob = probs[walk_token].item()
drive_prob = probs[drive_token].item()

choice_total = walk_prob + drive_prob
normalized_walk_prob = walk_prob / choice_total
normalized_drive_prob = drive_prob / choice_total

target_walk = 1.0 if reference_answer == "walk" else 0.0
target_drive = 1.0 if reference_answer == "drive" else 0.0
brier_score = (normalized_walk_prob - target_walk) ** 2 + (normalized_drive_prob - target_drive) ** 2

print(f"Prompt: {prompt}\n")
print(f"Reference answer: {reference_answer}")
print("Normalized answer choice probabilities:")
print(f"{'walk':<10} | {normalized_walk_prob:.2%} | {walk_prob:.2%}")
print(f"{'drive':<10} | {normalized_drive_prob:.2%} | {drive_prob:.2%}")
print(f"Brier score: {brier_score:.6f}")