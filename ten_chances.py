from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


MODEL_NAME = "mlx-community/OLMoE-1B-7B-0125-Instruct-8bit"
SYSTEM_PROMPT = (
	"You are a helpful assistant. Answer the user's question directly and briefly."
)
USER_PROMPT = (
	"Question: I want to wash my car. The car wash is 50 meters away. "
	"Should I walk or drive?"
)
TEMPERATURE = 0.9
MAX_TOKENS = 100
NUM_TRIALS = 10


def ask_yes_no(prompt: str) -> bool:
	while True:
		answer = input(prompt).strip().upper()
		if answer in {"Y", "YES"}:
			return True
		if answer in {"N", "NO"}:
			return False
		print("Please enter Y or N.")


def main() -> None:
	model, tokenizer = load(MODEL_NAME)
	chat_prompt = tokenizer.apply_chat_template(
		[
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": USER_PROMPT},
		],
		add_generation_prompt=True,
		tokenize=False,
	)

	correct_count = 0

	print(f"System prompt: {SYSTEM_PROMPT}")
	print(f"User prompt: {USER_PROMPT}")
	print(f"Running {NUM_TRIALS} generations at temperature {TEMPERATURE}.\n")

	for trial in range(1, NUM_TRIALS + 1):
		response = generate(
			model,
			tokenizer,
			chat_prompt,
			sampler=make_sampler(temp=TEMPERATURE),
			max_tokens=MAX_TOKENS,
		)

		print(f"Trial {trial}: {response.strip()}")
		if ask_yes_no("Was that answer correct? [Y/N]: "):
			correct_count += 1
		print()

	print(f"Correct: {correct_count}/{NUM_TRIALS}")


if __name__ == "__main__":
	main()
