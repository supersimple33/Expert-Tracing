from mlx_lm import load
import mlx.core as mx
import torch
import torch.nn.functional as F
import numpy as np


TEMPERATURE = 0.01
GUMBEL_SEED = 0
GENERATOR = np.random.default_rng(GUMBEL_SEED)

def gumbel_logits(logits: mx.array):
    # Sample Gumbel noise
    gumbel_noise = mx.array(GENERATOR.gumbel(size=logits.shape)).astype(logits.dtype)
    return logits + (gumbel_noise * TEMPERATURE)


def logits_to_probs(logits_tensor):
    if isinstance(logits_tensor, torch.Tensor):
        return torch.nn.functional.softmax(logits_tensor, dim=-1)
    try:
        return mx.softmax(logits_tensor, axis=-1)
    except Exception:
        pass
    if hasattr(logits_tensor, "asnumpy"):
        arr = logits_tensor.asnumpy()
    elif hasattr(logits_tensor, "numpy"):
        arr = logits_tensor.numpy()
    else:
        arr = np.array(logits_tensor)
    return torch.nn.functional.softmax(torch.from_numpy(arr), dim=-1)


# Using PyTorch's gumbel_softmax for sampling; no numpy sampling needed.


def collect_router_modules(model):
    root = getattr(model, "model", model)
    router_modules = []
    try:
        layers = list(getattr(root, "layers", []))
    except Exception:
        layers = []

    for layer in layers:
        try:
            router_modules.append(layer.mlp.switch_mlp.gate_proj)
            continue
        except Exception:
            pass
        try:
            router_modules.append(layer.mlp.gate)
        except Exception:
            continue

    return router_modules


def run_with_gumbel_router(model, tokens, router_modules, temperature, seed):
    target_ids = {id(module) for module in router_modules}
    unique_classes = []
    patched_classes = []
    # Seed PyTorch RNG for reproducible Gumbel draws
    torch.manual_seed(seed)

    for module in router_modules:
        cls = module.__class__
        if cls not in unique_classes:
            unique_classes.append(cls)

    def make_class_wrapper(orig_call, ids):
        def wrapped(self, *args, **kwargs):
            out = orig_call(self, *args, **kwargs)
            if id(self) in ids:
                out = gumbel_logits(out)
            return out

        return wrapped

    for cls in unique_classes:
        original_call = cls.__call__
        cls.__call__ = make_class_wrapper(original_call, target_ids)
        patched_classes.append((cls, original_call))

    try:
        return model(tokens[None, :])
    finally:
        for cls, original_call in patched_classes:
            try:
                cls.__call__ = original_call
            except Exception:
                pass


model, tokenizer = load("mlx-community/OLMoE-1B-7B-0125-8bit")

prompt = "Question: I want to wash my car. The car wash is 50 meters away. Should I walk or drive?\nAnswer:"
tokens = mx.array(tokenizer.encode(prompt))

walk_token = tokenizer.encode(" Walk")[0]
drive_token = tokenizer.encode(" Drive")[0]

router_modules = collect_router_modules(model)
if not router_modules:
    print("Could not locate router modules to patch.")
    raise RuntimeError("No router modules found")

logits = run_with_gumbel_router(model, tokens, router_modules, TEMPERATURE, GUMBEL_SEED)
next_token_logits = logits[0, -1]
probs = logits_to_probs(next_token_logits)

if isinstance(probs, torch.Tensor):
    walk_prob = probs[walk_token].item()
    drive_prob = probs[drive_token].item()
else:
    walk_prob = float(probs[walk_token])
    drive_prob = float(probs[drive_token])

brier_score = (walk_prob - 0.0) ** 2 + (drive_prob - 1.0) ** 2
choice_total = walk_prob + drive_prob
normalized_drive_prob = drive_prob / choice_total if choice_total > 0.0 else 0.0

print(f'Prompt: "{prompt}"\n')
print(f"Temperature: {TEMPERATURE}")
print(f"Routers patched: {len(router_modules)}")
print("drive | brier | walk_raw | drive_raw")
print(f"{normalized_drive_prob:>6.2%} | {brier_score:>5.4f} | {walk_prob:>8.5%} | {drive_prob:>9.5%}")
