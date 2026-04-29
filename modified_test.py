from mlx_lm import load
import mlx.core as mx
import torch
import numpy as np


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


def find_target_gate_modules(model):
    root = getattr(model, "model", model)
    gate_modules = []
    try:
        layers = list(getattr(root, "layers", []))
    except Exception:
        layers = []

    for layer in layers[::-1]:
        try:
            gate_modules.append(layer.mlp.switch_mlp.gate_proj)
            break
        except Exception:
            try:
                gate_modules.append(layer.mlp.gate)
                break
            except Exception:
                continue

    return gate_modules


def run_trial(model, tokens, walk_token, drive_token, gate_modules, expert_idx):
    target_ids = {id(module) for module in gate_modules}
    patched_classes = []
    trial_hits = {"count": 0}

    unique_classes = []
    for module in gate_modules:
        cls = module.__class__
        if cls not in unique_classes:
            unique_classes.append(cls)

    def make_class_wrapper(orig_call, idx, ids, hit_counter):
        def wrapped(self, *args, **kwargs):
            out = orig_call(self, *args, **kwargs)
            if id(self) in ids:
                hit_counter["count"] += 1
                try:
                    if isinstance(out, torch.Tensor):
                        if out.shape[-1] >= 64:
                            out = out.clone()
                            out[..., idx] = out[..., idx] + 100.0
                    else:
                        arr = np.array(out)
                        if arr.shape[-1] >= 64:
                            arr = arr.copy()
                            arr[..., idx] = arr[..., idx] + 100.0
                            out = mx.array(arr)
                except Exception:
                    pass
            return out

        return wrapped

    for cls in unique_classes:
        original_call = cls.__call__
        cls.__call__ = make_class_wrapper(original_call, expert_idx, target_ids, trial_hits)
        patched_classes.append((cls, original_call))

    try:
        logits = model(tokens[None, :])
        next_token_logits = logits[0, -1]
        probs = logits_to_probs(next_token_logits)

        if isinstance(probs, torch.Tensor):
            walk_prob = probs[walk_token].item()
            drive_prob = probs[drive_token].item()
        else:
            walk_prob = float(probs[walk_token])
            drive_prob = float(probs[drive_token])

        choice_total = walk_prob + drive_prob
        if choice_total > 0.0:
            normalized_walk_prob = walk_prob / choice_total
            normalized_drive_prob = drive_prob / choice_total
        else:
            normalized_walk_prob = 0.0
            normalized_drive_prob = 0.0

        return (
            expert_idx,
            normalized_walk_prob,
            normalized_drive_prob,
            walk_prob,
            drive_prob,
            trial_hits["count"],
        )

    finally:
        for cls, original_call in patched_classes:
            try:
                cls.__call__ = original_call
            except Exception:
                pass


def main():
    model, tokenizer = load("mlx-community/OLMoE-1B-7B-0125-8bit")

    prompt = "Question: I want to wash my car. The car wash is 50 meters away. Should I walk or drive?\nAnswer:"
    tokens = mx.array(tokenizer.encode(prompt))

    walk_token = tokenizer.encode(" Walk")[0]
    drive_token = tokenizer.encode(" Drive")[0]

    gate_modules = find_target_gate_modules(model)
    if not gate_modules:
        print("Could not locate gate projection modules to patch. Running unmodified runs.")

    results = []
    for expert_idx in range(64):
        results.append(run_trial(model, tokens, walk_token, drive_token, gate_modules, expert_idx))

    print(f"Prompt: \"{prompt}\"\n")
    print("Trial | expert_idx | hits | walk% | drive% | walk_raw% | drive_raw%")
    for i, nw, nd, wraw, draw, hits in results:
        print(f"{i:02d}    | {i:02d}         | {hits:>4} | {nw:.2%} | {nd:.2%} | {wraw:.2%}   | {draw:.2%}")


if __name__ == "__main__":
    main()

