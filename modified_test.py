from mlx_lm import load
import mlx.core as mx
import torch
import numpy as np

BONUS_FACTOR = 4.5


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
    trial_stats = {"count": 0, "orig_score": None, "orig_weight": None, "modified_weight": None}

    unique_classes = []
    for module in gate_modules:
        cls = module.__class__
        if cls not in unique_classes:
            unique_classes.append(cls)

    def make_class_wrapper(orig_call, idx, ids, stats):
        def wrapped(self, *args, **kwargs):
            out = orig_call(self, *args, **kwargs)
            if id(self) in ids:
                stats["count"] += 1
                try:
                    if isinstance(out, torch.Tensor):
                        if out.shape[-1] >= 64:
                            # Capture original score BEFORE bonus
                            if stats["orig_score"] is None:
                                stats["orig_score"] = float(out[..., idx].reshape(-1)[-1].item())
                            # Calculate softmax weight BEFORE bonus
                            if stats["orig_weight"] is None:
                                orig_softmax = torch.softmax(out, dim=-1)
                                stats["orig_weight"] = float(orig_softmax[..., idx].reshape(-1)[-1].item())
                            # Add bonus and calculate softmax weight AFTER bonus
                            out = out.clone()
                            out[..., idx] = out[..., idx] + BONUS_FACTOR
                            if stats["modified_weight"] is None:
                                modified_softmax = torch.softmax(out, dim=-1)
                                stats["modified_weight"] = float(modified_softmax[..., idx].reshape(-1)[-1].item())
                    else:
                        arr = np.array(out)
                        if arr.shape[-1] >= 64:
                            # Capture original score BEFORE bonus
                            if stats["orig_score"] is None:
                                stats["orig_score"] = float(np.ravel(arr[..., idx])[-1])
                            # Calculate softmax weight BEFORE bonus (numerically stable)
                            if stats["orig_weight"] is None:
                                arr_max = np.max(arr, axis=-1, keepdims=True)
                                orig_exp = np.exp(arr - arr_max)
                                orig_softmax = orig_exp / np.sum(orig_exp, axis=-1, keepdims=True)
                                stats["orig_weight"] = float(np.ravel(orig_softmax[..., idx])[-1])
                            # Add bonus and calculate softmax weight AFTER bonus (numerically stable)
                            arr = arr.copy()
                            arr[..., idx] = arr[..., idx] + BONUS_FACTOR
                            if stats["modified_weight"] is None:
                                arr_max = np.max(arr, axis=-1, keepdims=True)
                                modified_exp = np.exp(arr - arr_max)
                                modified_softmax = modified_exp / np.sum(modified_exp, axis=-1, keepdims=True)
                                stats["modified_weight"] = float(np.ravel(modified_softmax[..., idx])[-1])
                            out = mx.array(arr)
                except Exception:
                    pass
            return out

        return wrapped

    for cls in unique_classes:
        original_call = cls.__call__
        cls.__call__ = make_class_wrapper(original_call, expert_idx, target_ids, trial_stats)
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
            trial_stats["count"],
            trial_stats["orig_score"],
            trial_stats["orig_weight"],
            trial_stats["modified_weight"],
        )

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

gate_modules = find_target_gate_modules(model)
if not gate_modules:
    print("Could not locate gate projection modules to patch. Running unmodified runs.")

results = []
for expert_idx in range(64):
    results.append(run_trial(model, tokens, walk_token, drive_token, gate_modules, expert_idx))

print(f"Prompt: \"{prompt}\"\n")

# Find top-8 experts by modified weight
sorted_results = sorted(enumerate(results), key=lambda x: x[1][8] if x[1][8] is not None else 0, reverse=True)
top_8_indices = [idx for idx, _ in sorted_results[:8]]
top_8_results = [(idx, results[idx]) for idx in top_8_indices]

# Calculate total weights for renormalization
total_orig_weight = sum(result[6] for _, result in top_8_results if result[6] is not None)
total_modified_weight = sum(result[8] for _, result in top_8_results if result[8] is not None)

print("eid |  drive | walk_raw | drive_raw | orig_score | orig_weight | orig_weight(renorm) | modified_weight | modified_weight(renorm)")
for expert_idx, result in top_8_results:
    i, nw, nd, wraw, draw, hits, orig_score, orig_weight, modified_weight = result
    score_str = "n/a" if orig_score is None else f"{orig_score:>10.4f}"
    
    # Raw weights
    if orig_weight is not None:
        orig_weight_raw_str = f"{orig_weight:>11.4%}"
    else:
        orig_weight_raw_str = "n/a"
    
    if modified_weight is not None:
        modified_weight_raw_str = f"{modified_weight:>15.4%}"
    else:
        modified_weight_raw_str = "n/a"
    
    # Renormalize within top-8
    if orig_weight is not None and total_orig_weight > 0:
        renorm_orig = orig_weight / total_orig_weight
        orig_weight_str = f"{renorm_orig:>18.4%}"
    else:
        orig_weight_str = "n/a"
    
    if modified_weight is not None and total_modified_weight > 0:
        renorm_modified = modified_weight / total_modified_weight
        modified_weight_str = f"{renorm_modified:>20.4%}"
    else:
        modified_weight_str = "n/a"
    
    print(
        f"{i:02d}  | {nd:>6.2%} | {wraw:>8.2%} | {draw:>9.2%} | {score_str} | {orig_weight_raw_str} | {orig_weight_str} | {modified_weight_raw_str} | {modified_weight_str}"
    )

