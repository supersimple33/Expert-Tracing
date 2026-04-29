from mlx_lm import load
import mlx.core as mx
import torch
import types
import numpy as np


model, tokenizer = load("mlx-community/OLMoE-1B-7B-0125-8bit")

prompt = "Question: I want to wash my car. The car wash is 50 meters away. Should I walk or drive?\nAnswer:"
tokens = mx.array(tokenizer.encode(prompt))

walk_token = tokenizer.encode(" Walk")[0]
drive_token = tokenizer.encode(" Drive")[0]

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

root = getattr(model, "model", model)
gate_projs = []
try:
    layers = list(getattr(root, "layers", []))
except Exception:
    layers = []

for layer in layers[::-1]:
    try:
        gp = layer.mlp.switch_mlp.gate_proj
        gate_projs.append(gp)
        break
    except Exception:
        try:
            gp = layer.mlp.gate
            gate_projs.append(gp)
            break
        except Exception:
            continue

if not gate_projs:
    print("Could not locate gate projection modules to patch. Running unmodified runs.")

results = []

for expert_idx in range(64):
    # Special methods like __call__ are resolved on the class, not the instance.
    # Patch the class __call__ temporarily and gate by object id.
    target_ids = {id(gp) for gp in gate_projs}
    patched_classes = []
    trial_hits = {"count": 0}

    unique_classes = []
    for gp in gate_projs:
        cls = gp.__class__
        if cls not in unique_classes:
            unique_classes.append(cls)

    for cls in unique_classes:
        orig_call = cls.__call__

        def make_class_wrapper(orig, idx, ids, hit_counter):
            def wrapped(self, *a, **k):
                out = orig(self, *a, **k)
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

        cls.__call__ = make_class_wrapper(orig_call, expert_idx, target_ids, trial_hits)
        patched_classes.append((cls, orig_call))

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
        results.append((expert_idx, normalized_walk_prob, normalized_drive_prob, walk_prob, drive_prob, trial_hits["count"]))

    finally:
        for cls, orig_call in patched_classes:
            try:
                cls.__call__ = orig_call
            except Exception:
                pass

print(f"Prompt: \"{prompt}\"\n")
print("Trial | expert_idx | hits | walk% | drive% | walk_raw% | drive_raw%")
for i, nw, nd, wraw, draw, hits in results:
    print(f"{i:02d}    | {i:02d}         | {hits:>4} | {nw:.2%} | {nd:.2%} | {wraw:.2%}   | {draw:.2%}")

