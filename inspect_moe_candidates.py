from mlx_lm import load
import torch


def main():
    model, tokenizer = load("mlx-community/OLMoE-1B-7B-0125-8bit")

    print("Scanning model for MoE-like modules, tensors with dim==64, and hook support...\n")

    # named modules
    for name, mod in model.named_modules():
        try:
            has_hook = hasattr(mod, "register_forward_hook")
            print(f"MODULE: {name} | class={mod.__class__.__name__} | hook_support={has_hook}")
        except Exception:
            print(f"MODULE: {name} | class=<?> | hook_support=?")

    # Try PyTorch-style named parameters/buffers first, but many models won't expose them.
    print("\nNamed parameters with any dim==64 (if available):\n")
    try:
        if hasattr(model, "named_parameters"):
            for name, p in model.named_parameters():
                try:
                    if any(s == 64 for s in p.shape):
                        print(f"PARAM: {name} | shape={tuple(p.shape)} | dtype={getattr(p, 'dtype', type(p))}")
                except Exception:
                    pass
    except Exception:
        pass

    print("\nNamed buffers with any dim==64 (if available):\n")
    try:
        if hasattr(model, "named_buffers"):
            for name, b in model.named_buffers():
                try:
                    if any(s == 64 for s in b.shape):
                        print(f"BUFFER: {name} | shape={tuple(b.shape)} | dtype={getattr(b, 'dtype', type(b))}")
                except Exception:
                    pass
    except Exception:
        pass

    # Fallback: recursive attribute scan to find arrays/tensors with dim==64 and hookable modules
    print("\nRecursive attribute scan (depth-limited) for objects with dim==64 or hook support:\n")

    visited = set()

    def scan_obj(obj, prefix, depth=0, max_depth=3):
        if depth > max_depth:
            return
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        # report hook support
        try:
            if hasattr(obj, "register_forward_hook"):
                print(f"HOOKABLE: {prefix} | type={type(obj)}")
        except Exception:
            pass

        # report shape-like attributes
        try:
            if hasattr(obj, "shape"):
                shape = tuple(getattr(obj, "shape"))
                if any(s == 64 for s in shape):
                    print(f"SHAPED: {prefix} | shape={shape} | type={type(obj)}")
        except Exception:
            pass

        # drill into simple containers
        if isinstance(obj, (list, tuple, set)):
            for i, v in enumerate(obj):
                scan_obj(v, f"{prefix}[{i}]", depth + 1, max_depth)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan_obj(v, f"{prefix}[{repr(k)}]", depth + 1, max_depth)
            return

        # inspect attributes
        if depth < max_depth:
            for attr in dir(obj):
                if attr.startswith("__"):
                    continue
                low = attr.lower()
                if any(k in low for k in ("weight", "bias", "expert", "moe", "gate", "router", "gating")):
                    try:
                        val = getattr(obj, attr)
                        scan_obj(val, f"{prefix}.{attr}", depth + 1, max_depth)
                    except Exception:
                        pass

    print("\nTop-level attributes on model with 'moe'/'router'/'expert' in name:\n")
    for attr in dir(model):
        low = attr.lower()
        if any(k in low for k in ("moe", "router", "expert", "gate", "gating")):
            try:
                val = getattr(model, attr)
                print(f"ATTR: {attr} | type={type(val)}")
            except Exception:
                print(f"ATTR: {attr} | <error accessing>")

    print("\nScan complete.")


if __name__ == '__main__':
    main()
