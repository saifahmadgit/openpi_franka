"""Compare two openpi norm_stats.json files side by side.

Usage:
    python scratchpad_compare_stats.py OLD.json NEW.json
"""
import json
import sys

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=160)


def load(p):
    d = json.load(open(p))
    return d["norm_stats"] if "norm_stats" in d else d


def main(old_path, new_path):
    old, new = load(old_path), load(new_path)
    print(f"OLD: {old_path}")
    print(f"NEW: {new_path}\n")
    for key in old:
        o, n = old[key], new[key]
        for field in ("mean", "std", "q01", "q99"):
            if field not in o or field not in n:
                continue
            a, b = np.array(o[field]), np.array(n[field])
            absdiff = np.abs(a - b)
            denom = np.maximum(np.abs(a), 1e-6)
            reldiff = absdiff / denom
            print(f"=== {key}.{field} (dim={a.shape[0]}) ===")
            print("  old   :", a)
            print("  new   :", b)
            print(f"  |Δ|max={absdiff.max():.4f}  rel|Δ|max={reldiff.max()*100:.1f}%")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
