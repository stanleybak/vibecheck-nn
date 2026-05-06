"""Quick end-to-end timing on biasfield_28 with defaults."""
import time, torch, pickle, traceback
from vibecheck.settings import default_settings
from vibecheck.onnx_loader import load_onnx
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.verify_graph import verify_graph

ONNX = "/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter/onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_28.onnx"
VNN = "/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter/vnnlib/cifar_biasfield_vnncomp2022_prop_28.vnnlib"

results = []
for sa in [True, False]:
    s = default_settings()
    s.alpha_crown_sparse_alpha = sa
    s.verbose = True
    s.print_progress = True
    g = load_onnx(ONNX); g.optimize(s)
    spec = load_vnnlib(VNN)
    print(f"--- sparse={sa} default settings ---", flush=True)
    t0 = time.time()
    try:
        r, d = verify_graph(g, spec, s)
        wall = time.time() - t0
        timing = dict(d.get("timing", {}))
        phase = d.get("phase")
        remaining = d.get("remaining")
        print(f"verdict={r} wall={wall:.2f}s phase={phase} remaining={remaining}", flush=True)
        print(f"timing={timing}", flush=True)
        results.append({"sparse": sa, "verdict": r, "wall": wall,
                        "timing": timing, "phase": phase, "remaining": remaining})
    except Exception as e:
        wall = time.time() - t0
        print(f"FAILED at {wall:.2f}s: {type(e).__name__}: {str(e)[:200]}", flush=True)
        traceback.print_exc()
        results.append({"sparse": sa, "error": str(e)[:200], "wall": wall})
    torch.cuda.empty_cache()

with open("/tmp/abcrown_runs/biasfield28_endtoend_default.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSummary:", flush=True)
for r in results:
    print(f"  {r}", flush=True)
