"""GPU-batched dual-ascent node bound — fast drop-in for the Phase-8 BaB.

Vendored from the standalone deliverable (see HANDOFF.md). `Verifier.verify_query(
state, qw, qb, scored_keys, time_limit=...)` takes exactly the per-disjunct inputs
the legacy `dual_ascent_bab.verify_query_dual_ascent_bab` takes, and returns
`(verdict, info)` with `verdict in {'unsat'(robust), 'unknown'}`. It computes the
same per-node α-zonotope LP bound (`min c0 + d·e` over the box + branch half-spaces;
certify `unsat` iff min>0) but ~4.7× faster, with a warm-start dual step carried to
children and a sort-free log-bucket line search so the kernel is torch.compile-fused.

Soundness is structural: `g(λ) ≤ LP_min` for any `λ≥0`, so a positive bound always
certifies. TF32 is forced off (it flips node decisions). The verifier finds NO
counterexamples — SAT detection stays with vibecheck's PGD/witness machinery, which
already gates on `disable_sat_finding`.
"""
from .fast_verify_topk import Verifier
from .fast_verify_dual import parse_problem, parse_problem_gpu, Problem

__all__ = ['Verifier', 'parse_problem', 'parse_problem_gpu', 'Problem']
