"""vibecheck2: clean-slate verifier core (see docs/clean_slate_design.md).

Reuses the v1 frontend (ONNX loading, VNNLIB parsing, CE validation) and
reimplements the middle: one IR, one forward propagator, one backward
propagator, one attack engine, one BaB search. Soundness > design > size >
speed, in that order.
"""
