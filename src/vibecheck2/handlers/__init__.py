"""Benchmark-specific strategies built ON TOP of the core (design 3.6).

A handler may pre/post-process an instance or replace the pipeline for a
benchmark the core intentionally does not model (discrete grids, quantized
surrogates, output inversion), but every verdict still flows through the
same results-file discipline and every sat through the ORT chokepoint.
"""
