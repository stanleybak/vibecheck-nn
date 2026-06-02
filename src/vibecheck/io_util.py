"""Shared I/O helpers for the ONNX / VNNLIB loaders."""

import gzip
import os
import shutil


def ensure_decompressed(path):
    """Return a path to a usable, non-gzipped copy of ``path``.

    VNNCOMP benchmark files often ship gzipped (``foo.onnx.gz``) while the
    ``instances.csv`` references the decompressed name (``foo.onnx``). This
    resolver bridges that gap transparently for both spellings:

      - ``path`` is a plain file that exists -> returned unchanged.
      - ``path`` ends in ``.gz`` -> decompressed to its sibling (``path``
        without the ``.gz``).
      - ``path`` is a plain name that does NOT exist but ``path + '.gz'``
        does -> the ``.gz`` is decompressed to ``path``.

    The decompressed file is written alongside the ``.gz`` (same directory)
    and reused on later loads: if a decompressed sibling already exists and
    is at least as new as the ``.gz``, it is returned without re-inflating.
    A one-line notice is printed to stdout whenever decompression actually
    runs.

    If the directory is not writable, the ``.gz`` path is returned unchanged
    so the caller's gzip-aware loader can fall back to in-memory inflation.
    """
    if path.endswith('.gz'):
        gz_path, plain_path = path, path[:-3]
    else:
        plain_path, gz_path = path, path + '.gz'

    # Reuse an existing decompressed file when it's present and not stale
    # relative to the .gz (or when there's no .gz at all).
    if os.path.exists(plain_path) and (
            not os.path.exists(gz_path)
            or os.path.getmtime(plain_path) >= os.path.getmtime(gz_path)):
        return plain_path

    # Materialize the decompressed file next to the .gz.
    if os.path.exists(gz_path):
        print(f'Decompressing {gz_path} -> {plain_path}', flush=True)
        try:
            with gzip.open(gz_path, 'rb') as f_in, \
                    open(plain_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        except OSError:
            # Directory not writable (read-only mount, etc.). Hand back the
            # .gz so the caller decompresses it in memory instead.
            return gz_path
        return plain_path

    # No .gz to fall back on — return the original path; the caller's loader
    # raises the usual FileNotFoundError if `plain_path` is also absent.
    return path
