"""Minimal ONNX-to-torch interpreter for forward-only execution.

Handles enough ops for cgan small_transformer. Used to run PGD when
gpu_graph can't (e.g., bilinear MatMul + attention shapes don't fit
the flat-tensor abstraction). Each op type maps to its torch equivalent;
runtime maintains a name → tensor dict.
"""
import numpy as np
import torch
import torch.nn.functional as F
import onnx
from onnx import numpy_helper


def _torch_op(op_type, inputs, attrs):
    """Apply one onnx op to torch tensors. inputs: list of tensors."""
    if op_type == 'Relu':
        return F.relu(inputs[0])
    if op_type == 'Sigmoid':
        return torch.sigmoid(inputs[0])
    if op_type == 'Tanh':
        return torch.tanh(inputs[0])
    if op_type == 'Add':
        return inputs[0] + inputs[1]
    if op_type == 'Sub':
        return inputs[0] - inputs[1]
    if op_type == 'Mul':
        return inputs[0] * inputs[1]
    if op_type == 'Div':
        return inputs[0] / inputs[1]
    if op_type == 'Neg':
        return -inputs[0]
    if op_type == 'MatMul':
        return inputs[0] @ inputs[1]
    if op_type == 'Gemm':
        a = float(attrs.get('alpha', 1.0))
        b = float(attrs.get('beta', 1.0))
        transA = bool(attrs.get('transA', 0))
        transB = bool(attrs.get('transB', 0))
        A = inputs[0].t() if transA else inputs[0]
        B = inputs[1].t() if transB else inputs[1]
        out = a * (A @ B)
        if len(inputs) > 2:
            out = out + b * inputs[2]
        return out
    if op_type == 'Conv':
        x = inputs[0]; w = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        strides = attrs.get('strides', [1, 1])
        pads = attrs.get('pads', [0, 0, 0, 0])
        return F.conv2d(x, w, bias=bias, stride=tuple(strides),
                          padding=(pads[0], pads[1]))
    if op_type == 'ConvTranspose':
        x = inputs[0]; w = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        strides = attrs.get('strides', [1, 1])
        pads = attrs.get('pads', [0, 0, 0, 0])
        opads = attrs.get('output_padding', [0, 0])
        return F.conv_transpose2d(x, w, bias=bias, stride=tuple(strides),
                                    padding=(pads[0], pads[1]),
                                    output_padding=tuple(opads))
    if op_type == 'BatchNormalization':
        x = inputs[0]; scale = inputs[1]; bias = inputs[2]
        mean = inputs[3]; var = inputs[4]
        eps = float(attrs.get('epsilon', 1e-5))
        return F.batch_norm(x, mean, var, scale, bias, training=False, eps=eps)
    if op_type == 'Upsample':
        x = inputs[0]
        scales = inputs[1] if len(inputs) > 1 else None
        if scales is None or scales.numel() == 0:
            return x
        s = scales.tolist()
        # ONNX 4D NCHW: scales = [1, 1, sH, sW]
        return F.interpolate(x, scale_factor=(s[2], s[3]),
                              mode=attrs.get('mode', 'nearest'))
    if op_type == 'Resize':
        x = inputs[0]
        # inputs: x, roi (unused), scales, sizes
        scales = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
        if scales is not None and scales.numel() > 0:
            s = scales.tolist()
            return F.interpolate(x, scale_factor=(s[2], s[3]),
                                  mode=attrs.get('mode', 'nearest'))
        sizes = inputs[3] if len(inputs) > 3 else None
        if sizes is not None and sizes.numel() > 0:
            return F.interpolate(x, size=(int(sizes[2]), int(sizes[3])),
                                  mode=attrs.get('mode', 'nearest'))
        return x
    if op_type == 'Reshape':
        shape = inputs[1].tolist()
        return inputs[0].reshape(shape)
    if op_type == 'Transpose':
        perm = attrs.get('perm')
        return inputs[0].permute(*perm) if perm else inputs[0].t()
    if op_type == 'Squeeze':
        axes = attrs.get('axes') or list(inputs[1].tolist()) if len(inputs) > 1 else None
        if axes:
            for a in sorted(axes, reverse=True):
                inputs[0] = inputs[0].squeeze(a)
            return inputs[0]
        return inputs[0].squeeze()
    if op_type == 'Unsqueeze':
        axes = attrs.get('axes') or (inputs[1].tolist() if len(inputs) > 1 else [])
        x = inputs[0]
        for a in sorted(axes):
            x = x.unsqueeze(a)
        return x
    if op_type == 'Softmax':
        ax = int(attrs.get('axis', -1))
        return F.softmax(inputs[0], dim=ax)
    if op_type == 'Concat':
        ax = int(attrs.get('axis', 0))
        return torch.cat(inputs, dim=ax)
    if op_type == 'Flatten':
        ax = int(attrs.get('axis', 1))
        return inputs[0].flatten(start_dim=ax)
    if op_type == 'AveragePool':
        kernel = tuple(attrs.get('kernel_shape', [2, 2]))
        strides = tuple(attrs.get('strides', kernel))
        pads = attrs.get('pads', [0, 0, 0, 0])
        return F.avg_pool2d(inputs[0], kernel, strides, (pads[0], pads[1]))
    if op_type == 'MaxPool':
        kernel = tuple(attrs.get('kernel_shape', [2, 2]))
        strides = tuple(attrs.get('strides', kernel))
        pads = attrs.get('pads', [0, 0, 0, 0])
        return F.max_pool2d(inputs[0], kernel, strides, (pads[0], pads[1]))
    if op_type == 'Pad':
        # ONNX 11+ pads is a tensor input
        if len(inputs) > 1:
            pads = inputs[1].tolist()
        else:
            pads = attrs.get('pads', [0] * (inputs[0].ndim * 2))
        # ONNX pads: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        # torch.nn.functional.pad takes [last_dim_left, last_dim_right, ...]
        ndim = inputs[0].ndim
        torch_pads = []
        for d in range(ndim - 1, -1, -1):
            torch_pads.extend([pads[d], pads[d + ndim]])
        mode = attrs.get('mode', 'constant')
        if isinstance(mode, bytes): mode = mode.decode()
        val = float(inputs[2].item()) if len(inputs) > 2 else 0.0
        return F.pad(inputs[0], torch_pads, mode=mode, value=val)
    if op_type == 'Identity':
        return inputs[0]
    if op_type == 'Cast':
        # Already torch tensor, just keep dtype (downstream ops handle).
        return inputs[0]
    if op_type == 'Constant':
        # Constant node — handled outside (returns the embedded tensor).
        for k, v in attrs.items():
            if k == 'value':
                return torch.as_tensor(numpy_helper.to_array(v))
        return torch.tensor(0.0)
    if op_type == 'Shape':
        return torch.tensor(list(inputs[0].shape))
    if op_type == 'Gather':
        axis = int(attrs.get('axis', 0))
        return inputs[0].index_select(axis, inputs[1].long())
    if op_type == 'Sin':
        return torch.sin(inputs[0])
    if op_type == 'Cos':
        return torch.cos(inputs[0])
    if op_type == 'Pow':
        return torch.pow(inputs[0], inputs[1])
    if op_type == 'Floor':
        return torch.floor(inputs[0])
    if op_type == 'Equal':
        return torch.eq(inputs[0], inputs[1])
    if op_type == 'Where':
        return torch.where(inputs[0].bool(), inputs[1], inputs[2])
    if op_type == 'Expand':
        # ONNX Expand: bidirectional broadcast of data to `shape`.
        shape = [int(s) for s in inputs[1].tolist()]
        return inputs[0] * torch.ones(shape, dtype=inputs[0].dtype,
                                      device=inputs[0].device)
    if op_type == 'ConstantOfShape':
        shape = [int(s) for s in inputs[0].tolist()]
        v = attrs.get('value')
        fill = float(numpy_helper.to_array(v).flatten()[0]) if v is not None else 0.0
        return torch.full(shape, fill)
    if op_type == 'Slice':
        # ONNX 10+: starts/ends/axes/steps are tensor inputs; <10: attributes.
        data = inputs[0]
        if len(inputs) > 1:
            starts = [int(s) for s in inputs[1].tolist()]
            ends = [int(e) for e in inputs[2].tolist()]
            axes = ([int(a) for a in inputs[3].tolist()]
                    if len(inputs) > 3 and inputs[3] is not None
                    else list(range(len(starts))))
            steps = ([int(s) for s in inputs[4].tolist()]
                     if len(inputs) > 4 and inputs[4] is not None
                     else [1] * len(starts))
        else:
            starts = list(attrs.get('starts', []))
            ends = list(attrs.get('ends', []))
            axes = list(attrs.get('axes', list(range(len(starts)))))
            steps = [1] * len(starts)
        sl = [slice(None)] * data.ndim
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = ax if ax >= 0 else ax + data.ndim
            sl[ax] = slice(s, e, st)
        return data[tuple(sl)]
    raise NotImplementedError(f'onnx_torch_runner: unsupported op {op_type!r}')


def _attrs(node):
    """ONNX attribute → python dict."""
    d = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT: d[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT: d[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING: d[a.name] = a.s.decode() if isinstance(a.s, bytes) else a.s
        elif a.type == onnx.AttributeProto.INTS: d[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS: d[a.name] = list(a.floats)
        elif a.type == onnx.AttributeProto.TENSOR: d[a.name] = a.t
    return d


def onnx_forward(model, x, device=None, dtype=None):
    """Run forward through an ONNX model. x: torch tensor of input(s).

    Returns the output tensor. Supports the cgan small_transformer
    op set (Conv, ConvTranspose, BN, Relu, Add, Mul, MatMul, Gemm,
    Upsample, AveragePool, MaxPool, Softmax, Tanh, Sigmoid, Reshape,
    Transpose, Squeeze, Concat, Pad, Cast, etc.)."""
    device = device or x.device
    dtype = dtype or x.dtype
    g = model.graph
    # Initializers
    vals = {}
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        vals[init.name] = torch.as_tensor(arr, device=device,
                                            dtype=dtype if arr.dtype.kind == 'f' else None)
    # Input. Old ONNX exporters (e.g. acasxu) also list every initializer in
    # g.input, so g.input[0] may be an initializer rather than the real data
    # input. Pick the one g.input entry that is NOT an initializer.
    _init_names = {init.name for init in g.initializer}
    _real_inputs = [i.name for i in g.input if i.name not in _init_names]
    inp_name = _real_inputs[0] if _real_inputs else g.input[0].name
    vals[inp_name] = x
    # Run nodes in order
    for node in g.node:
        ins = []
        for n in node.input:
            if n in vals:
                ins.append(vals[n])
            elif n == '':
                ins.append(None)
            else:
                raise KeyError(f'missing input {n!r} for op {node.op_type!r}')
        attrs = _attrs(node)
        if node.op_type == 'Constant':
            out = _torch_op(node.op_type, ins, attrs)
            out = out.to(device=device, dtype=dtype if out.dtype.is_floating_point else out.dtype)
            vals[node.output[0]] = out
            continue
        out = _torch_op(node.op_type, ins, attrs)
        if isinstance(out, (list, tuple)):
            for nm, o in zip(node.output, out):
                vals[nm] = o
        else:
            vals[node.output[0]] = out
    return vals[g.output[0].name]


def pgd_via_onnx(onnx_path, spec, n_restarts=256, n_iter=100, lr=0.1,
                 device=None, dtype=None, simplify=True, min_restarts=8):
    """Run PGD against the raw ONNX model. Returns (sat: bool, witness or None).

    Used as a last-resort SAT-finder when gpu_graph can't run forward
    (e.g., transformer attention with shape-sensitive ops). OOM-halves
    n_restarts down to min_restarts before giving up.
    """
    import gzip, time
    import onnx as _onnx
    if onnx_path.endswith('.gz'):
        with gzip.open(onnx_path, 'rb') as f:
            model = _onnx.load_model_from_string(f.read())
    else:
        model = _onnx.load(onnx_path)
    if simplify:
        try:
            import onnxsim
            model_s, ok = onnxsim.simplify(model)
            if ok:
                model = model_s
        except (ImportError, RuntimeError):
            # onnxsim absent or failed; fall back to unsimplified model.
            pass
    device = device or (torch.device('cuda') if torch.cuda.is_available()
                        else torch.device('cpu'))
    dtype = dtype or torch.float32
    xl = torch.as_tensor(spec.x_lo, dtype=dtype, device=device).flatten()
    xh = torch.as_tensor(spec.x_hi, dtype=dtype, device=device).flatten()
    n_in = xl.numel()
    width = xh - xl
    # Build per-disjunct (W, b): margin = y @ W^T + b; SAT iff some disjunct
    # has all margins <= 0.
    Ws, bs = [], []
    for conj in spec.disjuncts:
        n_c = len(conj.constraints)
        # Infer n_out by probing with the lower-bound input.
        if not Ws:
            with torch.no_grad():
                try:
                    y0 = onnx_forward(model, xl.reshape(1, *_infer_input_shape(model)),
                                       device=device, dtype=dtype)
                except (RuntimeError, ValueError):
                    # RuntimeError: shape inferred from ONNX metadata didn't
                    # match the runtime input (e.g. dynamic batch dim).
                    # ValueError: reshape size mismatch. Both indicate the
                    # input shape probe failed — fall back to flat-vector
                    # shape which works for fully-connected nets.
                    y0 = onnx_forward(model, xl.unsqueeze(0), device=device, dtype=dtype)
                n_out = int(y0.numel())
        W = torch.zeros(n_c, n_out, dtype=dtype, device=device)
        b = torch.zeros(n_c, dtype=dtype, device=device)
        for i, c in enumerate(conj.constraints):
            if hasattr(c, 'pred'):
                W[i, c.pred] = 1.0; W[i, c.comp] = -1.0
            elif c.op == '>=':
                W[i, c.index] = -1.0; b[i] = float(c.value)
            elif c.op == '<=':
                W[i, c.index] = 1.0; b[i] = -float(c.value)
        Ws.append(W); bs.append(b)
    in_shape = _infer_input_shape(model)
    nr = n_restarts
    while nr >= min_restarts:
        try:
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            torch.manual_seed(0)
            x = xl + width * torch.rand(nr, n_in, device=device, dtype=dtype)
            x = x.detach().requires_grad_(True)
            for it in range(n_iter):
                ys = onnx_forward(model, x.reshape(nr, *in_shape),
                                   device=device, dtype=dtype)
                ys_flat = ys.reshape(nr, -1)
                # Per-disjunct max margin → min over disjuncts.
                stacked = torch.stack(
                    [(ys_flat @ W.T + b).max(dim=1).values for W, b in zip(Ws, bs)],
                    dim=1)
                min_m, _ = stacked.min(dim=1)
                with torch.no_grad():
                    # Accept only a REAL violation (margin <= 0 — the unsafe
                    # region, >= boundary inclusive). The old `<= 1e-6` accepted
                    # near-misses 1e-6 OUTSIDE the unsafe region (the witness is
                    # actually safe), producing false-`sat` on boundary cases
                    # (ml4acopf 14_ieee prop3: witness margin +1e-6, ABC=unsat,
                    # no real violation reachable). A genuine counterexample has
                    # min_m <= 0; if PGD only reaches a positive near-miss it
                    # keeps searching and ultimately returns no-sat (-> unknown),
                    # never a false-sat.
                    sat_mask = min_m <= 0.0
                    if sat_mask.any():
                        idx = int(sat_mask.nonzero()[0].item())
                        return True, x[idx].detach().cpu().numpy().reshape(spec.x_lo.shape)
                loss = min_m.sum()
                loss.backward()
                with torch.no_grad():
                    step = lr * x.grad.sign() * width
                    x = (x - step).clamp(min=xl, max=xh)
                x = x.detach().requires_grad_(True)
            return False, None
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' not in str(e).lower():
                return False, None
            nr //= 2
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return False, None


def _infer_input_shape(model):
    """Get the input shape (excluding batch dim) from an onnx model."""
    # g.input[0] may be an initializer (old exporters list them in g.input);
    # the real data input is the one without an initializer entry.
    _init = {init.name for init in model.graph.initializer}
    _real = [i for i in model.graph.input if i.name not in _init]
    inp = _real[0] if _real else model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    shape = []
    for i, d in enumerate(dims):
        if i == 0:
            continue  # batch dim
        v = d.dim_value if d.HasField('dim_value') else 1
        shape.append(v if v > 0 else 1)
    return tuple(shape) if shape else (1,)
