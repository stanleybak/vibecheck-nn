"""ONNX model loading, constant folding, and graph optimization."""

import numpy as np
import gzip

from .network import (
    ComputeGraph, GraphNode, OP_REGISTRY, _prod,
    PassthroughNode, SplitOutputNode, GemmNode, MatMulBilinearNode,
)


def load_onnx(onnx_path, dtype=None, simplify=None):
    """Load an ONNX model into a ComputeGraph."""
    import numpy as _np
    if dtype is None:
        dtype = _np.float32
    import onnx
    from onnx import numpy_helper

    # Accept gzipped inputs, and resolve a `.onnx` reference whose only
    # on-disk copy is `.onnx.gz` (common in VNNCOMP benchmark dirs).
    from .io_util import ensure_decompressed
    onnx_path = ensure_decompressed(onnx_path)

    if onnx_path.endswith('.gz'):
        with gzip.open(onnx_path, 'rb') as f:
            model = onnx.load_from_string(f.read())
    else:
        model = onnx.load(onnx_path)

    # Pre-simplify with onnxsim only when the model has patterns that
    # require it (transformer attention with 68 MatMul + 44 Div folded
    # to constants, etc.). Skipping onnxsim on FC-only nets avoids the
    # MatMul+Add → Gemm fusion that measurably loosens milp_verify
    # bounds on acasxu prop_3 (verified → unknown).
    if simplify is None:
        op_types = {n.op_type for n in model.graph.node}
        # Trigger on attention-shaped sub-graphs that our loader can't
        # eat without constant folding (Softmax + bilinear MatMul).
        simplify = ('Softmax' in op_types
                    and sum(1 for n in model.graph.node
                            if n.op_type == 'MatMul') > 4)
    if simplify:
        try:
            import onnxsim
            model_sim, ok = onnxsim.simplify(model)
            if ok:
                model = model_sim
        except (ImportError, RuntimeError):
            # ImportError: onnxsim not installed (optional dependency).
            # RuntimeError: onnxsim choked on this model (rare; happens on
            # some unsupported op combinations). In both cases, falling back
            # to the unsimplified model is sound — simplification is a perf
            # nicety, not a soundness requirement.
            pass

    graph = ComputeGraph(dtype=dtype)
    inits = {init.name: numpy_helper.to_array(init).astype(np.float64)
             for init in model.graph.initializer}

    # Identify graph input (skip initializers)
    init_names = set(inits.keys())
    for inp in model.graph.input:
        if inp.name not in init_names:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            graph.input_name = inp.name
            graph._raw_input_dims = dims
            break

    graph.output_name = model.graph.output[0].name

    # Parse Constant nodes
    constants = {}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)

    def _const(name):
        if name in inits:
            return inits[name]
        if name in constants:
            return constants[name]
        return None

    # Build nodes
    for node in model.graph.node:
        if node.op_type == 'Constant':
            continue

        out_name = node.output[0]
        computed_inputs = []
        params = {}
        op = node.op_type

        attrs = _parse_attrs(node)

        if op == 'Conv':
            computed_inputs = [node.input[0]]
            params['kernel'] = inits[node.input[1]]
            if len(node.input) > 2 and node.input[2] in inits:
                params['bias'] = inits[node.input[2]]
            else:
                params['bias'] = np.zeros(inits[node.input[1]].shape[0],
                                          dtype=np.float64)
            params['stride'] = tuple(attrs.get('strides', [1, 1]))
            pads = attrs.get('pads', [0, 0, 0, 0])
            params['padding'] = (pads[0], pads[1])
            params['group'] = attrs.get('group', 1)

        elif op == 'ConvTranspose':
            computed_inputs = [node.input[0]]
            params['kernel'] = inits[node.input[1]]
            if len(node.input) > 2 and node.input[2] in inits:
                params['bias'] = inits[node.input[2]]
            else:
                params['bias'] = np.zeros(inits[node.input[1]].shape[1],
                                          dtype=np.float64)
            params['stride'] = tuple(attrs.get('strides', [1, 1]))
            pads = attrs.get('pads', [0, 0, 0, 0])
            params['padding'] = (pads[0], pads[1])
            params['output_padding'] = tuple(attrs.get('output_padding', [0, 0]))
            params['group'] = attrs.get('group', 1)

        elif op == 'Gemm':
            computed_inputs = [node.input[0]]
            W = inits[node.input[1]]
            b = (inits[node.input[2]]
                 if len(node.input) > 2 and node.input[2] in inits
                 else np.zeros(W.shape[0], dtype=np.float64))
            transB = attrs.get('transB', 0)
            if not transB:
                W = W.T
            params['W'] = W
            params['b'] = b

        elif op == 'MatMul':
            c1 = _const(node.input[1])
            c0 = _const(node.input[0])
            if c0 is not None and c1 is not None:
                constants[out_name] = c0 @ c1
                continue
            elif c1 is not None:
                computed_inputs = [node.input[0]]
                params['W'] = c1.T if c1.ndim == 2 else c1
                out_dim = c1.shape[1] if c1.ndim == 2 else c1.shape[0]
                params['b'] = np.zeros(out_dim, dtype=np.float64)
            elif c0 is not None:
                computed_inputs = [node.input[1]]
                params['W'] = c0
                params['b'] = np.zeros(c0.shape[0], dtype=np.float64)
            else:
                # Bilinear MatMul — use MatMulBilinearNode
                computed_inputs = [node.input[0], node.input[1]]

        elif op == 'Add':
            c0 = _const(node.input[0])
            c1 = _const(node.input[1])
            if c0 is not None and c1 is not None:
                constants[out_name] = c0 + c1
                continue
            elif c0 is not None:
                computed_inputs = [node.input[1]]
                params['bias'] = c0
            elif c1 is not None:
                computed_inputs = [node.input[0]]
                params['bias'] = c1
            else:
                computed_inputs = [node.input[0], node.input[1]]

        elif op == 'Sub':
            c0_s = _const(node.input[0])
            c1_s = _const(node.input[1])
            if c0_s is not None and c1_s is not None:
                constants[out_name] = c0_s - c1_s
                continue
            if c0_s is not None:
                computed_inputs = [node.input[1]]
                params['negate'] = True
                params['bias'] = c0_s
            elif c1_s is not None:
                computed_inputs = [node.input[0]]
                params['sub_val'] = c1_s
            else:
                computed_inputs = [node.input[0], node.input[1]]

        elif op == 'Mul':
            c0 = _const(node.input[0])
            c1 = _const(node.input[1])
            if c0 is not None and c1 is not None:
                constants[out_name] = c0 * c1
                continue
            elif c0 is not None:
                computed_inputs = [node.input[1]]
                params['scale'] = c0
            elif c1 is not None:
                computed_inputs = [node.input[0]]
                params['scale'] = c1
            else:
                computed_inputs = [node.input[0], node.input[1]]

        elif op == 'Div':
            c1 = _const(node.input[1])
            if c1 is not None:
                computed_inputs = [node.input[0]]
                params['scale'] = 1.0 / c1
            else:
                computed_inputs = [node.input[0], node.input[1]]

        elif op == 'Neg':
            computed_inputs = [node.input[0]]

        elif op in ('Relu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Sign', 'Softmax'):
            computed_inputs = [node.input[0]]
            if 'alpha' in attrs:
                params['alpha'] = attrs['alpha']
            if 'axis' in attrs:
                params['axis'] = attrs['axis']

        elif op == 'Clip':
            computed_inputs = [node.input[0]]
            if len(node.input) > 1:
                c_min = _const(node.input[1])
                if c_min is not None:
                    params['min'] = float(c_min)
            if len(node.input) > 2:
                c_max = _const(node.input[2])
                if c_max is not None:
                    params['max'] = float(c_max)

        elif op == 'BatchNormalization':
            computed_inputs = [node.input[0]]
            params['scale'] = inits[node.input[1]]
            params['bias'] = inits[node.input[2]]
            params['mean'] = inits[node.input[3]]
            params['var'] = inits[node.input[4]]
            params['epsilon'] = attrs.get('epsilon', 1e-5)

        elif op == 'MaxPool':
            computed_inputs = [node.input[0]]
            params['kernel_shape'] = tuple(attrs.get('kernel_shape', [2, 2]))
            params['stride'] = tuple(attrs.get('strides', [2, 2]))
            pads = attrs.get('pads', [0, 0, 0, 0])
            params['padding'] = (pads[0], pads[1])

        elif op == 'AveragePool':
            computed_inputs = [node.input[0]]
            params['kernel_shape'] = tuple(attrs.get('kernel_shape', [2, 2]))
            params['stride'] = tuple(attrs.get('strides', [2, 2]))
            pads = attrs.get('pads', [0, 0, 0, 0])
            params['padding'] = (pads[0], pads[1])

        elif op == 'Pad':
            computed_inputs = [node.input[0]]
            if len(node.input) > 1:
                c_pads = _const(node.input[1])
                if c_pads is not None:
                    params['pads'] = c_pads.astype(int).tolist()
            if len(node.input) > 2:
                c_val = _const(node.input[2])
                if c_val is not None:
                    params['constant_value'] = float(c_val)

        elif op == 'Concat':
            computed_inputs = [i for i in node.input
                               if _const(i) is None and i != '']
            const_inputs = [(idx, _const(i))
                            for idx, i in enumerate(node.input)
                            if _const(i) is not None]
            if const_inputs:
                params['const_inputs'] = const_inputs
            params['axis'] = attrs.get('axis', 0)

        elif op == 'Split':
            computed_inputs = [node.input[0]]
            params['axis'] = attrs.get('axis', 0)
            if len(node.input) > 1:
                c_split = _const(node.input[1])
                if c_split is not None:
                    params['split'] = c_split.astype(int).tolist()
            elif 'split' in attrs:
                params['split'] = attrs['split']
            # Register secondary outputs
            for i, out in enumerate(node.output):
                if i == 0:
                    continue
                graph.nodes[out] = SplitOutputNode(
                    name=out, op_type='SplitOutput',
                    inputs=[out_name], params={'index': i})

        elif op == 'Slice':
            computed_inputs = [node.input[0]]
            if len(node.input) > 1:
                starts = _const(node.input[1])
                if starts is not None:
                    params['starts'] = starts.astype(int).tolist()
            if len(node.input) > 2:
                ends = _const(node.input[2])
                if ends is not None:
                    params['ends'] = ends.astype(int).tolist()
            if len(node.input) > 3:
                axes = _const(node.input[3])
                if axes is not None:
                    params['axes'] = axes.astype(int).tolist()
            if len(node.input) > 4:
                steps = _const(node.input[4])
                if steps is not None:
                    params['steps'] = steps.astype(int).tolist()

        elif op == 'Gather':
            computed_inputs = [node.input[0]]
            c_indices = _const(node.input[1])
            if c_indices is not None:
                params['indices'] = c_indices
            else:
                computed_inputs.append(node.input[1])
            params['axis'] = attrs.get('axis', 0)

        elif op == 'ReduceSum':
            computed_inputs = [node.input[0]]
            if len(node.input) > 1:
                c_axes = _const(node.input[1])
                if c_axes is not None:
                    params['axes'] = c_axes.astype(int).tolist()
            if 'axes' in attrs:
                params['axes'] = attrs['axes']
            params['keepdims'] = attrs.get('keepdims', 1)

        elif op == 'ReduceMean':
            computed_inputs = [node.input[0]]
            if len(node.input) > 1:
                c_axes = _const(node.input[1])
                if c_axes is not None:
                    params['axes'] = c_axes.astype(int).tolist()
            if 'axes' in attrs:
                params['axes'] = attrs['axes']
            params['keepdims'] = attrs.get('keepdims', 1)

        elif op in ('Resize', 'Upsample'):
            computed_inputs = [node.input[0]]
            for idx, param_name in [(1, 'scales'), (2, 'scales'), (3, 'sizes')]:
                if len(node.input) > idx and node.input[idx] != '':
                    c = _const(node.input[idx])
                    if c is not None and c.size > 0:
                        params[param_name] = c

        elif op == 'Transpose':
            computed_inputs = [node.input[0]]
            if 'perm' in attrs:
                params['perm'] = attrs['perm']

        elif op in ('Flatten', 'Squeeze', 'Unsqueeze'):
            computed_inputs = [node.input[0]]
            if 'axis' in attrs:
                params['axis'] = attrs['axis']
            if op == 'Unsqueeze' and len(node.input) > 1:
                c_axes = _const(node.input[1])
                if c_axes is not None:
                    params['axes'] = c_axes.astype(int).tolist()

        elif op == 'Reshape':
            computed_inputs = [node.input[0]]
            if len(node.input) > 1:
                c_shape = _const(node.input[1])
                if c_shape is not None:
                    params['shape'] = tuple(int(x) for x in c_shape)

        elif op == 'Dropout':
            computed_inputs = [node.input[0]]

        elif op in ('Sin', 'Cos', 'Pow', 'Floor'):
            computed_inputs = [node.input[0]]
            if op == 'Pow' and len(node.input) > 1:
                c_exp = _const(node.input[1])
                if c_exp is not None:
                    params['exponent'] = float(c_exp)

        elif op in ('ConstantOfShape',):
            computed_inputs = [node.input[0]]
            params['value'] = 0.0

        elif op in ('Expand', 'Where', 'Equal', 'ScatterND', 'ArgMax',
                     'Min', 'Max'):
            computed_inputs = [i for i in node.input
                               if _const(i) is None and i != '']
            for idx, i in enumerate(node.input):
                c = _const(i)
                if c is not None:
                    params[f'const_{idx}'] = c

        elif op == 'Shape':
            computed_inputs = [node.input[0]]

        elif op == 'Identity':
            computed_inputs = [node.input[0]]

        else:
            computed_inputs = [i for i in node.input
                               if _const(i) is None and i != ''
                               and i not in init_names]

        # Constant folding
        if computed_inputs and all(_const(i) is not None
                                   for i in computed_inputs):
            folded = _try_fold_constant(op, computed_inputs, params, _const)
            if folded is not None:
                constants[out_name] = folded
                continue

        # Choose the right node class
        if op == 'MatMul' and 'W' not in params:
            cls = MatMulBilinearNode
        else:
            cls = OP_REGISTRY.get(op, GraphNode)

        graph.nodes[out_name] = cls(
            name=out_name, op_type=op,
            inputs=computed_inputs, params=params)

    # Resolve input shape — keep batch dim (always 1)
    # dims from ONNX: first is batch (0 or dynamic), rest are data dims
    # e.g. [0, 1, 20, 20] -> (1, 1, 20, 20)
    # e.g. [0, 0, 0, 5]   -> (1, 5)  (only last dim is real data)
    # e.g. [12, 8]        -> (12, 8)  (no batch dim — both dims are data)
    dims = graph._raw_input_dims
    is_concrete = [isinstance(d, int) and d > 0 for d in dims]
    if all(is_concrete):
        # Every dim is concrete — no batch dim to strip. Use as-is.
        # nn4sys pensieve_*_parallel uses fixed input shape [12, 8]
        # (no batch dim); pre-fix the loader incorrectly stripped dim 0
        # and produced (1, 8), breaking downstream propagate_fc.
        graph.input_shape = tuple(dims)
    else:
        # Standard case: dim 0 is dynamic batch (or 0). Replace with 1
        # and find first concrete data dim.
        first_data = 1
        while first_data < len(dims) and not is_concrete[first_data]:
            first_data += 1
        if first_data >= len(dims):
            # All dims dynamic except maybe last — treat as flat
            last_val = dims[-1] if isinstance(dims[-1], int) and dims[-1] > 0 else 1
            graph.input_shape = (1, last_val)
        else:
            resolved = [1] + [d if isinstance(d, int) and d > 0 else 1
                              for d in dims[first_data:]]
            graph.input_shape = tuple(resolved)

    graph.topological_sort()
    _infer_shapes(graph)
    _fold_batchnorm(graph)
    _cast_params(graph)
    _precache_conv_tensors(graph)
    # Stash the original ONNX path so verify_graph can re-run it via
    # ONNXRuntime to validate any SAT witness (defense-in-depth: catches
    # spec-encoding or graph-builder bugs that would otherwise silently
    # produce spurious counterexamples).
    graph.onnx_path = str(onnx_path)
    return graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_attrs(node):
    """Extract ONNX node attributes into a dict."""
    attrs = {}
    for attr in node.attribute:
        if attr.ints:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == 2:  # INT
            attrs[attr.name] = attr.i
        elif attr.type == 1:  # FLOAT
            attrs[attr.name] = attr.f
        elif attr.type == 3:  # STRING
            attrs[attr.name] = (attr.s.decode()
                                if isinstance(attr.s, bytes) else attr.s)
    return attrs


def _try_fold_constant(op, computed_inputs, params, const_fn):
    """Try to evaluate an op on constant inputs. Returns result or None."""
    vals = [const_fn(i) for i in computed_inputs]
    if op == 'Relu':
        return np.maximum(vals[0], 0)
    elif op == 'LeakyRelu':
        alpha = params.get('alpha', 0.01)
        return np.where(vals[0] >= 0, vals[0], alpha * vals[0])
    elif op == 'Neg':
        return -vals[0]
    elif op == 'Sigmoid':
        return 1.0 / (1.0 + np.exp(-vals[0]))
    elif op in ('Flatten', 'Squeeze', 'Unsqueeze', 'Reshape',
                 'Dropout', 'Identity', 'Cast'):
        return vals[0]
    elif op == 'Concat':
        return np.concatenate([v.flatten() for v in vals])
    elif op == 'Slice':
        starts = params.get('starts', [0])
        ends = params.get('ends', [len(vals[0])])
        return vals[0].flatten()[starts[0]:ends[0]]
    elif op == 'Gather':
        indices = params.get('indices')
        if indices is not None:
            return vals[0].flatten()[indices.flatten().astype(int)]
    elif op == 'Transpose':
        return vals[0]
    elif op in ('Gemm', 'MatMul') and 'W' in params:
        return params['W'] @ vals[0].flatten() + params['b']
    elif op == 'Div' and 'scale' in params:
        return vals[0].flatten() * params['scale']
    elif op == 'Sign':
        return np.sign(vals[0])
    elif op == 'ReduceSum':
        return np.array([vals[0].sum()])
    elif op == 'ReduceMean':
        return np.array([vals[0].mean()])
    return None


def _infer_shapes(graph):
    """Propagate shapes through the graph via polymorphic dispatch."""
    shapes = {graph.input_name: graph.input_shape}
    for name in graph.topo_order:
        node = graph.nodes[name]
        node.infer_shape(shapes)
        # For SplitOutput, compute shape from parent Split
        if node.op_type == 'SplitOutput':
            parent = graph.nodes.get(node.inputs[0])
            if parent and parent.op_type == 'Split':
                parent_inp = shapes.get(parent.inputs[0])
                split_sizes = parent.params.get('split')
                axis = parent.params.get('axis', 0)
                idx = node.params.get('index', 0)
                if parent_inp and split_sizes and idx < len(split_sizes):
                    out = list(parent_inp)
                    out[axis] = split_sizes[idx]
                    node.output_shape = tuple(out)
        shapes[name] = node.output_shape


def _fold_batchnorm(graph):
    """Fold BatchNormalization into preceding Conv, ConvTranspose or Gemm."""
    to_remove = []
    for name in graph.topo_order:
        node = graph.nodes[name]
        if node.op_type != 'BatchNormalization':
            continue

        pred_name = node.inputs[0]
        if pred_name not in graph.nodes:
            continue
        pred = graph.nodes[pred_name]
        if pred.op_type not in ('Conv', 'ConvTranspose', 'Gemm'):
            continue

        scale = node.params['scale']
        bn_bias = node.params['bias']
        mean = node.params['mean']
        var = node.params['var']
        eps = node.params['epsilon']
        factor = scale / np.sqrt(var + eps)

        if pred.op_type == 'Conv':
            pred.params['kernel'] = (
                pred.params['kernel'] * factor[:, None, None, None])
            pred.params['bias'] = (
                factor * (pred.params['bias'] - mean) + bn_bias)
        elif pred.op_type == 'ConvTranspose':
            # Kernel layout (C_in, C_out, kH, kW) — broadcast on C_out (axis 1).
            pred.params['kernel'] = (
                pred.params['kernel'] * factor[None, :, None, None])
            pred.params['bias'] = (
                factor * (pred.params['bias'] - mean) + bn_bias)
        else:
            pred.params['W'] = pred.params['W'] * factor[:, None]
            pred.params['b'] = factor * (pred.params['b'] - mean) + bn_bias

        for other in graph.nodes.values():
            other.inputs = [pred_name if inp == name else inp
                            for inp in other.inputs]
        if graph.output_name == name:
            graph.output_name = pred_name

        to_remove.append(name)

    for name in to_remove:
        del graph.nodes[name]

    if to_remove:
        graph.topological_sort()


def _cast_params(graph):
    """Cast all float64 numpy params to graph.dtype."""
    dt = graph.dtype
    for node in graph.nodes.values():
        for key, val in node.params.items():
            if isinstance(val, np.ndarray) and val.dtype == np.float64:
                node.params[key] = val.astype(dt)


def _precache_conv_tensors(graph):
    """Pre-build cached torch tensors for Conv nodes.

    Called after shape inference and batchnorm folding so that
    zonotope_propagate pays no tensor-creation overhead at runtime.
    """
    from .network import ConvNode
    for name in graph.topo_order:
        node = graph.nodes[name]
        if isinstance(node, ConvNode):
            node.precache_conv_layer(graph)
