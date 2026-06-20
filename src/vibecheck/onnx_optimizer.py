"""Semantics-preserving graph optimizations applied after ONNX loading.

Passes:

- `fold_conv(graph)` folds the expanded
  `Conv(C→2C, k×k) → ReLU → Conv(2C→C, 1×1) → ReLU` pattern (used by some
  exporters to make every neuron pairwise-symmetric) back into a single
  `Conv(C→C, k×k) → ReLU`. Exact because ReLU(z) − ReLU(−z) = z. Mirrors
  auto_LiRPA / α,β-CROWN's `optimize_relu_relation`.

- `fold_gemm(graph)` is the fully-connected analogue: it folds
  `Gemm(C→C+S) → ReLU → Gemm(C+S→C) → ReLU` (S split neurons added as
  ±-paired rows, plus passthrough rows) back into a single
  `Gemm(C→C) → ReLU`, exact by the same ReLU(z) − ReLU(−z) = z identity.

- `fuse_gemm_reshape_conv(graph)` fuses `Gemm → Reshape → Conv` into a
  single equivalent FC layer, so the backward pass operates on fewer,
  larger layers (matches what CROWN-style tightening expects).

All are called from `ComputeGraph.optimize(settings)`.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .network import _prod


def fuse_gemm_reshape_conv(graph):
    """Fuse Gemm → Reshape → Conv into a single FC layer.

    Pattern: Gemm(n→m) → Reshape(m→C,H,W) → Conv(C→C',kH,kW)
    Result:  Gemm(n→C'*H'*W') with composed weights.

    The fused weight is computed by pushing each of the n input basis
    vectors through the original Gemm+Reshape+Conv to get the n columns
    of the fused (C'*H'*W', n) weight matrix.
    """
    fused_any = False
    topo = list(graph.topo_order)

    for i, name in enumerate(topo):
        node = graph.nodes.get(name)
        if node is None or node.op_type not in ('Gemm', 'MatMul'):
            continue

        # Find successor chain: Gemm → Reshape → Conv
        succs_reshape = [n for n in graph.nodes.values()
                         if name in n.inputs
                         and n.op_type == 'Reshape']
        if len(succs_reshape) != 1:
            continue
        reshape_node = succs_reshape[0]

        succs_conv = [n for n in graph.nodes.values()
                      if reshape_node.name in n.inputs
                      and n.op_type == 'Conv']
        if len(succs_conv) != 1:
            continue
        conv_node = succs_conv[0]

        # Gemm must feed only the reshape (not used elsewhere)
        gemm_consumers = [n for n in graph.nodes.values()
                          if name in n.inputs]
        if len(gemm_consumers) != 1:
            continue
        reshape_consumers = [n for n in graph.nodes.values()
                             if reshape_node.name in n.inputs]
        if len(reshape_consumers) != 1:
            continue

        # Skip if Gemm doesn't have a constant W/b (e.g., after onnxsim
        # produces a bilinear Gemm whose W comes from another node).
        if 'W' not in node.params or 'b' not in node.params:
            continue
        W_gemm = node.params['W']       # (m, n)
        b_gemm = node.params['b']       # (m,)
        kernel = conv_node.params['kernel']   # (C_out, C_in, kH, kW)
        b_conv = conv_node.params['bias']     # (C_out,)
        stride = conv_node.params['stride']
        padding = conv_node.params['padding']

        # Determine the reshape target shape (C, H, W)
        reshape_shape = reshape_node.output_shape
        if reshape_shape is None:
            continue
        if len(reshape_shape) == 4 and reshape_shape[0] == 1:
            chw = reshape_shape[1:]
        elif len(reshape_shape) == 3:
            chw = reshape_shape
        else:
            continue

        if _prod(chw) != W_gemm.shape[0]:
            continue

        # output = Conv(reshape(W_gemm @ x + b_gemm)) + b_conv
        #        = Conv_linear(W_gemm) @ x + Conv_linear(b_gemm) + b_conv
        n_in = W_gemm.shape[1]
        dt = torch.float64

        k_t = torch.tensor(kernel, dtype=dt)
        W_t = torch.tensor(W_gemm, dtype=dt)  # (m, n)
        b_gemm_t = torch.tensor(b_gemm, dtype=dt)

        # Push b_gemm through conv to get fused bias
        b_4d = b_gemm_t.reshape(1, *chw)
        b_fused_t = F.conv2d(b_4d, k_t, bias=torch.tensor(b_conv, dtype=dt),
                              stride=stride, padding=padding).flatten()

        # Push each column of W_gemm through conv (batched over n cols)
        W_4d = W_t.T.reshape(n_in, *chw)  # (n, C, H, W)
        W_fused_t = F.conv2d(W_4d, k_t, stride=stride,
                              padding=padding)  # (n, C', H', W')
        out_flat = _prod(W_fused_t.shape[1:])
        W_fused = W_fused_t.reshape(n_in, out_flat).T  # (out_flat, n)

        W_fused_np = W_fused.numpy().astype(graph.dtype)
        b_fused_np = b_fused_t.numpy().astype(graph.dtype)

        node.params['W'] = W_fused_np
        node.params['b'] = b_fused_np

        # Rewire: conv's consumers now point to gemm
        conv_name = conv_node.name
        for other in graph.nodes.values():
            other.inputs = [name if inp == conv_name else inp
                            for inp in other.inputs]
        if graph.output_name == conv_name:
            graph.output_name = name

        del graph.nodes[reshape_node.name]
        del graph.nodes[conv_name]
        fused_any = True

    if fused_any:
        graph.topological_sort()
        from .onnx_loader import _infer_shapes, _precache_conv_tensors
        _infer_shapes(graph)
        _precache_conv_tensors(graph)

    return fused_any


def fold_conv(graph):
    """Fold ReLU-split pattern back into a single Conv → ReLU.

    Detects: Conv(C_in → 2C, k×k) → ReLU → Conv(2C → C, 1×1) → ReLU
    where the expanded conv has adjacent paired filters (w, -w) with
    biases (b, -b), and the 1×1 conv has [+1, -1] entries recombining
    each pair into a single output channel.

    Equivalent to the original Conv(C_in → C, k×k) → ReLU because
    ReLU(z) - ReLU(-z) = z, so the 1×1 recombination recovers the
    pre-ReLU value and the second ReLU gives the same result as a
    single ReLU on the original.

    The 1×1 bias is absorbed:  b_fused = b_orig + b_1x1
    """
    folded_any = False
    topo = list(graph.topo_order)

    for i, name in enumerate(topo):
        node = graph.nodes.get(name)
        if node is None or node.op_type != 'Conv':
            continue

        relu1_list = [n for n in graph.nodes.values()
                      if name in n.inputs and n.op_type == 'Relu']
        if len(relu1_list) != 1:
            continue
        relu1 = relu1_list[0]

        conv1x1_list = [n for n in graph.nodes.values()
                        if relu1.name in n.inputs and n.op_type == 'Conv']
        if len(conv1x1_list) != 1:
            continue
        conv1x1 = conv1x1_list[0]

        relu2_list = [n for n in graph.nodes.values()
                      if conv1x1.name in n.inputs and n.op_type == 'Relu']
        if len(relu2_list) != 1:
            continue

        # Each intermediate node must have exactly one consumer.
        if len([n for n in graph.nodes.values()
                if name in n.inputs]) != 1:
            continue
        if len([n for n in graph.nodes.values()
                if relu1.name in n.inputs]) != 1:
            continue
        if len([n for n in graph.nodes.values()
                if conv1x1.name in n.inputs]) != 1:
            continue

        K_exp = node.params['kernel']       # (2C, C_in, kH, kW)
        b_exp = node.params['bias']         # (2C,)
        A = conv1x1.params['kernel']        # (C, 2C, 1, 1)
        b_1x1 = conv1x1.params['bias']     # (C,)

        if A.shape[2] != 1 or A.shape[3] != 1:
            continue
        C_out = A.shape[0]
        C_exp = A.shape[1]
        if C_exp != 2 * C_out:
            continue

        A_mat = A[:, :, 0, 0]  # (C, 2C)
        tol = 1e-5

        K_orig = np.zeros((C_out,) + K_exp.shape[1:], dtype=K_exp.dtype)
        b_orig = np.zeros(C_out, dtype=b_exp.dtype)
        valid = True

        for j in range(C_out):
            nonzero = np.where(np.abs(A_mat[j]) > tol)[0]
            if len(nonzero) != 2:
                valid = False
                break
            i_pos, i_neg = nonzero
            v_pos, v_neg = A_mat[j, i_pos], A_mat[j, i_neg]

            if not (abs(abs(v_pos) - 1.0) < tol and abs(abs(v_neg) - 1.0) < tol
                    and v_pos * v_neg < 0):
                valid = False
                break

            if not np.allclose(K_exp[i_pos], -K_exp[i_neg], atol=tol):
                valid = False
                break

            if not abs(b_exp[i_pos] + b_exp[i_neg]) < tol:
                valid = False
                break

            # The +1 channel gives the original filter
            if v_pos > 0:
                K_orig[j] = K_exp[i_pos]
                b_orig[j] = b_exp[i_pos]
            else:
                K_orig[j] = K_exp[i_neg]
                b_orig[j] = b_exp[i_neg]

        if not valid:
            continue

        node.params['kernel'] = K_orig
        node.params['bias'] = b_orig + b_1x1

        # Rewire: relu2's consumers now point to relu1.
        relu2_name = relu2_list[0].name
        for other in graph.nodes.values():
            other.inputs = [relu1.name if inp == relu2_name else inp
                            for inp in other.inputs]
        if graph.output_name == relu2_name:
            graph.output_name = relu1.name

        del graph.nodes[conv1x1.name]
        del graph.nodes[relu2_name]
        folded_any = True

    if folded_any:
        graph.topological_sort()
        from .onnx_loader import _infer_shapes, _precache_conv_tensors
        _infer_shapes(graph)
        _precache_conv_tensors(graph)

    return folded_any


def fold_gemm(graph):
    """Fold ReLU-split pattern back into a single Gemm → ReLU.

    Detects: Gemm(C_in → C+S) → ReLU → Gemm(C+S → C) → ReLU
    where the expanded Gemm has S split neurons with paired rows
    (w, -w) and biases (b, -b), and the merge Gemm has a (C, C+S)
    selector matrix with two kinds of rows:

      - **passthrough** (one +1 nonzero): selects an unsplit neuron's
        ReLU output; the merge bias entry must be 0 — otherwise
        ReLU(ReLU(z)+b) ≠ ReLU(z+b).
      - **split pair** (one +1 and one −1 nonzero on a w/−w pair):
        recombines via ReLU(z) − ReLU(−z) = z; the merge bias entry
        absorbs cleanly into the original bias.
    """
    folded_any = False
    topo = list(graph.topo_order)

    for name in topo:
        node = graph.nodes.get(name)
        if node is None or node.op_type != 'Gemm':
            continue

        relu1_list = [n for n in graph.nodes.values()
                      if name in n.inputs and n.op_type == 'Relu']
        if len(relu1_list) != 1:
            continue
        relu1 = relu1_list[0]

        gemm2_list = [n for n in graph.nodes.values()
                      if relu1.name in n.inputs and n.op_type == 'Gemm']
        if len(gemm2_list) != 1:
            continue
        gemm2 = gemm2_list[0]

        relu2_list = [n for n in graph.nodes.values()
                      if gemm2.name in n.inputs and n.op_type == 'Relu']
        if len(relu2_list) != 1:
            continue

        # Each intermediate node must have exactly one consumer.
        if len([n for n in graph.nodes.values()
                if name in n.inputs]) != 1:
            continue
        if len([n for n in graph.nodes.values()
                if relu1.name in n.inputs]) != 1:
            continue
        if len([n for n in graph.nodes.values()
                if gemm2.name in n.inputs]) != 1:
            continue

        W1 = node.params['W']        # (C+S, C_in)
        b1 = node.params['b']        # (C+S,)
        W2 = gemm2.params['W']       # (C, C+S)
        b2 = gemm2.params['b']       # (C,)

        C_out = W2.shape[0]
        C_exp = W2.shape[1]
        if W1.shape[0] != C_exp or C_exp < C_out:
            continue

        tol = 1e-5
        W_orig = np.zeros((C_out, W1.shape[1]), dtype=W1.dtype)
        b_orig = np.zeros(C_out, dtype=b1.dtype)
        valid = True

        for j in range(C_out):
            nonzero = np.where(np.abs(W2[j]) > tol)[0]
            if len(nonzero) == 1:
                # Passthrough: must be +1 with zero merge bias.
                i = nonzero[0]
                if not (abs(W2[j, i] - 1.0) < tol and abs(b2[j]) < tol):
                    valid = False
                    break
                W_orig[j] = W1[i]
                b_orig[j] = b1[i]
                continue
            if len(nonzero) != 2:
                valid = False
                break
            i_pos, i_neg = nonzero
            v_pos, v_neg = W2[j, i_pos], W2[j, i_neg]
            if not (abs(abs(v_pos) - 1.0) < tol
                    and abs(abs(v_neg) - 1.0) < tol
                    and v_pos * v_neg < 0):
                valid = False
                break
            if not np.allclose(W1[i_pos], -W1[i_neg], atol=tol):
                valid = False
                break
            if not abs(b1[i_pos] + b1[i_neg]) < tol:
                valid = False
                break
            if v_pos > 0:
                W_orig[j] = W1[i_pos]
                b_orig[j] = b1[i_pos]
                b_orig[j] += b2[j]
            else:
                W_orig[j] = W1[i_neg]
                b_orig[j] = b1[i_neg]
                b_orig[j] += b2[j]

        if not valid:
            continue

        node.params['W'] = W_orig
        node.params['b'] = b_orig

        relu2_name = relu2_list[0].name
        for other in graph.nodes.values():
            other.inputs = [relu1.name if inp == relu2_name else inp
                            for inp in other.inputs]
        if graph.output_name == relu2_name:
            graph.output_name = relu1.name

        del graph.nodes[gemm2.name]
        del graph.nodes[relu2_name]
        folded_any = True

    if folded_any:
        graph.topological_sort()
        from .onnx_loader import _infer_shapes, _precache_conv_tensors
        _infer_shapes(graph)
        _precache_conv_tensors(graph)

    return folded_any


def maxpool_to_relu(graph):
    """Replace each MaxPool with an EXACT ReLU decomposition.

    `max(a, b) = a + ReLU(b - a)` is exact. For a kH×kW window we:
      1. extract the P = kH*kW window positions with a single 1-hot
         phase-extraction Conv (C → P*C, stride = pool stride, PHASE-MAJOR
         channel layout out_ch = p*C + c, so phase p lands in a contiguous
         channel block);
      2. contiguous-channel-Slice that into P phase tensors of the pooled
         shape;
      3. reduce them with a binary-max TREE of `Sub`/`Relu`/`Add` nodes.

    The result is EXACT (not a relaxation) and uses only Conv + contiguous
    Slice + Sub + Relu + Add — all of which the gpu_graph / CROWN-backward /
    input-split backends support. MaxPool itself has NO handler in those
    backends (only a point-only dense path and a loose box-approx forward),
    so this rewrite is what lets conv nets with pooling (vggnet16) verify.
    The forks it introduces also stop `verify_graph` from auto-routing the
    (now non-sequential) graph to the no-MaxPool milp/gpu_layers path.

    Padding is NOT supported: MaxPool pads with -inf, which a Conv/ReLU
    decomposition cannot represent — a padded MaxPool raises (loud, never a
    silent wrong bound). vggnet16 uses pad=0.
    """
    from .network import OP_REGISTRY

    maxpools = [n for n in list(graph.topo_order)
                if graph.nodes.get(n) is not None
                and graph.nodes[n].op_type == 'MaxPool']
    if not maxpools:
        return False

    for mp_name in maxpools:
        mp = graph.nodes[mp_name]
        inp = mp.inputs[0]
        kH, kW = mp.params['kernel_shape']
        sH, sW = mp.params['stride']
        pH, pW = mp.params['padding']
        if pH != 0 or pW != 0:
            raise NotImplementedError(
                f'maxpool_to_relu: padded MaxPool at {mp_name!r} (pad='
                f'{(pH, pW)}) unsupported — maxpool pads with -inf, which the '
                'ReLU decomposition cannot represent.')
        ish = (graph.nodes[inp].output_shape if inp in graph.nodes
               else graph.input_shape)
        if ish is None or len(ish) not in (3, 4):
            raise NotImplementedError(
                f'maxpool_to_relu: need a 3/4-D input shape at {mp_name!r}, '
                f'got {ish!r}')
        cax = 1 if len(ish) == 4 else 0          # channel axis (N,C,H,W or C,H,W)
        C = int(ish[cax])
        P = kH * kW

        # (P*C, C, kH, kW) 1-hot phase-extraction kernel, phase-major:
        # out_ch p*C+c reads input channel c at window position (p//kW, p%kW).
        ker = np.zeros((P * C, C, kH, kW), dtype=np.float32)
        for p in range(P):
            ph, pw = p // kW, p % kW
            for c in range(C):
                ker[p * C + c, c, ph, pw] = 1.0
        conv_name = mp_name + '__mp2relu_conv'
        graph.nodes[conv_name] = OP_REGISTRY['Conv'](
            name=conv_name, op_type='Conv', inputs=[inp],
            params={'kernel': ker, 'bias': np.zeros(P * C, np.float32),
                    'stride': (sH, sW), 'padding': (0, 0), 'group': 1})

        # contiguous channel slices -> P phase tensors of the pooled shape
        phases = []
        for p in range(P):
            pn = f'{mp_name}__mp2relu_phase{p}'
            graph.nodes[pn] = OP_REGISTRY['Slice'](
                name=pn, op_type='Slice', inputs=[conv_name],
                params={'axes': [cax], 'starts': [p * C],
                        'ends': [(p + 1) * C]})
            phases.append(pn)

        # binary-max tree: max(a, b) = a + ReLU(b - a)
        _ctr = [0]

        def _binmax(a, b):
            i = _ctr[0]; _ctr[0] += 1
            sub = f'{mp_name}__mp2relu_sub{i}'
            rel = f'{mp_name}__mp2relu_relu{i}'
            mx = f'{mp_name}__mp2relu_max{i}'
            graph.nodes[sub] = OP_REGISTRY['Sub'](
                name=sub, op_type='Sub', inputs=[b, a], params={})
            graph.nodes[rel] = OP_REGISTRY['Relu'](
                name=rel, op_type='Relu', inputs=[sub], params={})
            graph.nodes[mx] = OP_REGISTRY['Add'](
                name=mx, op_type='Add', inputs=[a, rel], params={})
            return mx

        level = phases
        while len(level) > 1:
            nxt = [_binmax(level[k], level[k + 1])
                   for k in range(0, len(level) - 1, 2)]
            if len(level) % 2 == 1:
                nxt.append(level[-1])      # carry the odd one to the next level
            level = nxt
        out_name = level[0]

        # rewire consumers of the MaxPool to the decomposition output; drop it
        for other in graph.nodes.values():
            other.inputs = [out_name if x == mp_name else x
                            for x in other.inputs]
        if getattr(graph, 'output_name', None) == mp_name:
            graph.output_name = out_name
        if getattr(graph, 'output_names', None):
            graph.output_names = [out_name if x == mp_name else x
                                  for x in graph.output_names]
        del graph.nodes[mp_name]

    graph.topological_sort()
    from .onnx_loader import _infer_shapes, _precache_conv_tensors
    _infer_shapes(graph)
    _precache_conv_tensors(graph)
    return True


def min_max_to_relu(graph):
    """Replace each Min/Max with an EXACT ReLU+affine decomposition.

    Min/Max have no sound general handler (the `MiscNode` fallback only passes a
    POINT through input 0), so any net that actually bounds a Min/Max — e.g. the
    clamp `clamp(v,LO,HI)=Min(Max(v,LO),HI)` the network-pair (monotonic_acasxu)
    converter emits — needs this. Like `maxpool_to_relu`, the result is exact (not
    a relaxation) and uses only Sub/Relu/Add, which every backend supports.

    Constant operand (the common case — `Min(x, c)` / `Max(x, c)`, the const folded
    into `params['const_*']`), with single-input affine ops (work with noise):
        max(x,c) = c + ReLU(x - c)     Sub(sub_val=c) -> Relu -> Add(bias=c)
        min(x,c) = c - ReLU(c - x)     Sub(negate,bias=c) -> Relu -> Sub(negate,bias=c)
    Two variable inputs `op(a,b)` (mirrors maxpool's binary-max):
        max(a,b) = a + ReLU(b - a)     Sub([b,a]) -> Relu -> Add([a,relu])
        min(a,b) = a - ReLU(a - b)     Sub([a,b]) -> Relu -> Sub([a,relu])
    """
    from .network import OP_REGISTRY

    mm = [n for n in list(graph.topo_order)
          if graph.nodes.get(n) is not None
          and graph.nodes[n].op_type in ('Min', 'Max')]
    if not mm:
        return False

    for nm in mm:
        node = graph.nodes[nm]
        is_max = node.op_type == 'Max'
        ckey = next((k for k in node.params if k.startswith('const_')), None)
        sub = f'{nm}__mm2relu_sub'
        rel = f'{nm}__mm2relu_relu'
        out = f'{nm}__mm2relu_out'

        if ckey is not None:
            # constant operand: variable is inputs[0], constant is params[ckey]
            c = np.asarray(node.params[ckey])
            x = node.inputs[0]
            if is_max:                                   # c + ReLU(x - c)
                graph.nodes[sub] = OP_REGISTRY['Sub'](
                    name=sub, op_type='Sub', inputs=[x], params={'sub_val': c})
                graph.nodes[rel] = OP_REGISTRY['Relu'](
                    name=rel, op_type='Relu', inputs=[sub], params={})
                graph.nodes[out] = OP_REGISTRY['Add'](
                    name=out, op_type='Add', inputs=[rel], params={'bias': c})
            else:                                        # c - ReLU(c - x)
                graph.nodes[sub] = OP_REGISTRY['Sub'](
                    name=sub, op_type='Sub', inputs=[x],
                    params={'negate': True, 'bias': c})
                graph.nodes[rel] = OP_REGISTRY['Relu'](
                    name=rel, op_type='Relu', inputs=[sub], params={})
                graph.nodes[out] = OP_REGISTRY['Sub'](
                    name=out, op_type='Sub', inputs=[rel],
                    params={'negate': True, 'bias': c})
        else:
            # two variable inputs a,b
            assert len(node.inputs) == 2, \
                f'min_max_to_relu: {node.op_type} {nm!r} has no const and {len(node.inputs)} inputs'
            a, b = node.inputs
            if is_max:                                   # a + ReLU(b - a)
                graph.nodes[sub] = OP_REGISTRY['Sub'](
                    name=sub, op_type='Sub', inputs=[b, a], params={})
                graph.nodes[rel] = OP_REGISTRY['Relu'](
                    name=rel, op_type='Relu', inputs=[sub], params={})
                graph.nodes[out] = OP_REGISTRY['Add'](
                    name=out, op_type='Add', inputs=[a, rel], params={})
            else:                                        # a - ReLU(a - b)
                graph.nodes[sub] = OP_REGISTRY['Sub'](
                    name=sub, op_type='Sub', inputs=[a, b], params={})
                graph.nodes[rel] = OP_REGISTRY['Relu'](
                    name=rel, op_type='Relu', inputs=[sub], params={})
                graph.nodes[out] = OP_REGISTRY['Sub'](
                    name=out, op_type='Sub', inputs=[a, rel], params={})

        # rewire consumers of the Min/Max to the decomposition output; drop it
        for other in graph.nodes.values():
            other.inputs = [out if x == nm else x for x in other.inputs]
        if getattr(graph, 'output_name', None) == nm:
            graph.output_name = out
        if getattr(graph, 'output_names', None):
            graph.output_names = [out if x == nm else x
                                  for x in graph.output_names]
        del graph.nodes[nm]

    graph.topological_sort()
    from .onnx_loader import _infer_shapes, _precache_conv_tensors
    _infer_shapes(graph)
    _precache_conv_tensors(graph)
    return True


def drop_identity_pads(graph):
    """Remove Pad nodes whose pads are all zero (exact identity).

    TinyYOLO (yolo_2023) carries `Pad(mode=constant, pads=[0]*8, value=0)`
    nodes in front of its AveragePools — pure no-ops left by the exporter.
    The gpu_graph serializer (correctly) raises NotImplementedError on Pad
    rather than silently aliasing, so these identity pads would block the
    whole pipeline. Splicing them out is semantics-preserving by definition:
    zero padding on every edge changes nothing. Non-zero pads are left
    untouched (still raise loudly downstream until a real handler exists).
    """
    dropped_any = False
    for name in list(graph.nodes):
        node = graph.nodes.get(name)
        if node is None or node.op_type != 'Pad':
            continue
        pads = node.params.get('pads')
        if pads is None or any(int(p) != 0 for p in pads):
            # No pads info (e.g. dynamic pads input) → must NOT assume
            # identity; non-zero pads → real padding. Keep the node either
            # way so gpu_graph's NotImplementedError stays loud.
            continue
        assert len(node.inputs) >= 1, f'Pad {name!r} has no input'
        src = node.inputs[0]
        for other in graph.nodes.values():
            other.inputs = [src if inp == name else inp for inp in other.inputs]
        if graph.output_name == name:
            graph.output_name = src
        del graph.nodes[name]
        dropped_any = True
    if dropped_any:
        graph.topological_sort()
