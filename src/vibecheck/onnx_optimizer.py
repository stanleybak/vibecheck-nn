"""Graph optimizations for BnB verification.

Fuses Gemm → Reshape → Conv into a single equivalent FC layer so that
the backward pass operates on fewer, larger layers (matching the
structure expected by CROWN-style tightening).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .network import ComputeGraph, GemmNode, _prod


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

        # Reshape must have exactly one consumer which is Conv
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
        # Strip batch dim
        if len(reshape_shape) == 4 and reshape_shape[0] == 1:
            chw = reshape_shape[1:]
        elif len(reshape_shape) == 3:
            chw = reshape_shape
        else:
            continue

        if _prod(chw) != W_gemm.shape[0]:
            continue

        # Compute fused weight by matrix composition:
        # output = Conv(reshape(W_gemm @ x + b_gemm)) + b_conv
        # = Conv_linear(W_gemm @ x + b_gemm) + b_conv
        # = Conv_linear(W_gemm) @ x + Conv_linear(b_gemm) + b_conv
        #
        # Conv_linear is the conv applied as a linear operator (no bias).
        n_in = W_gemm.shape[1]
        dt = torch.float64

        k_t = torch.tensor(kernel, dtype=dt)
        W_t = torch.tensor(W_gemm, dtype=dt)  # (m, n)
        b_gemm_t = torch.tensor(b_gemm, dtype=dt)

        # Push b_gemm through conv to get fused bias
        b_4d = b_gemm_t.reshape(1, *chw)
        b_fused_t = F.conv2d(b_4d, k_t, bias=torch.tensor(b_conv, dtype=dt),
                              stride=stride, padding=padding).flatten()

        # Push each column of W_gemm through conv to get fused W
        # W_gemm is (m, n), columns are (m,) vectors to reshape to (C,H,W)
        # Batch all n columns at once: (n, C, H, W)
        W_4d = W_t.T.reshape(n_in, *chw)  # (n, C, H, W)
        W_fused_t = F.conv2d(W_4d, k_t, stride=stride,
                              padding=padding)  # (n, C', H', W')
        out_flat = _prod(W_fused_t.shape[1:])
        W_fused = W_fused_t.reshape(n_in, out_flat).T  # (out_flat, n)

        W_fused_np = W_fused.numpy().astype(graph.dtype)
        b_fused_np = b_fused_t.numpy().astype(graph.dtype)

        # Replace: update the Gemm node with fused params
        node.params['W'] = W_fused_np
        node.params['b'] = b_fused_np

        # Rewire: conv's consumers now point to gemm
        conv_name = conv_node.name
        for other in graph.nodes.values():
            other.inputs = [name if inp == conv_name else inp
                            for inp in other.inputs]
        if graph.output_name == conv_name:
            graph.output_name = name

        # Remove reshape and conv nodes
        del graph.nodes[reshape_node.name]
        del graph.nodes[conv_name]
        fused_any = True

    if fused_any:
        graph.topological_sort()
        # Re-infer shapes
        from .onnx_loader import _infer_shapes
        _infer_shapes(graph)
        # Re-precache conv tensors
        from .onnx_loader import _precache_conv_tensors
        _precache_conv_tensors(graph)

    return fused_any


def fold_relusplitter(graph):
    """Fold ReLU-split pattern back into a single Conv → ReLU.

    Mirrors ``optimize_relu_relation`` in auto_LiRPA's
    ``auto_LiRPA/optimize_graph.py`` (α,β-CROWN). That implementation
    operates on BoundedModule nodes; this one operates on our ONNX
    ComputeGraph with numpy weights.

    Detects: Conv(C_in → 2C, k×k) → ReLU → Conv(2C → C, 1×1) → ReLU
    where the expanded conv has adjacent paired filters (w, -w) with
    biases (b, -b), and the 1×1 conv has [+1, -1] entries recombining
    each pair into a single output channel.

    This is equivalent to the original Conv(C_in → C, k×k) → ReLU because:
        ReLU(z) - ReLU(-z) = z  for all z
    so the 1×1 recombination recovers the pre-ReLU value, and the second
    ReLU gives the same result as a single ReLU on the original.

    The 1×1 bias is absorbed into the fused bias:
        b_fused = b_orig + b_1x1
    """
    from .network import ConvNode, ReluNode
    folded_any = False
    topo = list(graph.topo_order)

    for i, name in enumerate(topo):
        node = graph.nodes.get(name)
        if node is None or node.op_type != 'Conv':
            continue

        # Pattern: Conv → ReLU → Conv(1x1) → ReLU
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

        # Each node must have exactly one consumer
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

        # Check: 1x1 kernel, output channels = half of expanded
        if A.shape[2] != 1 or A.shape[3] != 1:
            continue
        C_out = A.shape[0]
        C_exp = A.shape[1]
        if C_exp != 2 * C_out:
            continue

        # Check: A has [1, -1] block structure and biases are (b, -b)
        A_mat = A[:, :, 0, 0]  # (C, 2C)
        tol = 1e-5

        # Verify structure and extract original kernel
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

            # One should be +1 and other -1
            if not (abs(abs(v_pos) - 1.0) < tol and abs(abs(v_neg) - 1.0) < tol
                    and v_pos * v_neg < 0):
                valid = False
                break

            # Filters should be negatives of each other
            if not np.allclose(K_exp[i_pos], -K_exp[i_neg], atol=tol):
                valid = False
                break

            # Biases should be negatives of each other
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

        # Replace: update the expanded conv with original weights
        # The 1x1 bias gets absorbed: output = z + b_1x1, so
        # ReLU(z + b_1x1) = ReLU(w·x + b_orig + b_1x1)
        node.params['kernel'] = K_orig
        node.params['bias'] = b_orig + b_1x1

        # Rewire: relu2's consumers now point to relu1,
        # and relu1 now follows the (now-original) conv directly
        relu2_name = relu2_list[0].name
        for other in graph.nodes.values():
            other.inputs = [relu1.name if inp == relu2_name else inp
                            for inp in other.inputs]
        if graph.output_name == relu2_name:
            graph.output_name = relu1.name

        # Remove relu1's old successor chain: conv1x1 and relu2
        del graph.nodes[conv1x1.name]
        del graph.nodes[relu2_name]
        folded_any = True

    if folded_any:
        graph.topological_sort()
        from .onnx_loader import _infer_shapes, _precache_conv_tensors
        _infer_shapes(graph)
        _precache_conv_tensors(graph)

    return folded_any
