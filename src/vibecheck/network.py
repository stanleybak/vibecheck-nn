"""Network representation: ComputeGraph and GraphNode op subclasses."""

from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

from .zonotope import DenseZonotope


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _prod(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def _infer_conv_input_shape(flat_shape_or_size, kernel, transpose=False):
    """Infer (C, H, W) from a flat input size and conv kernel."""
    import math
    if isinstance(flat_shape_or_size, (tuple, list)):
        total = _prod(flat_shape_or_size)
    else:
        total = flat_shape_or_size
    C_in = kernel.shape[0] if transpose else kernel.shape[1]
    if total % C_in != 0:
        return (1, 1, total)
    spatial = total // C_in
    side = int(math.sqrt(spatial))
    if side * side == spatial:
        return (C_in, side, side)
    for h in range(side, 0, -1):  # h=1 always divides, so loop always returns
        if spatial % h == 0:
            return (C_in, h, spatial // h)


def _find_shared_gens(name_a, name_b, graph, gen_count):
    """Find the generator count at the fork point of two merging branches."""
    forks = graph.fork_points()

    def _ancestors(name):
        visited, stack, seen = [], [name], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            visited.append(n)
            if n in graph.nodes:
                for inp in graph.nodes[n].inputs:
                    stack.append(inp)
        return visited

    anc_a = _ancestors(name_a)
    anc_b_set = set(_ancestors(name_b))
    for anc in anc_a:
        if anc in anc_b_set and anc in forks:
            return gen_count[anc]
    # Graph input is always a common ancestor and fork for merge nodes
    return gen_count.get(graph.input_name, 0)


def _get_spatial_shape(node, graph, actual_len, kernel=None, transpose=False):
    """Resolve (C, H, W) input shape for torch spatial ops (no batch dim)."""
    inp_name = node.inputs[0]
    inp_shape = (graph.nodes[inp_name].output_shape
                 if inp_name in graph.nodes else graph.input_shape)
    # Strip batch if present: (1, C, H, W) -> (C, H, W)
    if len(inp_shape) == 4 and inp_shape[0] == 1:
        inp_shape = inp_shape[1:]
    if len(inp_shape) == 3 and _prod(inp_shape) == actual_len:
        return inp_shape
    if kernel is not None:
        return _infer_conv_input_shape(actual_len, kernel, transpose)
    return inp_shape


def _point_zono(center):
    """Create a zero-generator zonotope from a center value."""
    return DenseZonotope(center, np.zeros((len(center), 0), dtype=center.dtype))


def _require_point(node, z):
    """Raise if zonotope has generators (op only supports concrete execution)."""
    if z.generators.shape[1] > 0:
        raise NotImplementedError(
            f"Zonotope propagation not implemented for op "
            f"'{node.op_type}' (node '{node.name}', "
            f"{z.generators.shape[1]} generators)")


def _bilinear_point_op(z_a, z_b, op_fn, node, graph):
    """Apply op_fn on two point zonotopes with ND broadcast."""
    if len(z_a.center) == len(z_b.center):
        return _point_zono(op_fn(z_a.center, z_b.center))
    shape_a = (graph.nodes[node.inputs[0]].output_shape
               if node.inputs[0] in graph.nodes else graph.input_shape)
    shape_b = (graph.nodes[node.inputs[1]].output_shape
               if node.inputs[1] in graph.nodes else graph.input_shape)
    a_nd = z_a.center.reshape(shape_a) if _prod(shape_a) == len(z_a.center) else z_a.center
    b_nd = z_b.center.reshape(shape_b) if _prod(shape_b) == len(z_b.center) else z_b.center
    return _point_zono(op_fn(a_nd, b_nd).flatten())


def _broadcast_const_op(z, const, op_fn, node, graph):
    """Apply op_fn(center_nd, const) with numpy broadcasting, then flatten.

    If broadcasting changes size, only works for point zonotopes (0 generators).
    For same-size operations, generators are preserved.
    """
    inp_name = node.inputs[0]
    inp_shape = (graph.nodes[inp_name].output_shape
                 if inp_name in graph.nodes else graph.input_shape)

    # If the shape doesn't match the flat center size, fall back to flat op
    if _prod(inp_shape) != len(z.center):
        new_center = op_fn(z.center, const.flatten() if isinstance(const, np.ndarray) else const)
        z.center = new_center
        return z

    center_nd = z.center.reshape(inp_shape)
    result = op_fn(center_nd, const)
    new_center = result.flatten()

    if len(new_center) == len(z.center):
        # Same size — generators still valid
        z.center = new_center
        return z
    else:
        # Size changed via broadcast — generators can't be reused
        _require_point(node, z)
        return DenseZonotope(new_center, np.zeros((len(new_center), 0),
                                                   dtype=new_center.dtype))


# ---------------------------------------------------------------------------
# GraphNode base
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """Base class for all operations in the compute graph."""
    name: str
    op_type: str
    inputs: list
    params: dict = field(default_factory=dict)
    output_shape: tuple = None

    def infer_shape(self, input_shapes):
        """Default: same shape as first input."""
        if self.inputs and self.inputs[0] in input_shapes:
            self.output_shape = input_shapes[self.inputs[0]]

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        """Default: raise for unknown ops."""
        raise NotImplementedError(
            f"Op '{self.op_type}' not supported (node '{self.name}')")


# ---------------------------------------------------------------------------
# Passthrough / shape-changing ops
# ---------------------------------------------------------------------------

class PassthroughNode(GraphNode):
    """Flatten, Dropout, Identity — data unchanged, shape flattened."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            if len(inp) > 2:
                self.output_shape = (inp[0], _prod(inp[1:]))
            else:
                self.output_shape = inp

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        zono_state[self.name] = get_input(self.inputs[0])


class UnsqueezeNode(GraphNode):
    """Unsqueeze — insert size-1 dimensions. Data unchanged."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            axes = self.params.get('axes', [])
            out = list(inp)
            for a in sorted(axes):
                if a < 0:
                    a = len(out) + 1 + a
                out.insert(a, 1)
            self.output_shape = tuple(out)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        zono_state[self.name] = get_input(self.inputs[0])


class SqueezeNode(GraphNode):
    """Squeeze — remove size-1 dimensions. Data unchanged."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            axes = self.params.get('axes', None)
            if axes:
                out = [d for i, d in enumerate(inp) if i not in axes]
            else:
                out = [d for d in inp if d != 1]
            if not out:
                out = [1]
            self.output_shape = tuple(out)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        zono_state[self.name] = get_input(self.inputs[0])


class ReshapeNode(GraphNode):
    """Reshape — preserves data, changes shape metadata."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            target = self.params.get('shape')
            if target:
                total = _prod(inp)
                out = list(target)
                neg_idx = None
                known = 1
                for i, d in enumerate(out):
                    if d == -1:
                        neg_idx = i
                    elif d == 0:
                        if i < len(inp):
                            out[i] = inp[i]
                        known *= out[i]
                    else:
                        known *= d
                if neg_idx is not None and known > 0:
                    out[neg_idx] = total // known
                self.output_shape = tuple(out)
            else:
                self.output_shape = inp  # no target shape, keep as-is

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        zono_state[self.name] = get_input(self.inputs[0])


class SplitOutputNode(PassthroughNode):
    """Placeholder for Split's secondary outputs."""
    def infer_shape(self, input_shapes):
        # Get shape from parent Split node's params
        parent_shape = input_shapes.get(self.inputs[0])
        if parent_shape is None:
            return
        # Find parent Split's split sizes
        # We need to look this up from the graph, but we only have input_shapes.
        # The parent's infer_shape set its output_shape to the first split.
        # For secondary outputs, we compute from the full input to Split.
        # Since we don't have the graph here, use the passthrough shape.
        # The correct shape will be set during zonotope propagation by SplitNode.
        if len(parent_shape) > 2:
            self.output_shape = (parent_shape[0], _prod(parent_shape[1:]))
        else:
            self.output_shape = parent_shape


class TransposeNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            perm = self.params.get('perm')
            if perm is None:
                perm = list(range(len(inp) - 1, -1, -1))  # reverse
            if len(perm) == len(inp):
                self.output_shape = tuple(inp[p] for p in perm)
            else:
                self.output_shape = inp

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)
        perm = self.params.get('perm')
        if perm is None:
            perm = list(range(len(inp_shape) - 1, -1, -1))

        if len(perm) != len(inp_shape) or len(inp_shape) < 2:
            zono_state[self.name] = z
            return

        center = np.transpose(z.center.reshape(inp_shape), perm).flatten()
        n_gens = z.generators.shape[1]
        if n_gens > 0:
            g_nd = z.generators.reshape(*inp_shape, n_gens)
            g_nd = np.transpose(g_nd, list(perm) + [len(perm)])
            gens = g_nd.reshape(-1, n_gens)
        else:
            gens = np.zeros((len(center), 0))
        zono_state[self.name] = DenseZonotope(center, gens)


# ---------------------------------------------------------------------------
# Activation ops
# ---------------------------------------------------------------------------

class ReluNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        z_lo, z_hi = z.bounds()
        z.apply_relu(z_lo, z_hi, relu_type)
        zono_state[self.name] = z


class LeakyReluNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        alpha = self.params.get('alpha', 0.01)
        center = np.where(z.center >= 0, z.center, alpha * z.center)
        zono_state[self.name] = _point_zono(center)


class SigmoidNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = _point_zono(
            1.0 / (1.0 + np.exp(-z.center)))


class ClipNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        center = z.center.copy()
        if 'min' in self.params:
            center = np.maximum(center, self.params['min'])
        if 'max' in self.params:
            center = np.minimum(center, self.params['max'])
        zono_state[self.name] = _point_zono(center)


class SignNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = _point_zono(np.sign(z.center))


class SoftmaxNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        e = np.exp(z.center - z.center.max())
        zono_state[self.name] = _point_zono(e / e.sum())


class TanhNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = _point_zono(np.tanh(z.center))


class TrigNode(GraphNode):
    """Sin, Cos."""
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        fn = np.sin if self.op_type == 'Sin' else np.cos
        zono_state[self.name] = _point_zono(fn(z.center))


class PowNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        exp = self.params.get('exponent', 2.0)
        zono_state[self.name] = _point_zono(z.center ** exp)


class FloorNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = _point_zono(np.floor(z.center))


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------

class NegNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        z.center = -z.center
        z.generators = -z.generators
        zono_state[self.name] = z


class AddNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            bias = self.params.get('bias')
            if bias is not None and isinstance(bias, np.ndarray):
                try:
                    out = np.broadcast_shapes(inp, bias.shape)
                    self.output_shape = out
                    return
                except ValueError:
                    pass
            self.output_shape = inp

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        # Two computed inputs = skip connection merge
        if (len(self.inputs) == 2
                and (self.inputs[1] in graph.nodes
                     or self.inputs[1] == graph.input_name)):
            z_a = get_input(self.inputs[0])
            z_b = get_input(self.inputs[1])
            shared = _find_shared_gens(
                self.inputs[0], self.inputs[1], graph, gen_count)
            zono_state[self.name] = z_a.add(z_b, shared)
        else:
            z = get_input(self.inputs[0])
            bias = self.params.get('bias', 0)
            zono_state[self.name] = _broadcast_const_op(
                z, bias, np.add, self, graph)


class SubNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            const = self.params.get('sub_val')
            if const is None:
                const = self.params.get('bias')
            if const is not None and isinstance(const, np.ndarray):
                try:
                    self.output_shape = np.broadcast_shapes(inp, const.shape)
                    return
                except ValueError:
                    pass
            self.output_shape = inp

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        if len(self.inputs) == 2 and self.inputs[1] in graph.nodes:
            # Two computed inputs: a - b
            z_b = get_input(self.inputs[1])
            _require_point(self, z)
            _require_point(self, z_b)
            zono_state[self.name] = _bilinear_point_op(
                z, z_b, np.subtract, self, graph)
        elif self.params.get('negate'):
            bias = self.params.get('bias', 0)
            z.center = -z.center
            z.generators = -z.generators
            zono_state[self.name] = _broadcast_const_op(
                z, bias, np.add, self, graph)
        else:
            sub_val = self.params.get('sub_val', 0)
            zono_state[self.name] = _broadcast_const_op(
                z, sub_val, np.subtract, self, graph)


class MulNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            scale = self.params.get('scale')
            if scale is not None and isinstance(scale, np.ndarray):
                try:
                    self.output_shape = np.broadcast_shapes(inp, scale.shape)
                    return
                except ValueError:
                    pass
            self.output_shape = inp

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        if 'scale' in self.params:
            z = get_input(self.inputs[0])
            s = self.params['scale']
            zono_state[self.name] = _broadcast_const_op(
                z, s, np.multiply, self, graph)
        else:
            # Bilinear mul — point only, with ND broadcast
            z = get_input(self.inputs[0])
            _require_point(self, z)
            z_b = get_input(self.inputs[1])
            _require_point(self, z_b)
            zono_state[self.name] = _bilinear_point_op(
                z, z_b, np.multiply, self, graph)


class DivNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        if 'scale' in self.params:
            s = self.params['scale']  # already inverted
            zono_state[self.name] = _broadcast_const_op(
                z, s, np.multiply, self, graph)
        else:
            # Both inputs computed — point only, with ND broadcast
            z_b = get_input(self.inputs[1])
            _require_point(self, z)
            _require_point(self, z_b)
            zono_state[self.name] = _bilinear_point_op(
                z, z_b, np.divide, self, graph)


# ---------------------------------------------------------------------------
# Linear ops: Conv, ConvTranspose, Gemm/MatMul
# ---------------------------------------------------------------------------

class ConvNode(GraphNode):
    def infer_shape(self, input_shapes):
        kernel = self.params['kernel']
        C_out = kernel.shape[0]
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None

        if kernel.ndim == 3:
            # 1D conv: kernel (C_out, C_in, kW)
            kW = kernel.shape[2]
            sW = self.params['stride'][0]
            pW = self.params['padding'][0]
            if inp_shape is not None and len(inp_shape) == 3:
                _, C_in, W_in = inp_shape
            elif inp_shape is not None:
                W_in = _prod(inp_shape) // kernel.shape[1]
            else:
                W_in = 1
            W_out = (W_in + 2 * pW - kW) // sW + 1
            self.output_shape = (1, C_out, W_out)
        else:
            # 2D conv: kernel (C_out, C_in, kH, kW)
            kH, kW = kernel.shape[2], kernel.shape[3]
            sH, sW = self.params['stride']
            pH, pW = self.params['padding']
            if inp_shape is not None and len(inp_shape) == 4:
                _, C_in, H_in, W_in = inp_shape
            elif inp_shape is not None and len(inp_shape) == 3:
                C_in, H_in, W_in = inp_shape
            elif inp_shape is not None:
                C_in = kernel.shape[1]
                import math
                total = _prod(inp_shape)
                spatial = total // C_in if total > 0 else 1
                side = int(math.sqrt(spatial))
                H_in = W_in = side
            else:
                H_in = W_in = 1
            H_out = (H_in + 2 * pH - kH) // sH + 1
            W_out = (W_in + 2 * pW - kW) // sW + 1
            self.output_shape = (1, C_out, H_out, W_out)

    def precache_conv_layer(self, graph):
        """Pre-build and cache the conv layer tuple with torch tensors.

        Called during graph loading (after shape inference) so that
        zonotope_propagate pays no tensor-creation overhead.
        """
        inp_name = self.inputs[0]
        inp_shape = (graph.nodes[inp_name].output_shape
                     if inp_name in graph.nodes else graph.input_shape)
        n_elems = _prod(inp_shape)
        spatial = self._spatial_shape(graph, n_elems)
        kernel = self.params['kernel']
        stride = self.params['stride']
        padding = self.params['padding']
        if kernel.ndim == 3:
            kernel = kernel[:, :, np.newaxis, :]
            stride = (1, stride[0])
            padding = (0, padding[0])
        torch_dt = torch.float32 if graph.dtype == np.float32 else torch.float64
        cache_key = '_torch_kernel_f32' if torch_dt == torch.float32 else '_torch_kernel'
        bias_key = '_torch_bias_f32' if torch_dt == torch.float32 else '_torch_bias'
        self._conv_layer = (kernel, self.params['bias'], {
            'input_shape': spatial,
            'stride': stride,
            'padding': padding,
            cache_key: torch.tensor(kernel, dtype=torch_dt),
            bias_key: torch.tensor(self.params['bias'], dtype=torch_dt),
        })

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        n_gens, n_elems = z.generators.shape[1], len(z.center)
        if n_gens * n_elems > 5_000_000:
            raise NotImplementedError(
                f"Conv generator matrix too large ({n_gens} × {n_elems} > 5M) "
                f"at node '{self.name}'")
        if not hasattr(self, '_conv_layer'):
            self.precache_conv_layer(graph)
        z.propagate_linear(self._conv_layer)
        zono_state[self.name] = z

    def _spatial_shape(self, graph, n_elems):
        """Get (C, H, W) for torch conv2d. For 1D conv, returns (C, 1, W)."""
        inp_name = self.inputs[0]
        inp_shape = (graph.nodes[inp_name].output_shape
                     if inp_name in graph.nodes else graph.input_shape)
        kernel = self.params['kernel']
        if len(inp_shape) == 4:
            return inp_shape[1:]  # (C, H, W)
        if len(inp_shape) == 3 and kernel.ndim == 3:
            # 1D: (1, C, W) -> unsqueeze to (C, 1, W) for conv2d
            return (inp_shape[1], 1, inp_shape[2])
        if len(inp_shape) == 3:
            return inp_shape
        return _infer_conv_input_shape(n_elems, kernel)


class ConvTransposeNode(GraphNode):
    def infer_shape(self, input_shapes):
        kernel = self.params['kernel']
        C_out = kernel.shape[1]
        kH, kW = kernel.shape[2], kernel.shape[3]
        sH, sW = self.params['stride']
        pH, pW = self.params['padding']
        opH, opW = self.params.get('output_padding', (0, 0))
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape is not None and len(inp_shape) == 4:
            _, C_in, H_in, W_in = inp_shape
        elif inp_shape is not None and len(inp_shape) == 3:
            C_in, H_in, W_in = inp_shape
        else:
            C_in = kernel.shape[0]
            H_in = W_in = 1
        H_out = (H_in - 1) * sH - 2 * pH + kH + opH
        W_out = (W_in - 1) * sW - 2 * pW + kW + opW
        self.output_shape = (1, C_out, H_out, W_out)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        inp_shape = _get_spatial_shape(
            self, graph, len(z.center), self.params['kernel'], transpose=True)
        z.propagate_conv_transpose(
            self.params['kernel'], self.params['bias'], inp_shape,
            self.params['stride'], self.params['padding'],
            self.params.get('output_padding', (0, 0)))
        zono_state[self.name] = z


class GemmNode(GraphNode):
    """Gemm and MatMul with constant weight matrix."""
    def infer_shape(self, input_shapes):
        W = self.params['W']
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if W.ndim == 1 and inp is not None:
            # (..., K) @ (K,) -> (...)
            self.output_shape = inp[:-1] if len(inp) > 1 else (1,)
        elif W.ndim == 2 and inp is not None and len(inp) > 2:
            # ND matmul: (..., K) @ (K, M) -> (..., M) where W stored as (M, K)
            if inp[-1] == W.shape[1]:
                self.output_shape = inp[:-1] + (W.shape[0],)
            else:
                self.output_shape = (1, W.shape[0])
        elif W.ndim == 2:
            self.output_shape = (1, W.shape[0])
        else:
            self.output_shape = (1, _prod(W.shape[:-1])) if W.ndim > 0 else (1,)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        W = self.params['W']
        b = self.params['b']
        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)

        # Standard 2D case: W @ flat_input + b
        if W.ndim == 2 and W.shape[1] == len(z.center):
            z.propagate_linear((W, b))
            zono_state[self.name] = z
            return

        # 1D weight: (..., K) @ (K,) -> (...)
        if W.ndim == 1 and _prod(inp_shape) == len(z.center) and inp_shape[-1] == len(W):
            _require_point(self, z)
            center_nd = z.center.reshape(inp_shape)
            result = np.matmul(center_nd, W) + b
            zono_state[self.name] = _point_zono(result.flatten())
            return

        # ND matmul: (..., K) @ (K, M) -> (..., M)
        # W is stored as (M, K), so transpose for matmul: input @ W.T
        if W.ndim >= 2 and _prod(inp_shape) == len(z.center) and inp_shape[-1] == W.shape[1]:
            _require_point(self, z)
            center_nd = z.center.reshape(inp_shape)
            result = np.matmul(center_nd, W.T) + b
            zono_state[self.name] = _point_zono(result.flatten())
            return

        raise NotImplementedError(
            f"Gemm/MatMul dimension mismatch: W is {W.shape} but "
            f"input shape {inp_shape} (flat {len(z.center)}) at node '{self.name}'")


class MatMulBilinearNode(GraphNode):
    """MatMul with two computed inputs (no constant weight)."""
    def infer_shape(self, input_shapes):
        sa = input_shapes.get(self.inputs[0])
        sb = input_shapes.get(self.inputs[1])
        if sa is None or sb is None:
            return
        # (..., M, K) @ (..., K, N) → (..., M, N). Broadcast the
        # leading dims.
        if len(sa) < 2 or len(sb) < 2:
            return
        M, K_a = sa[-2], sa[-1]
        K_b, N = sb[-2], sb[-1]
        if K_a != K_b:
            # Inner dims don't match standard matmul rule. Leave shape
            # unset and let downstream ops/tests deal with it.
            return
        # Broadcast leading dims (simple case: equal or one is empty).
        lead_a = sa[:-2]; lead_b = sb[:-2]
        if lead_a == lead_b or not lead_a:
            lead = lead_b
        elif not lead_b:
            lead = lead_a
        else:
            # Conservative broadcast: match lengths.
            lead = tuple(max(a, b) for a, b in zip(lead_a, lead_b))
        self.output_shape = tuple(lead) + (M, N)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z_a = get_input(self.inputs[0])
        z_b = get_input(self.inputs[1])
        _require_point(self, z_a)
        _require_point(self, z_b)
        # ND matmul with broadcast
        shape_a = (graph.nodes[self.inputs[0]].output_shape
                   if self.inputs[0] in graph.nodes else graph.input_shape)
        shape_b = (graph.nodes[self.inputs[1]].output_shape
                   if self.inputs[1] in graph.nodes else graph.input_shape)
        a_nd = z_a.center.reshape(shape_a) if _prod(shape_a) == len(z_a.center) else z_a.center
        b_nd = z_b.center.reshape(shape_b) if _prod(shape_b) == len(z_b.center) else z_b.center
        zono_state[self.name] = _point_zono(np.matmul(a_nd, b_nd).flatten())


# ---------------------------------------------------------------------------
# BatchNorm (when not folded into preceding Conv/Gemm)
# ---------------------------------------------------------------------------

class BatchNormNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        scale = self.params['scale']
        bn_bias = self.params['bias']
        mean = self.params['mean']
        var = self.params['var']
        eps = self.params['epsilon']
        factor = scale / np.sqrt(var + eps)
        offset = -factor * mean + bn_bias
        # Broadcast per-channel to flat vector
        if len(factor) < len(z.center):
            C = len(factor)
            spatial = len(z.center) // C
            factor = np.repeat(factor, spatial)
            offset = np.repeat(offset, spatial)
        z.center = factor * z.center + offset
        z.generators = factor[:, None] * z.generators
        zono_state[self.name] = z


# ---------------------------------------------------------------------------
# Pooling / Pad (concrete execution only)
# ---------------------------------------------------------------------------

class MaxPoolNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape and len(inp_shape) >= 3:
            kH, kW = self.params['kernel_shape']
            sH, sW = self.params['stride']
            pH, pW = self.params['padding']
            # Handle both (C,H,W) and (1,C,H,W)
            if len(inp_shape) == 4:
                _, C, H_in, W_in = inp_shape
                self.output_shape = (1, C, (H_in+2*pH-kH)//sH+1, (W_in+2*pW-kW)//sW+1)
            else:
                C, H_in, W_in = inp_shape
                self.output_shape = (1, C, (H_in+2*pH-kH)//sH+1, (W_in+2*pW-kW)//sW+1)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):

        z = get_input(self.inputs[0])
        _require_point(self, z)
        torch_dt = torch.float32 if z.dtype == np.float32 else torch.float64
        inp_shape = _get_spatial_shape(self, graph, len(z.center))
        c4d = torch.tensor(z.center, dtype=torch_dt).reshape(1, *inp_shape)
        kH, kW = self.params['kernel_shape']
        sH, sW = self.params['stride']
        pH, pW = self.params['padding']
        out = F.max_pool2d(c4d, kernel_size=(kH, kW), stride=(sH, sW),
                           padding=(pH, pW))
        zono_state[self.name] = _point_zono(out.flatten().numpy().astype(z.dtype))


class AveragePoolNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape and len(inp_shape) >= 3:
            kH, kW = self.params['kernel_shape']
            sH, sW = self.params['stride']
            pH, pW = self.params['padding']
            if len(inp_shape) == 4:
                _, C, H_in, W_in = inp_shape
                self.output_shape = (1, C, (H_in+2*pH-kH)//sH+1, (W_in+2*pW-kW)//sW+1)
            else:
                C, H_in, W_in = inp_shape
                self.output_shape = (1, C, (H_in+2*pH-kH)//sH+1, (W_in+2*pW-kW)//sW+1)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):

        z = get_input(self.inputs[0])
        _require_point(self, z)
        torch_dt = torch.float32 if z.dtype == np.float32 else torch.float64
        inp_shape = _get_spatial_shape(self, graph, len(z.center))
        c4d = torch.tensor(z.center, dtype=torch_dt).reshape(1, *inp_shape)
        kH, kW = self.params['kernel_shape']
        sH, sW = self.params['stride']
        pH, pW = self.params['padding']
        out = F.avg_pool2d(c4d, kernel_size=(kH, kW), stride=(sH, sW),
                           padding=(pH, pW))
        zono_state[self.name] = _point_zono(out.flatten().numpy().astype(z.dtype))


class PadNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):

        z = get_input(self.inputs[0])
        _require_point(self, z)
        torch_dt = torch.float32 if z.dtype == np.float32 else torch.float64
        pads = self.params.get('pads', [])
        val = self.params.get('constant_value', 0.0)
        inp_shape = _get_spatial_shape(self, graph, len(z.center))
        if pads and len(inp_shape) == 3:
            c4d = torch.tensor(z.center, dtype=torch_dt).reshape(
                1, *inp_shape)
            n = len(pads) // 2
            if n >= 4:
                torch_pad = (pads[3], pads[3 + n], pads[2], pads[2 + n])
            elif n >= 2:
                torch_pad = (pads[1], pads[1 + n], pads[0], pads[0 + n])
            else:
                zono_state[self.name] = z
                return
            out = F.pad(c4d, torch_pad, value=val)
            zono_state[self.name] = _point_zono(out.flatten().numpy().astype(z.dtype))
        else:
            zono_state[self.name] = z


# ---------------------------------------------------------------------------
# Structure ops: Concat, Split, Slice, Gather
# ---------------------------------------------------------------------------

class ConcatNode(GraphNode):
    def infer_shape(self, input_shapes):
        total = 0
        for i_name in self.inputs:
            if i_name in input_shapes and input_shapes[i_name] is not None:
                total += _prod(input_shapes[i_name])
        if total > 0:
            self.output_shape = (total,)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        parts = [get_input(inp) for inp in self.inputs]
        max_k = max(p.generators.shape[1] for p in parts)
        centers, gens = [], []
        for p in parts:
            centers.append(p.center)
            k = p.generators.shape[1]
            if k < max_k:
                pad = np.zeros((p.generators.shape[0], max_k - k))
                gens.append(np.hstack([p.generators, pad]))
            else:
                gens.append(p.generators)
        zono_state[self.name] = DenseZonotope(
            np.concatenate(centers), np.vstack(gens))


class SplitNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            split_sizes = self.params.get('split')
            axis = self.params.get('axis', 0)
            if split_sizes and axis < len(inp):
                out = list(inp)
                out[axis] = split_sizes[0]
                self.output_shape = tuple(out)
            else:
                self.output_shape = inp

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        split_sizes = self.params.get('split', None)
        if not split_sizes:
            zono_state[self.name] = z
            return

        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)
        axis = self.params.get('axis', 0)

        # Split along the axis in ND, then flatten each part
        parts_center = []
        parts_gens = []
        offset = 0
        for s in split_sizes:
            slices = [slice(None)] * len(inp_shape)
            slices[axis] = slice(offset, offset + s)
            slices = tuple(slices)
            c_part = z.center.reshape(inp_shape)[slices].flatten()
            parts_center.append(c_part)
            n_gens = z.generators.shape[1]
            if n_gens > 0:
                g_part = z.generators.reshape(*inp_shape, n_gens)[slices].reshape(-1, n_gens)
            else:
                g_part = np.zeros((len(c_part), 0))
            parts_gens.append(g_part)
            offset += s

        # First part is this node
        zono_state[self.name] = DenseZonotope(parts_center[0], parts_gens[0])

        # Set SplitOutput children
        for succ_name, succ_node in graph.nodes.items():
            if (succ_node.op_type == 'SplitOutput'
                    and succ_node.inputs[0] == self.name):
                idx = succ_node.params['index']
                if idx < len(parts_center):
                    zono_state[succ_name] = DenseZonotope(
                        parts_center[idx], parts_gens[idx])
                    gen_count[succ_name] = z.generators.shape[1]


class SliceNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            self.output_shape = self._sliced_shape(inp)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)

        if inp_shape is not None and len(inp_shape) > 1:
            # ND slice
            axes = self.params.get('axes', [0])
            starts = self.params.get('starts', [0])
            ends = self.params.get('ends', [None])
            slices = [slice(None)] * len(inp_shape)
            for ax, s, e in zip(axes, starts, ends):
                a = ax if ax >= 0 else len(inp_shape) + ax
                if a >= len(inp_shape):
                    continue
                dim = inp_shape[a]
                if s < 0:
                    s = dim + s
                if e is None or e > dim:
                    e = dim
                if e < 0:
                    e = dim + e
                slices[a] = slice(s, e)
            slices = tuple(slices)
            center = z.center.reshape(inp_shape)[slices].flatten()
            n_gens = z.generators.shape[1]
            if n_gens > 0:
                g_nd = z.generators.reshape(*inp_shape, n_gens)
                gens = g_nd[slices].reshape(-1, n_gens)
            else:
                gens = np.zeros((len(center), 0))
            zono_state[self.name] = DenseZonotope(center, gens)
        else:
            # 1D fallback
            n = len(z.center)
            s = self.params.get('starts', [0])[0]
            e = self.params.get('ends', [n])[0]
            if e > n: e = n
            if s < 0: s = n + s
            if e < 0: e = n + e
            zono_state[self.name] = DenseZonotope(
                z.center[s:e], z.generators[s:e, :])

    def _sliced_shape(self, inp_shape):
        axes = self.params.get('axes', [0])
        starts = self.params.get('starts', [0])
        ends = self.params.get('ends', [None])
        out = list(inp_shape)
        for ax, s, e in zip(axes, starts, ends):
            a = ax if ax >= 0 else len(inp_shape) + ax
            if a >= len(out):
                continue
            dim = out[a]
            if s < 0: s = dim + s
            if e is None or e > dim: e = dim
            if e < 0: e = dim + e
            out[a] = e - s
        return tuple(out)


class GatherNode(GraphNode):
    def infer_shape(self, input_shapes):
        indices = self.params.get('indices', None)
        if indices is not None:
            self.output_shape = (len(indices.flatten()),)
        elif self.inputs and self.inputs[0] in input_shapes:
            self.output_shape = input_shapes[self.inputs[0]]

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        indices = self.params.get('indices', None)
        if indices is not None:
            idx = indices.flatten().astype(int)
            zono_state[self.name] = DenseZonotope(
                z.center[idx], z.generators[idx, :])
        else:
            zono_state[self.name] = z


# ---------------------------------------------------------------------------
# Reduce ops
# ---------------------------------------------------------------------------

class ReduceNode(GraphNode):
    """ReduceSum and ReduceMean."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            axes = self.params.get('axes')
            keepdims = self.params.get('keepdims', 1)
            if axes:
                out = list(inp)
                for a in sorted(axes, reverse=True):
                    if a < 0:
                        a = len(out) + a
                    if keepdims:
                        out[a] = 1
                    else:
                        out.pop(a)
                self.output_shape = tuple(out) if out else (1,)
            else:
                # Reduce all axes
                if keepdims:
                    self.output_shape = tuple(1 for _ in inp)
                else:
                    self.output_shape = (1,)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        axes = self.params.get('axes')
        keepdims = bool(self.params.get('keepdims', 1))

        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)

        if axes and _prod(inp_shape) == len(z.center):
            # ND reduce along specific axes
            reduce_fn = np.sum if self.op_type == 'ReduceSum' else np.mean
            center_nd = z.center.reshape(inp_shape)
            new_center = reduce_fn(center_nd, axis=tuple(axes),
                                   keepdims=keepdims).flatten()
            n_gens = z.generators.shape[1]
            if n_gens > 0:
                g_nd = z.generators.reshape(*inp_shape, n_gens)
                new_gens = reduce_fn(g_nd, axis=tuple(axes),
                                     keepdims=keepdims).reshape(-1, n_gens)
            else:
                new_gens = np.zeros((len(new_center), 0))
            zono_state[self.name] = DenseZonotope(new_center, new_gens)
        else:
            # Reduce all
            if self.op_type == 'ReduceSum':
                zono_state[self.name] = DenseZonotope(
                    np.array([z.center.sum()]),
                    z.generators.sum(axis=0, keepdims=True))
            else:
                zono_state[self.name] = DenseZonotope(
                    np.array([z.center.mean()]),
                    z.generators.mean(axis=0, keepdims=True))


# ---------------------------------------------------------------------------
# Other ops
# ---------------------------------------------------------------------------

class ResizeNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape is not None and 'scales' in self.params:
            scales = self.params['scales']
            if len(scales) == len(inp_shape):
                self.output_shape = tuple(
                    int(d * s) for d, s in zip(inp_shape, scales))
            else:
                self.output_shape = inp_shape
        elif inp_shape is not None:
            self.output_shape = inp_shape

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        scales = self.params.get('scales')
        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)
        if scales is None or len(inp_shape) != 4:
            # No-op fallback when scales/shape are missing.
            zono_state[self.name] = z
            return
        torch_dt = torch.float32 if z.dtype == np.float32 else torch.float64
        scale_h, scale_w = float(scales[2]), float(scales[3])
        # Resize each generator column (and the center) by repeating values
        # spatially. Nearest-mode = pure linear map; sound for zonotopes.
        c_4d = torch.as_tensor(z.center, dtype=torch_dt).reshape(*inp_shape)
        c_out = F.interpolate(c_4d, scale_factor=(scale_h, scale_w),
                                mode='nearest')
        z.center = c_out.flatten().numpy().astype(z.dtype)
        n_gen = z.generators.shape[1]
        if n_gen == 0:
            z.generators = np.zeros((z.center.shape[0], 0), dtype=z.dtype)
        else:
            g_batch = torch.as_tensor(z.generators.T, dtype=torch_dt).reshape(
                n_gen, *inp_shape[1:])  # (n_gen, C, H, W)
            g_4d = g_batch.unsqueeze(1) if g_batch.ndim == 3 else g_batch
            # We need (n_gen, C, H, W) — already correct if inp_shape was (N, C, H, W) with N=1
            if g_batch.shape[1] != inp_shape[1]:
                # Fallback: reshape via numel
                g_batch = torch.as_tensor(z.generators.T, dtype=torch_dt)
                g_batch = g_batch.reshape(n_gen, *inp_shape[1:])
            g_out = F.interpolate(g_batch, scale_factor=(scale_h, scale_w),
                                    mode='nearest')
            z.generators = g_out.reshape(n_gen, -1).numpy().T.astype(z.dtype)
        zono_state[self.name] = z


class ConstantOfShapeNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        val = self.params.get('value', 0.0)
        n = max(1, len(z.center))
        zono_state[self.name] = _point_zono(np.full(n, val))


class ShapeOpNode(GraphNode):
    """Shape op — outputs dimension sizes."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            self.output_shape = (len(inp),)
        else:
            self.output_shape = (1,)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = z


class MiscNode(GraphNode):
    """Fallback for Cast, Equal, Where, Expand, ScatterND, ArgMax, Min, Max."""
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = z


# ---------------------------------------------------------------------------
# Op registry: ONNX op_type string -> GraphNode subclass
# ---------------------------------------------------------------------------

OP_REGISTRY = {
    # Passthrough
    'Flatten': PassthroughNode,
    'Squeeze': SqueezeNode,
    'Unsqueeze': UnsqueezeNode,
    'Reshape': ReshapeNode,
    'Dropout': PassthroughNode,
    'Identity': PassthroughNode,
    'SplitOutput': SplitOutputNode,
    # Transpose (actual permutation)
    'Transpose': TransposeNode,
    # Activations
    'Relu': ReluNode,
    'LeakyRelu': LeakyReluNode,
    'Sigmoid': SigmoidNode,
    'Clip': ClipNode,
    'Sign': SignNode,
    'Softmax': SoftmaxNode,
    'Tanh': TanhNode,
    'Sin': TrigNode,
    'Cos': TrigNode,
    'Pow': PowNode,
    'Floor': FloorNode,
    # Arithmetic
    'Neg': NegNode,
    'Add': AddNode,
    'Sub': SubNode,
    'Mul': MulNode,
    'Div': DivNode,
    # Linear
    'Conv': ConvNode,
    'ConvTranspose': ConvTransposeNode,
    'Gemm': GemmNode,
    'MatMul': GemmNode,  # overridden to MatMulBilinearNode when no weight
    # BatchNorm
    'BatchNormalization': BatchNormNode,
    # Pooling
    'MaxPool': MaxPoolNode,
    'AveragePool': AveragePoolNode,
    'Pad': PadNode,
    # Structure
    'Concat': ConcatNode,
    'Split': SplitNode,
    'Slice': SliceNode,
    'Gather': GatherNode,
    # Reduce
    'ReduceSum': ReduceNode,
    'ReduceMean': ReduceNode,
    # Other
    'Resize': ResizeNode,
    'Upsample': ResizeNode,
    'ConstantOfShape': ConstantOfShapeNode,
    'Shape': ShapeOpNode,
    'Cast': MiscNode,
    'Equal': MiscNode,
    'Where': MiscNode,
    'Expand': MiscNode,
    'ScatterND': MiscNode,
    'ArgMax': MiscNode,
    'Min': MiscNode,
    'Max': MiscNode,
}


# ---------------------------------------------------------------------------
# ComputeGraph
# ---------------------------------------------------------------------------

class ComputeGraph:
    """DAG of operations loaded from ONNX.

    Nodes are keyed by their output tensor name. Traversal order is
    topological (Kahn's algorithm), cached after construction.
    """

    def __init__(self, dtype=np.float32):
        self.nodes = {}          # name -> GraphNode
        self.input_name = None
        self.output_name = None
        self.input_shape = None  # without batch dim
        self.topo_order = []
        self.dtype = dtype       # numpy dtype for computation

    @classmethod
    def from_onnx(cls, onnx_path, dtype=np.float32):
        """Load an ONNX model into a ComputeGraph."""
        from .onnx_loader import load_onnx
        return load_onnx(onnx_path, dtype=dtype)

    def optimize(self, settings):
        """Apply semantics-preserving rewrites gated by settings flags."""
        from .onnx_optimizer import (fold_relusplitter,
                                     fold_relusplitter_gemm,
                                     fuse_gemm_reshape_conv)
        if settings.optimize_relu_relation:
            fold_relusplitter(self)
            fold_relusplitter_gemm(self)
        if settings.fuse_gemm_conv:
            fuse_gemm_reshape_conv(self)

    def topological_sort(self):
        """Kahn's algorithm."""
        in_degree = {name: 0 for name in self.nodes}
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp in self.nodes:
                    in_degree[node.name] += 1

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order = []

        successors = {name: [] for name in self.nodes}
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp in self.nodes:
                    successors[inp].append(node.name)

        while queue:
            name = queue.popleft()
            order.append(name)
            for succ in successors[name]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        assert len(order) == len(self.nodes), \
            f"Cycle detected: sorted {len(order)} of {len(self.nodes)} nodes"
        self.topo_order = order

    def fork_points(self):
        """Return set of node names whose output feeds multiple consumers."""
        ref_count = {}
        for node in self.nodes.values():
            for inp in node.inputs:
                ref_count[inp] = ref_count.get(inp, 0) + 1
        return {name for name, count in ref_count.items() if count > 1}

    def predecessors(self, name):
        if name in self.nodes:
            return list(self.nodes[name].inputs)
        return []

    def successors(self, name):
        return [n.name for n in self.nodes.values() if name in n.inputs]

    def relu_nodes(self):
        return {name for name, node in self.nodes.items()
                if node.op_type == 'Relu'}

    def flat_size(self, name):
        if name == self.input_name:
            shape = self.input_shape
        else:
            shape = self.nodes[name].output_shape
        if shape is None:
            return 0
        return _prod(shape)

    def gpu_layers(self, device, dtype):
        """Extract sequential linear layers for BnB backward pass.

        Returns (gpu_layers_list, fwd_data) where:
        - gpu_layers_list: list of dicts with 'type', weights, shapes for backward
        - fwd_data: dict with gpu_k, gpu_W_fwd, gpu_b_fwd, layer_types for PGD forward
        """
        from .zonotope import conv_output_shape
        layers = []
        gpu_k, gpu_W_fwd, gpu_b_fwd, layer_types = [], [], [], []

        for name in self.topo_order:
            node = self.nodes[name]
            if node.op_type == 'Conv':
                kernel = node.params['kernel']
                bias = node.params['bias']
                if not hasattr(node, '_conv_layer'):
                    node.precache_conv_layer(self)
                _, _, conv_params = node._conv_layer
                in_shape = conv_params['input_shape']
                stride = conv_params['stride']
                padding = conv_params['padding']
                # Ensure 4D kernel for conv2d
                k = kernel
                if k.ndim == 3:
                    k = k[:, :, np.newaxis, :]
                out_shape = conv_output_shape(in_shape, k, conv_params)
                oph = in_shape[1] - ((out_shape[1] - 1) * stride[0]
                                     - 2 * padding[0] + k.shape[2])
                opw = in_shape[2] - ((out_shape[2] - 1) * stride[1]
                                     - 2 * padding[1] + k.shape[3])
                gk = torch.tensor(k, dtype=dtype, device=device)
                gb = torch.tensor(bias, dtype=dtype, device=device)
                layers.append({
                    'type': 'conv', 'kernel': gk, 'bias': gb,
                    'in_shape': in_shape, 'out_shape': out_shape,
                    'stride': stride, 'padding': padding,
                    'output_padding': (oph, opw),
                    'n_out': out_shape[0] * out_shape[1] * out_shape[2],
                })
                gpu_k.append(gk)
                gpu_b_fwd.append(gb)
                gpu_W_fwd.append(None)
                layer_types.append(('conv', {
                    'input_shape': in_shape, 'stride': stride,
                    'padding': padding,
                }))
            elif node.op_type in ('Gemm', 'MatMul'):
                W = node.params['W']
                b = node.params['b']
                gW = torch.tensor(W, dtype=dtype, device=device)
                gb = torch.tensor(b, dtype=dtype, device=device)
                layers.append({
                    'type': 'fc', 'W': gW, 'bias': gb,
                })
                gpu_k.append(None)
                gpu_W_fwd.append(gW)
                gpu_b_fwd.append(gb)
                layer_types.append(('fc', None))
            # Skip Relu, Flatten, Reshape etc. — handled implicitly

        fwd_data = {
            'gpu_k': gpu_k, 'gpu_W_fwd': gpu_W_fwd,
            'gpu_b_fwd': gpu_b_fwd, 'layer_types': layer_types,
        }
        return layers, fwd_data

    def gpu_graph(self, device, dtype):
        """Extract graph-structured ops for verification of networks with skip connections.

        Returns a dict with:
        - 'ops': list of dicts in topo order, each with 'name', 'type', 'inputs', + type-specific data
        - 'relu_names': ordered list of ReLU node names (hidden layers only, not output)
        - 'fork_points': set of node names with >1 consumer
        - 'n_relu': number of hidden ReLU layers
        - 'input_name': graph input tensor name
        - 'input_n': number of input neurons (flat)
        """
        from .zonotope import conv_output_shape
        import torch

        ops = []
        relu_idx = 0
        relu_names = []
        forks = self.fork_points()
        # Track which names are the output of a node (vs graph input or initializer)
        computed = {self.input_name}
        # Map skipped passthrough nodes (Dropout, Identity, Cast) to their
        # real upstream producer so downstream ops reference the actual data.
        # Without this, the verify_zono_bnb forward can't find state[skipped].
        alias = {}
        # Track per-name shape (excluding batch dim) for ops that need it
        # (matmul-bilinear, transpose, softmax, etc. need true N-D shape).
        if self.input_shape:
            shapes_by_name = {self.input_name: tuple(d for d in self.input_shape if d != 1) or (self.input_shape[-1],)}
        else:
            shapes_by_name = {self.input_name: None}

        # Determine which ReLU is the last hidden one (before the final linear layer)
        # by checking: if ReLU's successor is the output node, it's still hidden;
        # but ReLU IS hidden only if it's followed by more computation.
        # Strategy: collect all ReLU names first, then strip the last one only if
        # the output node is a ReLU (which would make it not hidden).
        # Actually simpler: every Relu before the output linear layer is hidden.

        for name in self.topo_order:
            node = self.nodes[name]

            if node.op_type == 'Conv':
                kernel = node.params['kernel']
                bias = node.params['bias']
                if not hasattr(node, '_conv_layer'):
                    node.precache_conv_layer(self)
                _, _, conv_params = node._conv_layer
                in_shape = conv_params['input_shape']
                stride = conv_params['stride']
                padding = conv_params['padding']
                k = kernel
                if k.ndim == 3:
                    k = k[:, :, np.newaxis, :]
                out_shape = conv_output_shape(in_shape, k, conv_params)
                oph = in_shape[1] - ((out_shape[1] - 1) * stride[0]
                                     - 2 * padding[0] + k.shape[2])
                opw = in_shape[2] - ((out_shape[2] - 1) * stride[1]
                                     - 2 * padding[1] + k.shape[3])
                gk = torch.tensor(k, dtype=dtype, device=device)
                gb = torch.tensor(bias, dtype=dtype, device=device)
                # Map input names: use graph-level names
                inp_names = []
                for inp in node.inputs[:1]:  # only first input is the tensor
                    inp_names.append(inp if inp in computed else '__input__')
                ops.append({
                    'name': name, 'type': 'conv', 'inputs': inp_names,
                    'kernel': gk, 'bias': gb,
                    'kernel_np': k.astype(np.float64),
                    'bias_np': bias.astype(np.float64),
                    'in_shape': in_shape, 'out_shape': out_shape,
                    'stride': stride, 'padding': padding,
                    'output_padding': (oph, opw),
                    'n_out': out_shape[0] * out_shape[1] * out_shape[2],
                })
                computed.add(name)

            elif node.op_type == 'ConvTranspose':
                # Kernel layout: (C_in, C_out, kH, kW)
                kernel = node.params['kernel']
                bias = node.params['bias']
                stride = node.params['stride']
                padding = node.params['padding']
                output_padding = node.params.get('output_padding', (0, 0))
                inp_name = node.inputs[0]
                inp_shape = (self.nodes[inp_name].output_shape
                              if inp_name in self.nodes else self.input_shape)
                if len(inp_shape) == 4:
                    in_spatial = inp_shape[1:]
                elif len(inp_shape) == 3:
                    in_spatial = inp_shape
                else:
                    in_spatial = (kernel.shape[0], 1, 1)
                C_in, C_out, kH, kW = kernel.shape
                sH, sW = stride; pH, pW = padding
                opH, opW = output_padding
                H_out = (in_spatial[1] - 1) * sH - 2 * pH + kH + opH
                W_out = (in_spatial[2] - 1) * sW - 2 * pW + kW + opW
                out_shape = (C_out, H_out, W_out)
                gk = torch.tensor(kernel, dtype=dtype, device=device)
                gb = torch.tensor(bias, dtype=dtype, device=device)
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                ops.append({
                    'name': name, 'type': 'conv_transpose',
                    'inputs': inp_names,
                    'kernel': gk, 'bias': gb,
                    'kernel_np': kernel.astype(np.float64),
                    'bias_np': bias.astype(np.float64),
                    'in_shape': in_spatial, 'out_shape': out_shape,
                    'stride': stride, 'padding': padding,
                    'output_padding': output_padding,
                    'n_out': out_shape[0] * out_shape[1] * out_shape[2],
                })
                computed.add(name)

            elif node.op_type in ('Sigmoid', 'Tanh'):
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                ops.append({
                    'name': name,
                    'type': 'sigmoid' if node.op_type == 'Sigmoid' else 'tanh',
                    'inputs': inp_names,
                    'layer_idx': relu_idx,
                })
                relu_names.append(name)
                relu_idx += 1
                computed.add(name)

            elif node.op_type in ('Resize', 'Upsample'):
                # Nearest-mode integer upsample on (N, C, H, W). Adjoint:
                # avg_pool2d(divisor_override=1). Bilinear/other modes not
                # yet supported here (cgan models use nearest only).
                scales = node.params.get('scales', None)
                if scales is None or len(scales) != 4:
                    raise NotImplementedError(
                        f'Resize/Upsample: scales required, got {scales}')
                sH, sW = int(scales[2]), int(scales[3])
                if scales[2] != sH or scales[3] != sW:
                    raise NotImplementedError(
                        f'Resize/Upsample: integer scale only, got {scales}')
                inp_name = node.inputs[0]
                inp_shape = (self.nodes[inp_name].output_shape
                              if inp_name in self.nodes else self.input_shape)
                if len(inp_shape) == 4:
                    in_spatial = inp_shape[1:]
                elif len(inp_shape) == 3:
                    in_spatial = inp_shape
                else:
                    raise NotImplementedError(
                        f'Resize: cannot infer spatial shape from {inp_shape}')
                C, H_in, W_in = in_spatial
                out_shape = (C, H_in * sH, W_in * sW)
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                ops.append({
                    'name': name, 'type': 'upsample',
                    'inputs': inp_names,
                    'scale': (sH, sW),
                    'in_shape': in_spatial,
                    'out_shape': out_shape,
                    'n_out': out_shape[0] * out_shape[1] * out_shape[2],
                })
                computed.add(name)

            elif node.op_type == 'BatchNormalization':
                # Per-channel affine: y[c, h, w] = factor[c]*x[c, h, w] + offset[c].
                scale = node.params['scale']
                bn_bias = node.params['bias']
                mean = node.params['mean']
                var = node.params['var']
                eps = node.params['epsilon']
                factor = scale / np.sqrt(var + eps)
                offset = -factor * mean + bn_bias
                inp_name = node.inputs[0]
                inp_shape = (self.nodes[inp_name].output_shape
                              if inp_name in self.nodes else self.input_shape)
                if len(inp_shape) == 4:
                    spatial = inp_shape[2] * inp_shape[3]
                elif len(inp_shape) == 3:
                    spatial = inp_shape[1] * inp_shape[2]
                else:
                    spatial = 1
                factor_flat = np.repeat(factor.astype(np.float64), spatial)
                offset_flat = np.repeat(offset.astype(np.float64), spatial)
                gf = torch.tensor(factor_flat, dtype=dtype, device=device)
                go = torch.tensor(offset_flat, dtype=dtype, device=device)
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                ops.append({
                    'name': name, 'type': 'bn',
                    'inputs': inp_names,
                    'factor': gf, 'offset': go,
                    'factor_np': factor_flat, 'offset_np': offset_flat,
                })
                computed.add(name)

            elif node.op_type in ('Gemm', 'MatMul'):
                # Bilinear MatMul: both inputs computed (no W param). Forward
                # only — no zonotope/CROWN bound implemented. PGD still works.
                if 'W' not in node.params:
                    inp_names = []
                    for inp in node.inputs[:2]:
                        inp_names.append(inp if inp in computed else '__input__')
                    ops.append({
                        'name': name, 'type': 'matmul_bilinear',
                        'inputs': inp_names,
                    })
                    computed.add(name)
                    continue
                W = node.params['W']
                b = node.params['b']
                gW = torch.tensor(W, dtype=dtype, device=device)
                gb = torch.tensor(b, dtype=dtype, device=device)
                inp_names = []
                for inp in node.inputs[:1]:
                    inp_names.append(inp if inp in computed else '__input__')
                ops.append({
                    'name': name, 'type': 'fc', 'inputs': inp_names,
                    'W': gW, 'bias': gb,
                    'W_np': W.astype(np.float64),
                    'bias_np': b.astype(np.float64),
                })
                computed.add(name)

            elif node.op_type == 'Relu':
                inp_names = [node.inputs[0]]
                ops.append({
                    'name': name, 'type': 'relu', 'inputs': inp_names,
                    'layer_idx': relu_idx,
                })
                relu_names.append(name)
                relu_idx += 1
                computed.add(name)

            elif node.op_type == 'Add':
                inp_names = []
                for inp in node.inputs:
                    if inp in computed:
                        inp_names.append(inp)
                    elif inp == self.input_name:
                        inp_names.append(inp)
                    # else: constant/initializer — handled by node params
                is_merge = len(inp_names) == 2
                ops.append({
                    'name': name, 'type': 'add', 'inputs': inp_names,
                    'is_merge': is_merge,
                    'bias': node.params.get('bias', None) if not is_merge else None,
                })
                computed.add(name)

            elif node.op_type in ('Flatten', 'Reshape'):
                # Passthrough — tracked so successors can find the right name
                inp_names = [node.inputs[0]]
                ops.append({
                    'name': name, 'type': 'reshape', 'inputs': inp_names,
                })
                computed.add(name)

            elif node.op_type == 'Sub':
                inp_names = [node.inputs[0] if node.inputs[0] in computed
                             else self.input_name]
                bias = node.params.get('bias')
                ops.append({
                    'name': name, 'type': 'sub', 'inputs': inp_names,
                    'bias': bias,
                })
                computed.add(name)

            elif node.op_type in ('AveragePool', 'MaxPool'):
                kernel = node.params.get('kernel_shape', (2, 2))
                # Loader normalises to 'stride' (singular) and 'padding'.
                stride = node.params.get('stride',
                                          node.params.get('strides', kernel))
                padding = node.params.get('padding',
                                            node.params.get('pads', (0, 0)))
                inp_name = node.inputs[0]
                inp_shape = (self.nodes[inp_name].output_shape
                              if inp_name in self.nodes else self.input_shape)
                if len(inp_shape) == 4:
                    in_spatial = inp_shape[1:]
                else:
                    in_spatial = inp_shape
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                ops.append({
                    'name': name,
                    'type': 'avg_pool' if node.op_type == 'AveragePool' else 'max_pool',
                    'inputs': inp_names,
                    'kernel': tuple(kernel) if len(kernel) == 2 else (kernel[0], kernel[0]),
                    'stride': tuple(stride) if len(stride) == 2 else (stride[0], stride[0]),
                    'padding': tuple(padding[:2]) if len(padding) >= 2 else (padding[0], padding[0]),
                    'in_shape': in_spatial,
                })
                computed.add(name)

            elif node.op_type == 'Transpose':
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                ops.append({
                    'name': name, 'type': 'transpose',
                    'inputs': inp_names,
                    'perm': tuple(node.params.get('perm', ())),
                })
                computed.add(name)

            elif node.op_type == 'Squeeze':
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                ops.append({
                    'name': name, 'type': 'squeeze',
                    'inputs': inp_names,
                    'axes': tuple(node.params.get('axes', ())),
                })
                computed.add(name)

            elif node.op_type == 'Mul':
                # Constant Mul: scalar/per-channel multiplier in params['scale'].
                # Variable Mul (both inputs computed): hadamard.
                inp_names = []
                for inp in node.inputs:
                    if inp in computed or inp == self.input_name:
                        inp_names.append(inp)
                is_bilinear = len(inp_names) == 2
                ops.append({
                    'name': name,
                    'type': 'mul_bilinear' if is_bilinear else 'mul',
                    'inputs': inp_names,
                    'scale': node.params.get('scale', None) if not is_bilinear else None,
                })
                computed.add(name)

            elif node.op_type == 'Softmax':
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                ops.append({
                    'name': name, 'type': 'softmax',
                    'inputs': inp_names,
                    'axis': int(node.params.get('axis', -1)),
                })
                computed.add(name)

            # Skip other ops (Identity, Dropout, etc.) — pass through to
            # the real producer via the alias map.
            else:
                if node.inputs:
                    src = node.inputs[0]
                    alias[name] = alias.get(src, src)
                computed.add(name)

        # Rewrite all emitted ops' inputs through the alias map so
        # references to skipped passthrough nodes (Dropout, Identity)
        # point at the real upstream producer.
        if alias:
            for op in ops:
                op['inputs'] = [alias.get(inp, inp) for inp in op['inputs']]

        # Attach shape metadata to each emitted op so shape-sensitive ops
        # in the forward pass (matmul_bilinear, transpose, softmax, …) can
        # reshape inputs back to N-D. `out_shape` and `in_shapes` exclude
        # the batch dim by convention.
        def _strip_batch(s):
            if s is None: return None
            if len(s) == 0: return s
            return tuple(s[1:]) if s[0] == 1 else tuple(s)
        for op in ops:
            n_obj = self.nodes.get(op['name'])
            if n_obj is not None and getattr(n_obj, 'output_shape', None):
                op['out_shape_nd'] = _strip_batch(n_obj.output_shape)
            in_shapes = []
            for inp in op['inputs']:
                if inp in self.nodes and getattr(self.nodes[inp], 'output_shape', None):
                    in_shapes.append(_strip_batch(self.nodes[inp].output_shape))
                elif inp == self.input_name and self.input_shape:
                    in_shapes.append(_strip_batch(self.input_shape))
                else:
                    in_shapes.append(None)
            op['in_shapes_nd'] = in_shapes

        # The last relu_name may actually be the last hidden relu.
        # If the output node is a linear layer (Gemm/Conv), then all ReLUs
        # are hidden. If the output is a ReLU, we don't count it as hidden.
        output_node = self.nodes.get(self.output_name)
        if output_node and output_node.op_type == 'Relu' and relu_names:
            relu_names.pop()
            # Remove layer_idx from last relu op
            for op in reversed(ops):
                if op['type'] == 'relu' and op['name'] == self.output_name:
                    del op['layer_idx']
                    break
            relu_idx -= 1

        n_input = 1
        if self.input_shape:
            n_input = 1
            for d in self.input_shape:
                n_input *= d

        return {
            'ops': ops,
            'relu_names': relu_names,
            'fork_points': forks,
            'n_relu': len(relu_names),
            'input_name': self.input_name,
            'input_n': n_input,
            'input_shape': self.input_shape,
        }

    def __repr__(self):
        return (f"ComputeGraph(input={self.input_name}, output={self.output_name}, "
                f"nodes={len(self.nodes)}, input_shape={self.input_shape})")

    def __str__(self):
        forks = self.fork_points()
        topo_idx = {name: i for i, name in enumerate(self.topo_order)}
        succ_map = {name: [] for name in self.nodes}
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp in succ_map:
                    succ_map[inp].append(node.name)

        lines = []
        lines.append(f"ComputeGraph: {len(self.nodes)} ops, "
                      f"input={self.input_shape}")
        lines.append(f"  input: {self.input_name}  shape={self.input_shape}")
        lines.append("")

        idx_w = len(str(len(self.topo_order)))
        for i, name in enumerate(self.topo_order):
            node = self.nodes[name]
            shape_str = str(node.output_shape) if node.output_shape else '?'
            flat = _prod(node.output_shape) if node.output_shape else 0

            pred_indices = []
            for inp in node.inputs:
                if inp in topo_idx:
                    pred_indices.append(str(topo_idx[inp]))
                elif inp == self.input_name:
                    pred_indices.append('in')
            pred_str = ','.join(pred_indices) if pred_indices else 'in'

            succ_indices = [str(topo_idx[s]) for s in succ_map[name]
                            if s in topo_idx]
            succ_str = ','.join(succ_indices) if succ_indices else 'out'

            key_params = []
            if node.op_type == 'Conv':
                k = node.params.get('kernel')
                if k is not None:
                    key_params.append(f'kernel={k.shape}')
                key_params.append(f's={node.params.get("stride")}')
                key_params.append(f'p={node.params.get("padding")}')
            elif node.op_type == 'ConvTranspose':
                k = node.params.get('kernel')
                if k is not None:
                    key_params.append(f'kernel={k.shape}')
                key_params.append(f's={node.params.get("stride")}')
            elif node.op_type in ('Gemm', 'MatMul'):
                W = node.params.get('W')
                if W is not None:
                    key_params.append(f'W={W.shape}')
            elif node.op_type in ('MaxPool', 'AveragePool'):
                key_params.append(f'k={node.params.get("kernel_shape")}')
                key_params.append(f's={node.params.get("stride")}')
            elif node.op_type == 'LeakyRelu':
                key_params.append(f'alpha={node.params.get("alpha", 0.01)}')
            elif node.op_type == 'Transpose':
                key_params.append(f'perm={node.params.get("perm")}')
            param_str = f'  {" ".join(key_params)}' if key_params else ''

            fork_marker = ' *' if name in forks else ''
            lines.append(
                f"  [{i:>{idx_w}}] {node.op_type:20s} "
                f"{shape_str:>16s} ({flat:>6d})  "
                f"<-[{pred_str:>5s}]  ->[{succ_str:>5s}]"
                f"{fork_marker}{param_str}")

        lines.append("")
        lines.append(f"  output: {self.output_name}")
        if forks:
            fork_names = [f"[{topo_idx[f]}]" for f in forks if f in topo_idx]
            lines.append(f"  fork points: {', '.join(fork_names)}")
        return '\n'.join(lines)
