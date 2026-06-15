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
        pads = self.params.get('pads')
        if pads is None:
            # Unknown padding spec (e.g. dynamic pads input the loader could
            # not const-resolve). Passing through silently would alias output
            # to input and be unsound for any real padding — refuse loudly.
            raise NotImplementedError(
                f'Pad {self.name!r}: pads not statically known; refusing '
                f'silent passthrough')
        val = self.params.get('constant_value', 0.0)
        if all(int(p) == 0 for p in pads):
            zono_state[self.name] = z        # exact identity
            return
        inp_shape = _get_spatial_shape(self, graph, len(z.center))
        n = len(pads) // 2
        if len(inp_shape) != 3 or n < 2:
            raise NotImplementedError(
                f'Pad {self.name!r}: non-zero pads {pads} on unsupported '
                f'shape {inp_shape} (only CHW spatial pads handled)')
        c4d = torch.tensor(z.center, dtype=torch_dt).reshape(
            1, *inp_shape)
        if n >= 4:
            torch_pad = (pads[3], pads[3 + n], pads[2], pads[2 + n])
        else:
            torch_pad = (pads[1], pads[1 + n], pads[0], pads[0 + n])
        out = F.pad(c4d, torch_pad, value=val)
        zono_state[self.name] = _point_zono(out.flatten().numpy().astype(z.dtype))


# ---------------------------------------------------------------------------
# Structure ops: Concat, Split, Slice, Gather
# ---------------------------------------------------------------------------

class ConcatNode(GraphNode):
    def infer_shape(self, input_shapes):
        # True N-D inference: out = live shape with dim[axis] summed over
        # ALL inputs (live + const). The old flat `(total,)` ignored const
        # inputs entirely and erased rank — downstream shape-sensitive ops
        # (Transpose on vit's CLS-token concat) then saw a bogus shape.
        live = [input_shapes.get(i) for i in self.inputs]
        axis = self.params.get('axis', 0)
        consts = self.params.get('const_inputs') or []
        if live and all(s is not None for s in live):
            base = live[0]
            a = axis if axis >= 0 else len(base) + axis
            if 0 <= a < len(base):
                total_ax = sum(s[a] for s in live)
                for _pos, arr in consts:
                    ash = np.asarray(arr).shape
                    if len(ash) == len(base):
                        total_ax += ash[a]
                out = list(base)
                out[a] = total_ax
                self.output_shape = tuple(out)
                return
        total = sum(_prod(s) for s in live if s is not None)
        for _pos, arr in consts:
            total += int(np.asarray(arr).size)
        if total > 0:
            self.output_shape = (total,)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        consts = self.params.get('const_inputs') or []
        parts = [get_input(inp) for inp in self.inputs]
        if not consts:
            # All-live concat (flat). NOTE: ignores axis — correct when the
            # concat axis is the outermost varying dim (the only form seen
            # in practice for all-live concats).
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
            return
        # Const inputs present (vit CLS-token prepend): place every chunk
        # (const blocks as exact points, live zonos) at its true flat
        # positions via index scatter — the old code silently DROPPED the
        # const blocks. Needs static shapes to resolve the axis layout.
        live_shapes = [graph.nodes[inp].output_shape if inp in graph.nodes
                       else graph.input_shape for inp in self.inputs]
        if any(s is None for s in live_shapes):
            raise NotImplementedError(
                f'Concat {self.name!r}: const inputs need static live '
                f'input shapes')
        rank = len(live_shapes[0])
        axis = self.params.get('axis', 0)
        a = axis if axis >= 0 else rank + axis
        const_by_pos = {int(p): np.asarray(arr, np.float64)
                        for p, arr in consts}
        n_positions = len(self.inputs) + len(consts)
        live_iter = iter(zip(parts, live_shapes))
        chunks = []     # (kind, shape, payload) in ONNX position order
        for p in range(n_positions):
            if p in const_by_pos:
                arr = const_by_pos[p]
                if arr.ndim != rank:
                    raise NotImplementedError(
                        f'Concat {self.name!r}: const rank {arr.ndim} != '
                        f'live rank {rank}')
                chunks.append(('const', arr.shape, arr))
            else:
                z, sh = next(live_iter)
                chunks.append(('live', sh, z))
        out_shape = list(live_shapes[0])
        out_shape[a] = sum(sh[a] for _, sh, _ in chunks)
        n_out = int(np.prod(out_shape))
        out_grid = np.arange(n_out).reshape(out_shape)
        max_k = max(p.generators.shape[1] for p in parts)
        _dt = parts[0].center.dtype
        center = np.zeros(n_out, dtype=_dt)
        gens = np.zeros((n_out, max_k), dtype=_dt)
        off = 0
        for kind, sh, payload in chunks:
            sl = [slice(None)] * rank
            sl[a] = slice(off, off + sh[a])
            oidx = out_grid[tuple(sl)].reshape(-1)
            if kind == 'const':
                center[oidx] = payload.reshape(-1).astype(_dt)
            else:
                center[oidx] = payload.center
                gens[oidx, :payload.generators.shape[1]] = payload.generators
            off += sh[a]
        zono_state[self.name] = DenseZonotope(center, gens)


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
        inp_shape = (input_shapes.get(self.inputs[0])
                     if self.inputs else None)
        if indices is None:
            if inp_shape is not None:
                self.output_shape = inp_shape
            return
        axis = int(self.params.get('axis', 0))
        if inp_shape is not None:
            # ONNX semantics: output = input.shape[:axis] +
            # indices.shape + input.shape[axis+1:]. 0-D indices drop
            # the gather axis.
            a = axis if axis >= 0 else len(inp_shape) + axis
            if 0 <= a < len(inp_shape):
                idx_shape = tuple(indices.shape) if indices.ndim > 0 else ()
                out = list(inp_shape[:a]) + list(idx_shape) + \
                    list(inp_shape[a + 1:])
                self.output_shape = tuple(out) if out else (1,)
                return
        # Fallback (input shape unknown): flat indices count.
        self.output_shape = (len(indices.flatten()),)

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
        from .onnx_optimizer import (drop_identity_pads,
                                     fold_conv,
                                     fold_gemm,
                                     fuse_gemm_reshape_conv)
        # Ungated: removing all-zero Pad nodes is an exact identity (TinyYOLO
        # carries Pad(pads=[0]*8) no-ops that otherwise hit gpu_graph's loud
        # NotImplementedError). Non-zero pads are kept and still raise.
        drop_identity_pads(self)
        if settings.optimize_relu_relation:
            fold_conv(self)
            fold_gemm(self)
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
        # True only immediately after a Gemm/MatMul, so a standalone Add(bias)
        # is folded into the linear layer ONLY when it directly follows it (no
        # activation in between). Folding a post-ReLU Add into the pre-ReLU
        # linear would be unsound (bias can't commute through ReLU).
        _prev_was_linear = False
        # Pending input-side affine x ↦ s⊙x + t from leading normalization
        # ops (Mul scale / Add offset BEFORE the first linear layer; cora's
        # cifar10/svhn nets carry a scalar Mul+Add preamble). Folded into the
        # first fc as W(s⊙x+t)+b = (W·diag(s))x + (W t + b). These ops used
        # to fall into the silent catch-all below — the whole milp_verify
        # engine (zono bounds, joint-α bbr, exact MILP, PGD forward) then ran
        # on the UNNORMALIZED network and false-verified cora cifar10 SAT
        # cases (img339 canary, 2026-06-09).
        _pend_s = None
        _pend_t = None

        for name in self.topo_order:
            node = self.nodes[name]
            if node.op_type == 'Conv':
                if _pend_s is not None or _pend_t is not None:
                    raise NotImplementedError(
                        f'gpu_layers: input affine (Mul/Add preamble) before '
                        f'Conv at {name!r} is not foldable here')
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
                _prev_was_linear = False
            elif node.op_type in ('Gemm', 'MatMul'):
                W = node.params['W']
                b = node.params['b']
                gW = torch.tensor(W, dtype=dtype, device=device)
                gb = torch.tensor(b, dtype=dtype, device=device)
                if _pend_s is not None or _pend_t is not None:
                    n_in = int(gW.shape[1])
                    if _pend_t is not None:
                        t = np.broadcast_to(
                            np.asarray(_pend_t, np.float64).ravel(),
                            (n_in,)).copy()
                        gb = gb + gW @ torch.as_tensor(t, dtype=dtype,
                                                       device=device)
                    if _pend_s is not None:
                        s = np.broadcast_to(
                            np.asarray(_pend_s, np.float64).ravel(),
                            (n_in,)).copy()
                        gW = gW * torch.as_tensor(s, dtype=dtype,
                                                  device=device).unsqueeze(0)
                    _pend_s = _pend_t = None
                layers.append({
                    'type': 'fc', 'W': gW, 'bias': gb,
                })
                gpu_k.append(None)
                gpu_W_fwd.append(gW)
                gpu_b_fwd.append(gb)
                layer_types.append(('fc', None))
                _prev_was_linear = True
            elif (node.op_type in ('Mul', 'Div') and not layers
                  and node.params.get('scale') is not None):
                # Div stores params['scale'] already inverted (y = x·scale).
                s = np.asarray(node.params['scale'], np.float64).ravel()
                _pend_s = s if _pend_s is None else _pend_s * s
                if _pend_t is not None:
                    _pend_t = _pend_t * s
            elif (node.op_type == 'Add' and not layers
                  and node.params.get('bias') is not None):
                t = np.asarray(node.params['bias'], np.float64).ravel()
                _pend_t = t if _pend_t is None else _pend_t + t
            elif (node.op_type == 'Sub' and not layers
                  and node.params.get('negate')
                  and node.params.get('bias') is not None):
                # y = bias − x: s ← −s, t ← bias − t
                b = np.asarray(node.params['bias'], np.float64).ravel()
                _pend_s = (np.asarray([-1.0]) if _pend_s is None
                           else -_pend_s)
                _pend_t = b if _pend_t is None else b - _pend_t
            elif (node.op_type == 'Sub' and not layers
                  and node.params.get('sub_val') is not None):
                # y = x − sub_val
                c = np.asarray(node.params['sub_val'], np.float64).ravel()
                _pend_t = -c if _pend_t is None else _pend_t - c
            elif (node.op_type == 'Add' and 'bias' in node.params and layers
                  and _prev_was_linear):
                # TF/Keras export emits the affine layer as MatMul + a
                # SEPARATE Add(bias) node — the MatMul's own params['b'] is
                # zero. Fold this Add's constant bias into the preceding
                # linear layer; without it the gpu_layers net (and thus the
                # milp_verify MILP) is bias-free and disagrees with the real
                # network (it found phantom counterexamples on safenlp,
                # blocking every UNSAT verification). The main gpu_graph
                # keeps Add as its own op, so zono/CROWN/PGD were unaffected.
                # `_prev_was_linear` guard: only fold when the Add directly
                # follows the linear (no ReLU between) — folding a post-ReLU
                # bias into the pre-ReLU layer would be unsound.
                add_b = torch.as_tensor(
                    np.asarray(node.params['bias']).reshape(-1),
                    dtype=dtype, device=device)
                if add_b.numel() == layers[-1]['bias'].numel():
                    layers[-1]['bias'] = (layers[-1]['bias']
                                          + add_b.reshape(layers[-1]['bias'].shape))
                    gpu_b_fwd[-1] = (gpu_b_fwd[-1]
                                     + add_b.reshape(gpu_b_fwd[-1].shape))
                # stays foldable: a chained bias-Add adds to the same layer
            elif node.op_type == 'Relu':
                if _pend_s is not None or _pend_t is not None:
                    # ReLU(s⊙x+t) followed by linear is NOT affine-foldable.
                    raise NotImplementedError(
                        f'gpu_layers: input affine (Mul/Add preamble) hits '
                        f'ReLU at {name!r} before any linear layer')
                _prev_was_linear = False
            elif node.op_type in ('Flatten', 'Reshape', 'Identity',
                                  'Squeeze', 'Unsqueeze'):
                # Shape-only on the flat vector — handled implicitly, but
                # breaks linear→Add adjacency (a later Add(bias) must not
                # fold back past it).
                _prev_was_linear = False
            else:
                # NEVER silently skip an op: gpu_layers feeds the milp_verify
                # MILP encoding, its zono/CROWN bounds and its PGD forward —
                # a dropped op means every consumer runs on a DIFFERENT
                # network (cora cifar10 false-verifies shipped from exactly
                # this fall-through).
                raise NotImplementedError(
                    f'gpu_layers: unsupported op {node.op_type!r} at '
                    f'{name!r}')

        if _pend_s is not None or _pend_t is not None:
            raise NotImplementedError(
                'gpu_layers: input affine (Mul/Add preamble) never folded — '
                'no linear layer found')

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
                # ONNX BN normalizes the channel dim of an N-D tensor.
                # Broadcast the per-channel affine to the full flat size and
                # emit it as canonical 'mul' + 'add' ops — both already have
                # handlers in EVERY chain (zono forward, CROWN backward,
                # α-CROWN, gen-LP, PGD), so no new op type is needed. (The
                # old dedicated 'bn' op type had handlers in only 3 of the
                # propagation paths, and its rank-3 channel broadcast
                # assumed a batchless (C,H,W) layout — wrong for the
                # batch-led (1,T,d) shapes in vit.) Channel axis: dim 1 for
                # batch-led shapes, dim 0 for batchless (C,H,W).
                C = int(np.asarray(factor).size)
                if inp_shape is not None and len(inp_shape) >= 2 \
                        and inp_shape[1] == C:
                    ch_axis = 1
                elif inp_shape is not None and len(inp_shape) >= 1 \
                        and inp_shape[0] == C:
                    ch_axis = 0
                else:
                    raise NotImplementedError(
                        f'BatchNormalization {name!r}: cannot locate the '
                        f'channel dim (shape={inp_shape}, C={C})')
                bshape = tuple(C if a == ch_axis else 1
                               for a in range(len(inp_shape)))
                factor_flat = np.ascontiguousarray(np.broadcast_to(
                    np.asarray(factor, np.float64).reshape(bshape),
                    inp_shape)).reshape(-1)
                offset_flat = np.ascontiguousarray(np.broadcast_to(
                    np.asarray(offset, np.float64).reshape(bshape),
                    inp_shape)).reshape(-1)
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                scale_name = f'{name}__bnscale'
                ops.append({
                    'name': scale_name, 'type': 'mul',
                    'inputs': inp_names, 'scale': factor_flat,
                })
                ops.append({
                    'name': name, 'type': 'add', 'inputs': [scale_name],
                    'is_merge': False, 'bias': offset_flat,
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
                _mm_in_shape = (self.nodes[node.inputs[0]].output_shape
                                if node.inputs[0] in self.nodes
                                else self.input_shape)
                if (_mm_in_shape is not None and len(_mm_in_shape) >= 3
                        and int(np.prod(_mm_in_shape[:-1])) > 1
                        and _mm_in_shape[-1] == W.shape[1]):
                    # N-D MatMul (e.g. vit tokens (1, T, K) @ W): ONNX
                    # contracts only the LAST dim, batched over the T
                    # leading rows. The flat equivalent is the
                    # block-diagonal kron(I_T, W) with the bias tiled per
                    # row — emitting plain W here silently computed a
                    # different (shape-broken) function.
                    T = int(np.prod(_mm_in_shape[:-1]))
                    W = np.kron(np.eye(T), np.asarray(W, np.float64))
                    b = np.tile(np.asarray(b, np.float64), T)
                gW = torch.tensor(W, dtype=dtype, device=device)
                gb = torch.tensor(b, dtype=dtype, device=device)
                inp_names = []
                for inp in node.inputs[:1]:
                    inp_names.append(inp if inp in computed else '__input__')
                ops.append({
                    'name': name, 'type': 'fc', 'inputs': inp_names,
                    'W': gW, 'bias': gb,
                    'W_np': np.asarray(W, np.float64),
                    'bias_np': np.asarray(b, np.float64),
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
                _add_bias = (node.params.get('bias', None)
                             if not is_merge else None)
                if not is_merge and _add_bias is None:
                    raise NotImplementedError(
                        f'Add {name!r}: single live input but no captured '
                        f'constant — emitting an identity would silently '
                        f'drop the addition')
                if _add_bias is not None:
                    # ONNX Add broadcasts the constant against the live
                    # N-D shape (e.g. a per-token (48,) bias on (1,17,48)
                    # in vit). The flat 'add' handlers do a plain
                    # elementwise add, so expand the constant to the full
                    # flat size here. No-op when sizes already match.
                    _in_sh = (self.nodes[node.inputs[0]].output_shape
                              if node.inputs and node.inputs[0] in self.nodes
                              else self.input_shape)
                    _ba = np.asarray(_add_bias, np.float64)
                    if (_in_sh is not None
                            and _ba.size != int(np.prod(_in_sh))):
                        _add_bias = np.ascontiguousarray(
                            np.broadcast_to(_ba, _in_sh)).reshape(-1)
                ops.append({
                    'name': name, 'type': 'add', 'inputs': inp_names,
                    'is_merge': is_merge,
                    'bias': _add_bias,
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
                # Sub(a, b). Three sub-cases:
                #   1. b is constant ('bias' in params): emit `sub` op
                #      with one input; bias subtracted in forward.
                #   2. a is constant ('negate' set, with 'bias'): emit
                #      `sub` op negating the live input then adding bias.
                #   3. both inputs computed (skip-connection-style): emit
                #      `sub_bilinear` so forward subtracts y_b from y_a.
                # nn4sys pensieve_*_parallel: output = MatMul1 - MatMul2,
                # both computed → case 3. Previously this fell through
                # case 1 silently and dropped MatMul2, producing wildly
                # wrong final outputs (101.5 vs correct 5.57).
                inp0_computed = (node.inputs[0] in computed
                                  or node.inputs[0] == self.input_name)
                inp1_computed = (len(node.inputs) > 1
                                  and (node.inputs[1] in computed
                                        or node.inputs[1] == self.input_name))
                bias = node.params.get('bias')
                if inp0_computed and inp1_computed:
                    ops.append({
                        'name': name, 'type': 'sub_bilinear',
                        'inputs': [node.inputs[0], node.inputs[1]],
                    })
                elif node.params.get('negate'):
                    # Negate form bias − x (a is the constant, live operand
                    # is inputs[1]). NO consumer ever implemented the old
                    # `negate` flag on type 'sub' — every chain computed
                    # x − bias, a silent SIGN FLIP — and the old emission
                    # additionally wired inputs[0] (the constant) as the
                    # live input. Emit the exact canonical equivalent
                    # mul(−1) + add(bias) instead, which every chain
                    # already supports.
                    if bias is None:
                        raise NotImplementedError(
                            f'Sub {name!r}: negate form without a captured '
                            f'constant')
                    # The loader normalizes negate-form inputs to
                    # [live] (the constant lives in params['bias']).
                    live = node.inputs[0] if inp0_computed else None
                    if live is None:
                        raise NotImplementedError(
                            f'Sub {name!r}: negate form but its live '
                            f'operand is not a computed tensor')
                    _ba = np.asarray(bias, np.float64)
                    _in_sh = (self.nodes[live].output_shape
                              if live in self.nodes else self.input_shape)
                    if (_in_sh is not None
                            and _ba.size != int(np.prod(_in_sh))):
                        _ba = np.ascontiguousarray(
                            np.broadcast_to(_ba, _in_sh)).reshape(-1)
                    neg_name = f'{name}__neg'
                    ops.append({
                        'name': neg_name, 'type': 'mul', 'inputs': [live],
                        'scale': np.array([-1.0]),
                    })
                    ops.append({
                        'name': name, 'type': 'add', 'inputs': [neg_name],
                        'is_merge': False, 'bias': _ba.reshape(-1),
                    })
                else:
                    # Plain x − const (live operand is inputs[0]). The
                    # loader stores the constant as 'sub_val' (or 'bias'
                    # in some forms) — the old emission read ONLY 'bias',
                    # so every sub_val-form Sub became bias=None and the
                    # consumers' None-guards silently turned it into an
                    # IDENTITY (acasxu's input_Sub is all-zero, which is
                    # the only reason that never showed).
                    if not inp0_computed:
                        raise NotImplementedError(
                            f'Sub {name!r}: live operand is neither '
                            f'computed nor the graph input')
                    const = node.params.get('sub_val')
                    if const is None:
                        const = bias
                    if const is None:
                        raise NotImplementedError(
                            f'Sub {name!r}: no captured constant — '
                            f'emitting an identity would silently drop '
                            f'the subtraction')
                    ops.append({
                        'name': name, 'type': 'sub',
                        'inputs': [node.inputs[0]],
                        'bias': np.asarray(const, np.float64),
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
                kH, kW = (tuple(kernel) if len(kernel) == 2
                          else (kernel[0], kernel[0]))
                sH, sW = (tuple(stride) if len(stride) == 2
                          else (stride[0], stride[0]))
                pH, pW = (tuple(padding[:2]) if len(padding) >= 2
                          else (padding[0], padding[0]))
                if node.op_type == 'AveragePool':
                    # AveragePool(k, s, pad=0) is EXACTLY a Conv with a fixed
                    # depthwise-uniform kernel — emit it as a 'conv' op so
                    # every downstream chain (zono forwards, CROWN backwards,
                    # gen-LP/MILP builders, PGD) supports it with zero extra
                    # handlers. C×C×kH×kW kernel is tiny at pool scales
                    # (TinyYOLO: 16·16·4 floats). Padded average pooling is
                    # NOT expressible this way under ONNX's default
                    # count_include_pad=0 — refuse loudly.
                    if (pH, pW) != (0, 0):
                        raise NotImplementedError(
                            f'gpu_graph: AveragePool {name!r} with non-zero '
                            f'padding {(pH, pW)} (count_include_pad '
                            f'semantics not expressible as plain conv)')
                    C = in_spatial[0]
                    k_eq = np.zeros((C, C, kH, kW), dtype=np.float64)
                    for c in range(C):
                        k_eq[c, c] = 1.0 / (kH * kW)
                    b_eq = np.zeros(C, dtype=np.float64)
                    H_in, W_in = in_spatial[1], in_spatial[2]
                    H_out = (H_in - kH) // sH + 1
                    W_out = (W_in - kW) // sW + 1
                    out_shape = (C, H_out, W_out)
                    oph = H_in - ((H_out - 1) * sH + kH)
                    opw = W_in - ((W_out - 1) * sW + kW)
                    ops.append({
                        'name': name, 'type': 'conv', 'inputs': inp_names,
                        'kernel': torch.tensor(k_eq, dtype=dtype,
                                               device=device),
                        'bias': torch.tensor(b_eq, dtype=dtype,
                                             device=device),
                        'kernel_np': k_eq,
                        'bias_np': b_eq,
                        'in_shape': in_spatial, 'out_shape': out_shape,
                        'stride': (sH, sW), 'padding': (0, 0),
                        'output_padding': (oph, opw),
                        'n_out': C * H_out * W_out,
                    })
                else:
                    ops.append({
                        'name': name, 'type': 'max_pool',
                        'inputs': inp_names,
                        'kernel': (kH, kW),
                        'stride': (sH, sW),
                        'padding': (pH, pW),
                        'in_shape': in_spatial,
                    })
                computed.add(name)

            elif node.op_type == 'Transpose':
                # Emit as the canonical 'slice' (flat-index permutation):
                # every chain (zono forward, CROWN backward, gen-LP, PGD,
                # forward-LiRPA) already supports flat_idx gathers, and the
                # old dedicated 'transpose' handlers had a silent
                # passthrough fallback when shape metadata was missing —
                # i.e. a real permutation treated as identity (unsound).
                inp = node.inputs[0]
                inp_names = [inp if inp in computed else '__input__']
                in_shape = (self.nodes[inp].output_shape
                            if inp in self.nodes else self.input_shape)
                perm = tuple(node.params.get('perm', ()))
                if in_shape is None or not perm \
                        or len(perm) != len(in_shape):
                    raise NotImplementedError(
                        f'Transpose {name!r}: needs a static input shape '
                        f'matching perm (shape={in_shape}, perm={perm})')
                flat_idx = np.arange(int(np.prod(in_shape))).reshape(
                    in_shape).transpose(perm).reshape(-1).astype(np.int64)
                ops.append({
                    'name': name, 'type': 'slice', 'inputs': inp_names,
                    'flat_idx': flat_idx,
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
                # Decompose into primitives (the alpha,beta-CROWN
                # 'complex node' treatment, auto_LiRPA softmax.py):
                #   softmax(x) = exp(x) * reciprocal(reduce_sum(exp(x)))
                # so every chain only needs the 1-D convex relaxations of
                # exp/reciprocal plus the bilinear product — no direct
                # softmax relaxation anywhere. Shapes are carried
                # explicitly (synthetic ops have no graph nodes for the
                # in_shapes_nd post-pass to find).
                inp = node.inputs[0]
                inp_names = [inp if inp in computed else '__input__']
                in_shape = (self.nodes[inp].output_shape
                            if inp in self.nodes else self.input_shape)
                if in_shape is None:
                    raise NotImplementedError(
                        f'Softmax {name!r}: need a static input shape '
                        f'for decomposition')
                sh = (tuple(in_shape[1:]) if in_shape[0] == 1
                      else tuple(in_shape))
                axis = int(node.params.get('axis', -1))
                if axis >= 0 and in_shape[0] == 1:
                    axis -= 1
                ax = axis if axis >= 0 else len(sh) + axis
                if ax < 0 or ax >= len(sh):
                    raise NotImplementedError(
                        f'Softmax {name!r}: axis {axis} outside shape {sh}')
                sh_keep = tuple(1 if i == ax else d
                                for i, d in enumerate(sh))
                exp_name = f'{name}__exp'
                sum_name = f'{name}__sum'
                rcp_name = f'{name}__recip'
                # Max-shift (alpha,beta-CROWN's fixed-index trick):
                # softmax(x) = softmax(x - x_k) EXACTLY for any fixed
                # per-row index k (shift invariance), and subtracting a
                # fixed coordinate is a linear op (gather + negate +
                # add-merge — all exact on zonotopes). With k ~= the
                # row argmax, exp inputs stay <= ~0 so the exp
                # relaxation operates in its tame range (ABC measured
                # [-10.2, +0.19] on the pgd nets vs unshifted scores).
                # k=0 placeholder here is still exact; the pipeline
                # retargets flat_idx to the center-point argmax
                # (retarget_softmax_shifts) before verification.
                n_sm = int(np.prod(sh))
                pos = np.arange(n_sm).reshape(sh)
                sl0 = [slice(None)] * len(sh)
                sl0[ax] = slice(0, 1)
                idx0 = np.broadcast_to(pos[tuple(sl0)], sh).reshape(-1)
                shift_name = f'{name}__shift'
                nsh_name = f'{name}__nshift'
                sub_name = f'{name}__shifted'
                ops.append({
                    'name': shift_name, 'type': 'slice',
                    'inputs': inp_names,
                    'flat_idx': idx0.copy(),
                    'softmax_axis': ax,
                    'in_shapes_nd': [sh], 'out_shape_nd': sh,
                })
                ops.append({
                    'name': nsh_name, 'type': 'mul',
                    'inputs': [shift_name], 'scale': -1.0,
                    'in_shapes_nd': [sh], 'out_shape_nd': sh,
                })
                ops.append({
                    'name': sub_name, 'type': 'add', 'is_merge': True,
                    'inputs': [inp_names[0], nsh_name],
                    'in_shapes_nd': [sh, sh], 'out_shape_nd': sh,
                })
                ops.append({
                    'name': exp_name, 'type': 'exp', 'inputs': [sub_name],
                    'in_shapes_nd': [sh], 'out_shape_nd': sh,
                })
                ops.append({
                    'name': sum_name, 'type': 'reduce_sum',
                    'inputs': [exp_name],
                    'axes': (ax,), 'keepdims': True,
                    'in_shapes_nd': [sh], 'out_shape_nd': sh_keep,
                })
                ops.append({
                    'name': rcp_name, 'type': 'reciprocal',
                    'inputs': [sum_name],
                    'in_shapes_nd': [sh_keep], 'out_shape_nd': sh_keep,
                })
                ops.append({
                    'name': name, 'type': 'mul_bilinear',
                    'inputs': [exp_name, rcp_name],
                    'in_shapes_nd': [sh, sh_keep], 'out_shape_nd': sh,
                })
                computed.add(name)

            elif node.op_type == 'ReduceMean':
                # ReduceMean over fixed axes is an exact linear map — emit
                # the equivalent 'fc' (W = averaging matrix) so EVERY
                # downstream chain (zono forward, CROWN backward, gen-LP,
                # MILP, PGD) supports it with zero new handlers (same move
                # as AveragePool-as-conv). vit_2023 pools tokens this way.
                inp = node.inputs[0]
                inp_names = [inp if inp in computed else '__input__']
                in_shape = (self.nodes[inp].output_shape
                            if inp in self.nodes else self.input_shape)
                if in_shape is None:
                    raise NotImplementedError(
                        f'ReduceMean {name!r}: input shape unknown')
                axes = node.params.get('axes')
                if not axes:
                    raise NotImplementedError(
                        f'ReduceMean {name!r}: missing axes (reduce-all '
                        f'unsupported here)')
                axes = {a if a >= 0 else len(in_shape) + a for a in axes}
                n_in = int(np.prod(in_shape))
                oshape = [1 if a in axes else d
                          for a, d in enumerate(in_shape)]
                n_out = int(np.prod(oshape))
                out_pos = np.arange(n_out).reshape(oshape)
                out_map = np.broadcast_to(out_pos, in_shape).reshape(-1)
                k = n_in // n_out
                W = np.zeros((n_out, n_in))
                W[out_map, np.arange(n_in)] = 1.0 / k
                b = np.zeros(n_out)
                ops.append({
                    'name': name, 'type': 'fc', 'inputs': inp_names,
                    'W': torch.tensor(W, dtype=dtype, device=device),
                    'bias': torch.tensor(b, dtype=dtype, device=device),
                    'W_np': W, 'bias_np': b,
                })
                computed.add(name)

            elif node.op_type == 'Slice':
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed else '__input__']
                inp_shape = (self.nodes[node.inputs[0]].output_shape
                              if node.inputs[0] in self.nodes
                              else self.input_shape)
                axes = node.params.get('axes', [0])
                starts = node.params.get('starts', [0])
                ends = node.params.get('ends', [None])
                # Resolve negative / None bounds to absolute indices.
                rng = list(range(int(np.prod(inp_shape)))) if inp_shape \
                    else None
                if inp_shape is not None:
                    slices = [slice(None)] * len(inp_shape)
                    for ax, s, e in zip(axes, starts, ends):
                        a = ax if ax >= 0 else len(inp_shape) + ax
                        if a >= len(inp_shape):
                            continue
                        dim = inp_shape[a]
                        if s < 0: s = dim + s
                        if e is None or e > dim: e = dim
                        if e < 0: e = dim + e
                        slices[a] = slice(int(s), int(e))
                    # Compute flat index permutation: which input
                    # positions survive the slice.
                    idx_grid = np.arange(int(np.prod(inp_shape))).reshape(
                        inp_shape)[tuple(slices)].reshape(-1)
                    flat_idx = idx_grid.astype(np.int64)
                else:
                    flat_idx = None
                ops.append({
                    'name': name, 'type': 'slice',
                    'inputs': inp_names,
                    'flat_idx': flat_idx,
                })
                computed.add(name)

            elif node.op_type == 'Concat':
                inp_names = []
                for inp in node.inputs:
                    if inp in computed or inp == self.input_name:
                        inp_names.append(inp)
                axis = int(node.params.get('axis', 0))
                consts = node.params.get('const_inputs') or []
                live_shape = (self.nodes[node.inputs[0]].output_shape
                              if node.inputs and node.inputs[0] in self.nodes
                              else self.input_shape)
                if consts and len(inp_names) == 1 and live_shape is not None:
                    # One live input + constant blocks (vit CLS-token
                    # prepend): an EXACT affine map — emit 'fc' with a 0/1
                    # placement matrix + the constants as bias, so every
                    # chain supports it. (The legacy 'concat' op drops
                    # const blocks in several propagation paths.)
                    rank = len(live_shape)
                    a = axis if axis >= 0 else rank + axis
                    n_positions = len(node.inputs) + len(consts)
                    const_by_pos = {int(p): np.asarray(arr, np.float64)
                                    for p, arr in consts}
                    live_pos = [p for p in range(n_positions)
                                if p not in const_by_pos]
                    assert len(live_pos) == 1
                    chunks = []   # (kind, axis_dim, payload) in ONNX order
                    total_ax = 0
                    for p in range(n_positions):
                        if p in const_by_pos:
                            arr = const_by_pos[p]
                            if arr.ndim != rank:
                                raise NotImplementedError(
                                    f'Concat {name!r}: const rank '
                                    f'{arr.ndim} != live rank {rank}')
                            chunks.append(('const', arr.shape[a], arr))
                            total_ax += arr.shape[a]
                        else:
                            chunks.append(('live', live_shape[a],
                                           live_shape))
                            total_ax += live_shape[a]
                    out_shape = list(live_shape)
                    out_shape[a] = total_ax
                    n_in = int(np.prod(live_shape))
                    n_out = int(np.prod(out_shape))
                    out_grid = np.arange(n_out).reshape(out_shape)
                    W = np.zeros((n_out, n_in))
                    bvec = np.zeros(n_out)
                    off = 0
                    for kind, d, payload in chunks:
                        sl = [slice(None)] * rank
                        sl[a] = slice(off, off + d)
                        oidx = out_grid[tuple(sl)].reshape(-1)
                        if kind == 'live':
                            W[oidx, np.arange(n_in)] = 1.0
                        else:
                            csh = list(live_shape)
                            csh[a] = d
                            bvec[oidx] = np.broadcast_to(
                                payload, csh).reshape(-1)
                        off += d
                    ops.append({
                        'name': name, 'type': 'fc', 'inputs': inp_names,
                        'W': torch.tensor(W, dtype=dtype, device=device),
                        'bias': torch.tensor(bvec, dtype=dtype,
                                             device=device),
                        'W_np': W, 'bias_np': bvec,
                    })
                    computed.add(name)
                else:
                    # All-live concat: every chain handles type 'concat' as
                    # a FLAT concatenation (axis ignored). That is exact
                    # iff all dims before the concat axis are singleton for
                    # every input — verify it here instead of silently
                    # producing a permuted (wrong) layout downstream.
                    for inp in node.inputs:
                        sh = (self.nodes[inp].output_shape
                              if inp in self.nodes else self.input_shape)
                        if sh is None:
                            continue
                        a = axis if axis >= 0 else len(sh) + axis
                        if any(int(d) != 1 for d in sh[:a]):
                            raise NotImplementedError(
                                f'Concat {name!r}: axis {axis} over shape '
                                f'{sh} is not flat-contiguous — the flat '
                                f"'concat' op would silently interleave "
                                f'wrong positions')
                    ops.append({
                        'name': name, 'type': 'concat',
                        'inputs': inp_names,
                        'axis': axis,
                    })
                    computed.add(name)

            elif node.op_type in ('Split', 'SplitOutput'):
                # Emit each Split chunk as an explicit Slice op. The
                # primary Split output is chunk 0, SplitOutput nodes
                # carry an `index` (1, 2, ...) selecting later chunks.
                # nn4sys mscn uses Split along axis=-1 to separate
                # features (chunk 0, size N) from mask (chunk 1, size 1).
                if node.op_type == 'Split':
                    src_input = node.inputs[0]
                    split = node.params.get('split')
                    axis = int(node.params.get('axis', 0))
                    chunk_idx = 0
                else:
                    # SplitOutput's `inputs[0]` is the parent Split node.
                    split_node = self.nodes.get(node.inputs[0])
                    if split_node is None:
                        chunk_idx = node.params.get('index', 0)
                        split = None
                        axis = 0
                        src_input = node.inputs[0]
                    else:
                        src_input = split_node.inputs[0]
                        split = split_node.params.get('split')
                        axis = int(split_node.params.get('axis', 0))
                        chunk_idx = int(node.params.get('index', 0))
                inp_names = [src_input if src_input in computed
                              else '__input__']
                # Resolve to flat indices into the input.
                src_shape = (self.nodes[src_input].output_shape
                              if src_input in self.nodes
                              else self.input_shape)
                if split is None or src_shape is None:
                    flat_idx = None
                else:
                    a = axis if axis >= 0 else len(src_shape) + axis
                    if a >= len(src_shape):
                        flat_idx = None
                    else:
                        start = sum(split[:chunk_idx])
                        end = start + split[chunk_idx]
                        slicer = [slice(None)] * len(src_shape)
                        slicer[a] = slice(start, end)
                        full = np.arange(int(np.prod(src_shape))).reshape(
                            src_shape)
                        flat_idx = full[tuple(slicer)].reshape(-1).astype(
                            np.int64)
                ops.append({
                    'name': name, 'type': 'slice',
                    'inputs': inp_names,
                    'flat_idx': flat_idx,
                })
                computed.add(name)

            elif node.op_type == 'ReduceSum':
                # Linear reduction along given axes. Used in nn4sys
                # mscn_* (sum features * mask along axis=1 for masked
                # mean). ONNX axes are relative to the WITH-batch shape;
                # our in_shapes_nd strips the leading 1-dim, so decrement
                # positive axes by 1 when batch was stripped.
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed
                              else '__input__']
                raw_axes = list(node.params.get('axes', []))
                inp_shape = (self.nodes[node.inputs[0]].output_shape
                              if node.inputs[0] in self.nodes
                              else self.input_shape)
                if inp_shape and inp_shape[0] == 1:
                    # Batch will be stripped; shift axes.
                    adj_axes = tuple(
                        (a - 1) if (isinstance(a, int) and a > 0) else
                        (a + len(inp_shape) - 1 if a < 0 else a)
                        for a in raw_axes
                    )
                else:
                    adj_axes = tuple(raw_axes)
                keepdims = int(node.params.get('keepdims', 1))
                ops.append({
                    'name': name, 'type': 'reduce_sum',
                    'inputs': inp_names,
                    'axes': adj_axes, 'keepdims': bool(keepdims),
                })
                computed.add(name)

            elif node.op_type == 'Div':
                # Bilinear Div. The const-divisor case is already
                # rewritten to `mul` by `onnx_loader.py` (sets
                # `params['scale'] = 1/c`); the bilinear case (both
                # inputs computed) emits `div_bilinear` and is handled
                # only when the denominator is a point zonotope per
                # disjunct (nn4sys mscn pattern).
                inp_names = []
                for inp in node.inputs[:2]:
                    if inp in computed or inp == self.input_name:
                        inp_names.append(inp)
                if 'scale' in node.params:
                    # Const-divisor: same as a mul by reciprocal.
                    ops.append({
                        'name': name, 'type': 'mul',
                        'inputs': [inp_names[0]],
                        'scale': node.params['scale'],
                    })
                else:
                    ops.append({
                        'name': name, 'type': 'div_bilinear',
                        'inputs': inp_names,
                    })
                computed.add(name)

            elif node.op_type == 'Pow':
                # Pow(x, exponent) with constant integer exponent.
                # Used in pensieve_*_parallel (cubic softmax-style
                # normalization: x^3 / sum(x^3)). x^p is monotonic for
                # odd p, convex for even p, with closed-form box bounds
                # on [lo, hi]. The forward zono dispatches to a chord
                # linearization that returns (c_out, g_out) with a new
                # error generator per output element.
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed
                              else '__input__']
                exp_raw = node.params.get('exponent', 2.0)
                ops.append({
                    'name': name, 'type': 'pow',
                    'inputs': inp_names,
                    'exponent': float(exp_raw),
                })
                computed.add(name)

            elif node.op_type == 'Gather':
                # Constant-index Gather: output = input.flatten()[flat_idx]
                # for indices resolved against the named axis. Used in
                # nn4sys pensieve_* models (select single timestep from
                # rollout buffer). Behaves like Slice — both are flat
                # index selections — but indices needn't be contiguous.
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed
                              else '__input__']
                indices = node.params.get('indices')
                axis = int(node.params.get('axis', 0))
                inp_shape = (self.nodes[node.inputs[0]].output_shape
                              if node.inputs[0] in self.nodes
                              else self.input_shape)
                if indices is not None and inp_shape is not None:
                    a = axis if axis >= 0 else len(inp_shape) + axis
                    if a >= len(inp_shape):
                        # axis past rank — fall through to identity
                        flat_idx = None
                    else:
                        idx_arr = np.asarray(indices).flatten().astype(np.int64)
                        # Map (axis-relative indices) to (flat indices in
                        # input). For each idx in idx_arr, the gathered
                        # slice is `input_nd[:..., idx, :...]` along `a`.
                        full = np.arange(int(np.prod(inp_shape))).reshape(
                            inp_shape)
                        # Build slice tuple with `idx_arr` along `a`
                        slicer = [slice(None)] * len(inp_shape)
                        slicer[a] = idx_arr
                        flat_idx = full[tuple(slicer)].reshape(-1).astype(
                            np.int64)
                else:
                    flat_idx = None
                ops.append({
                    'name': name, 'type': 'gather',
                    'inputs': inp_names,
                    'flat_idx': flat_idx,
                })
                computed.add(name)

            elif node.op_type == 'Neg':
                # Negation y = -x is an exact linear map. Emit the
                # scalar `mul` op with scale=-1.0 so every downstream
                # chain (zono forward, CROWN backward) handles it with no
                # new handler (same move as the Softmax max-shift negate
                # at __nshift). Used in lsnc_relu's controller saturation
                # (u - relu(u - hi) - relu(... )), where Neg flips the
                # sign of a ReLU-clamp residual.
                inp_names = [node.inputs[0]
                              if node.inputs[0] in computed
                              else '__input__']
                ops.append({
                    'name': name, 'type': 'mul',
                    'inputs': inp_names, 'scale': -1.0,
                })
                computed.add(name)

            # Known-safe passthrough ops (Identity / Dropout / Cast /
            # Shape / Unsqueeze) get aliased to their input. Any OTHER
            # op silently aliased here is a soundness hazard: e.g. Pow
            # silently skipped would let an `x^3` activation be treated
            # as `x`, and any α-CROWN/zono bound computed downstream is
            # vacuously wrong (observed on nn4sys pensieve_*_parallel
            # before the Pow handler was added — false "verified" lb of
            # ~167000 on a network whose actual Y_0 is 5.55). Raise
            # NotImplementedError so missing ops surface loudly.
            elif node.op_type in (
                    'Identity', 'Dropout', 'Cast', 'Shape',
                    'Unsqueeze', 'Squeeze'):
                if node.inputs:
                    src = node.inputs[0]
                    alias[name] = alias.get(src, src)
                computed.add(name)
            else:
                raise NotImplementedError(
                    f'gpu_graph: unsupported op {node.op_type!r} '
                    f'(name={name!r}). Silent passthrough would alias '
                    f'output to input and produce unsound zono / CROWN '
                    f'bounds. Add an explicit handler before using.')

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
            _explicit = op.get('in_shapes_nd')
            in_shapes = []
            for _k, inp in enumerate(op['inputs']):
                if inp in self.nodes and getattr(self.nodes[inp], 'output_shape', None):
                    in_shapes.append(_strip_batch(self.nodes[inp].output_shape))
                elif inp == self.input_name and self.input_shape:
                    in_shapes.append(_strip_batch(self.input_shape))
                elif (_explicit is not None and _k < len(_explicit)
                      and _explicit[_k] is not None):
                    # synthetic ops (softmax decomposition, BN pair, ...)
                    # carry their shapes explicitly — the graph has no
                    # node for them, so keep the emission-time value.
                    in_shapes.append(tuple(_explicit[_k]))
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

        # Fork points must reflect the EMITTED op graph, not just the
        # ComputeGraph nodes: synthetic decompositions (softmax exp ->
        # {reduce_sum, mul_bilinear}; max-shift S -> {slice, add}) create
        # op-level multi-consumers whose zonotopes must be copy-on-get
        # (in-place handlers like fc/conv mutate their input otherwise).
        _op_refs = {}
        for _op in ops:
            for _inp in _op['inputs']:
                _op_refs[_inp] = _op_refs.get(_inp, 0) + 1
        forks = set(forks) | {n for n, c in _op_refs.items() if c > 1}

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
