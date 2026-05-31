"""Gurobi solve wrapper that detects numeric-trouble warnings.

Motivation: on numerically fragile models, Gurobi can silently certify
wrong bounds — NumericFocus=2 on our dense gen-LP formulation returned
ObjBound=+0.034 on a problem whose true bound is -0.355, with no
queryable attribute flagging the issue. The *only* reliable signal was
warning lines streamed into the solver log:

    Warning: Markowitz tolerance tightened to 0.5
    Warning: N variables dropped from basis
    Warning: switch to quad precision
    Warning: max constraint violation (...) exceeds tolerance

We watch the log stream via a message callback and raise if any of
these appear. The alternative (parsing a log file after the fact) is
fragile and racy; the callback gives us deterministic in-process
capture.

Every place that invokes `model.optimize()` in this package should use
`optimize_checked` instead.
"""
import gurobipy as grb


TROUBLE_TOKENS = (
    'variables dropped from basis',
    'switch to quad precision',
)


class GurobiNumericTrouble(RuntimeError):
    """Gurobi emitted numeric-trouble warnings during optimize().

    The captured log lines are in `self.lines`.
    """

    def __init__(self, lines):
        # lines may be list[str] (normal) or a str (pickle roundtrip via
        # multiprocessing reconstructs us via cls(*args) with the
        # formatted message; detect that case).
        self.lines = [lines] if isinstance(lines, str) else list(lines)
        preview = '; '.join(self.lines[:3])
        super().__init__(
            f'Gurobi numeric trouble ({len(self.lines)} warning(s)): {preview}')

    def __reduce__(self):
        # Make pickle roundtrip preserve self.lines as a list.
        return (type(self), (self.lines,))


class GurobiZeroFixedObjVar(RuntimeError):
    """A variable fixed to [0, 0] carries nonzero objective weight.

    This is the signature of the gen-LP "orphan column" soundness bug: a
    zonotope noise-symbol column (which MUST be free e ∈ [-1, 1]) was pinned
    to [0, 0]. Pinned, the column contributes 0 to the objective instead of
    swinging ±|coef|, so the minimization's optimum is too HIGH — it
    under-approximates the output zonotope and can certify a false UNSAT (it
    false-verified dist_shift index4312). A legitimately fixed-at-zero
    variable (e.g. a dead-neuron bias output) carries *zero* objective weight
    and is not flagged. See `set_zero_fixed_obj_var_check`.
    """

    def __init__(self, names, total):
        self.names = list(names)
        self.total = int(total)
        preview = ', '.join(self.names[:8])
        more = ' ...' if total > len(self.names[:8]) else ''
        super().__init__(
            f'{total} variable(s) fixed to [0,0] with nonzero objective '
            f'weight — a generator/noise-symbol column pinned to [0,0] '
            f'(the gen-LP orphan-column soundness bug). Offending: '
            f'{preview}{more}')


_NUM_TROUBLE_WARNED_ONCE = False

# When True, `optimize_checked` raises GurobiZeroFixedObjVar if the model has
# any variable fixed to [0, 0] that carries nonzero objective weight. OFF by
# default (a per-solve scan over every variable has a cost); enabled during the
# test suite (see tests/conftest.py) as a soundness regression guard against
# the gen-LP [0,0]-column bug class. Toggle via `set_zero_fixed_obj_var_check`.
_CHECK_ZERO_FIXED_OBJ_VARS = False
_ZERO_FIX_TOL = 1e-12


def set_zero_fixed_obj_var_check(enabled):
    """Enable/disable the [0,0]-fixed-objective-variable guard in
    `optimize_checked`. Returns the previous value (so callers can restore
    it). Process-local — subprocess Gurobi workers must enable it
    themselves."""
    global _CHECK_ZERO_FIXED_OBJ_VARS
    prev = _CHECK_ZERO_FIXED_OBJ_VARS
    _CHECK_ZERO_FIXED_OBJ_VARS = bool(enabled)
    return prev


def _assert_no_zero_fixed_obj_vars(model):
    """Raise GurobiZeroFixedObjVar if any variable is fixed to [0,0] AND has a
    nonzero objective coefficient. Batched getAttr keeps it ~O(n) C-side."""
    model.update()
    vars_ = model.getVars()
    if not vars_:
        return
    import numpy as np
    lb = np.asarray(model.getAttr('LB', vars_), dtype=np.float64)
    ub = np.asarray(model.getAttr('UB', vars_), dtype=np.float64)
    obj = np.asarray(model.getAttr('Obj', vars_), dtype=np.float64)
    mask = ((np.abs(lb) <= _ZERO_FIX_TOL) & (np.abs(ub) <= _ZERO_FIX_TOL)
            & (np.abs(obj) > _ZERO_FIX_TOL))
    if mask.any():
        idx = np.nonzero(mask)[0]
        names = [vars_[int(i)].VarName for i in idx[:8]]
        raise GurobiZeroFixedObjVar(names, int(mask.sum()))


def optimize_checked(model, user_callback=None, *, tokens=TROUBLE_TOKENS,
                     tolerate_numeric_warnings=False):
    """Run `model.optimize()` with a message callback that scans for
    numeric-trouble warnings. Raises `GurobiNumericTrouble` if any are
    captured — unless `tolerate_numeric_warnings=True`, in which case
    trouble is logged (via a one-shot `print`) and recorded on the
    model as `model._num_trouble_lines` without raising.

    The caller can downstream-check `getattr(model, "_num_trouble", False)`
    to see if this solve experienced trouble, and propagate that flag
    up to any final return object / report to the user.

    If `user_callback(model, where)` is provided it is chained after
    the trouble scan, so both can observe MESSAGE events.
    """
    global _NUM_TROUBLE_WARNED_ONCE
    if _CHECK_ZERO_FIXED_OBJ_VARS:
        _assert_no_zero_fixed_obj_vars(model)
    trouble = []

    def cb(m, where):
        if where == grb.GRB.Callback.MESSAGE:
            msg = m.cbGet(grb.GRB.Callback.MSG_STRING)
            for t in tokens:
                if t in msg:
                    trouble.append(msg.rstrip())
                    break
        if user_callback is not None:
            user_callback(m, where)

    model.optimize(cb)
    if trouble:
        if tolerate_numeric_warnings:
            model._num_trouble = True
            model._num_trouble_lines = trouble
            if not _NUM_TROUBLE_WARNED_ONCE:
                print('[optimize_checked] numeric-trouble warnings '
                      'encountered in at least one Gurobi solve; '
                      'tolerating because tolerate_numeric_warnings=True. '
                      'First trouble lines: ' + '; '.join(trouble[:3]))
                _NUM_TROUBLE_WARNED_ONCE = True
            return
        raise GurobiNumericTrouble(trouble)
    model._num_trouble = False
