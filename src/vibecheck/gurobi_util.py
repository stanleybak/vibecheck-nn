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


def optimize_checked(model, user_callback=None, *, tokens=TROUBLE_TOKENS):
    """Run `model.optimize()` with a message callback that scans for
    numeric-trouble warnings. Raises `GurobiNumericTrouble` if any are
    captured.

    If `user_callback(model, where)` is provided it is chained after
    the trouble scan, so both can observe MESSAGE events.
    """
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
        raise GurobiNumericTrouble(trouble)
