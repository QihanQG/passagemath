r"""Transportation-face embedding utilities for standard-form rational polytopes.

Embeds a rational polytope ``P = {y >= 0 : A*y = b}`` into a face of a 3-way transportation polytope [DLO2004]_.
Generates symbolic data to inspect the constructed tensor, plane marginals,
and variable coordinates. Stores the 3-way array as a list of ``h`` sparse
matrices over the Symbolic Ring (``SR``), representing each horizontal plane.

**Main Entry Points:**
- :func:`transportation_face_embedding_from_matrix` for raw ``A`` and ``b``.
- :func:`transportation_face_embedding` for a standard-form ``Polyhedron``.

Result objects expose the final tensor, active entries, marginals, and
coordinate maps. Intermediate states are preserved under ``.reduction``
and ``.embedding``.

REFERENCES:
- [DLO2004]_ De Loera, J. A.; Onn, S. "All Rational Polytopes Are
  Transportation Polytopes and All Polytopal Integer Sets Are Contingency
  Tables." IPCO 2004.
- [DLO2006]_ De Loera, J. A.; Onn, S. "All Linear and Integer Programs are
  Slim 3-Way Transportation Programs." SIAM J. Optim. 17 (2006), 806--821.

"""

from itertools import product

from sage.arith.functions import lcm
from sage.matrix.constructor import matrix
from sage.misc.latex import latex
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.structure.sage_object import SageObject
from sage.symbolic.assumptions import assume
from sage.symbolic.ring import SR

__all__ = [
    "coefficient_reduce",
    "coefficient_reduce_from_matrix",
    "polytope_3waytransportation",
    "polytope_3waytransportation_from_matrix",
    "transportation_face_embedding",
    "transportation_face_embedding_from_matrix",
    "CoefficientReductionResult",
    "TransportationEmbeddingResult",
    "TransportationFaceEmbeddingResult",
]


def _new_tensor_3d(shape):
    r"""
    Return the data structure holding the constructed 3-way array.

    The array of shape ``(r, c, h)`` is stored as a list of ``h`` sparse
    ``r x c`` matrices over ``SR``, one per horizontal plane
    ``k = 0, ..., h - 1``. Entry ``(i, j, k)`` is ``M[k][i, j]``.

    EXAMPLES::

        sage: from sage.geometry.polyhedron.transportation_face_embedding \
        ....:     import _new_tensor_3d
        sage: M = _new_tensor_3d((2, 2, 3))
        sage: len(M)
        3
        sage: M[0].nrows(), M[0].ncols()
        (2, 2)
        sage: M[0].is_sparse()
        True
    """
    r, c, h = shape
    return [matrix(SR, r, c, sparse=True) for _ in range(h)]


### Result objects ###

class _TransportationConstructionResult(SageObject):
    """Small Sage-style result object with attribute access."""

    _fields = ()

    def __init__(self, **kwds):
        missing = [field for field in self._fields if field not in kwds]
        if missing:
            raise ValueError(
                f"missing fields for {self.__class__.__name__}: {missing}"
            )

        extra = [field for field in kwds if field not in self._fields]
        if extra:
            raise ValueError(
                f"unknown fields for {self.__class__.__name__}: {extra}"
            )

        for field in self._fields:
            setattr(self, field, kwds[field])

    def as_dict(self):
        """Return a shallow dictionary copy of this result."""
        return {field: getattr(self, field) for field in self._fields}

    def _repr_(self):
        return f"{self.__class__.__name__} with fields {self._fields}"


class CoefficientReductionResult(_TransportationConstructionResult):
    _fields = (
        "new_constraints",
        "new_vars_list",
        "new_vars_grouped",
        "k_values",
        "sigma_1",
        "original_constraints",
        "original_vars_list",
        "A",
        "b",
        "n_chain",
    )

    def verify(self):
        r"""
        Check the Lemma 3.1 reduction invariants.

        Under `x[j,s] = 2^s * y[j]` the doubling-chain equations must reduce to
        `0 == 0` and the rewritten main equations must recover the rows of
        `A*y = b`. Return ``True`` or raise ``AssertionError``.
        """
        A, b, vars_list = self.A, self.b, self.original_vars_list
        sub_back = {
            self.new_vars_grouped[j][s]: 2 ** s * vars_list[j]
            for j in range(A.ncols())
            for s in range(self.k_values[j] + 1)
        }

        for c in range(self.n_chain):
            eq = self.new_constraints[c]
            diff = (eq.lhs() - eq.rhs()).subs(sub_back).expand()
            assert diff.is_trivial_zero(), f"Doubling chain check failed: {eq}"

        for i in range(A.nrows()):
            eq = self.new_constraints[self.n_chain + i]
            recovered = eq.lhs().subs(sub_back).expand()
            expected = sum(int(A[i, j]) * vars_list[j] for j in range(A.ncols()))
            assert (recovered - expected).expand().is_trivial_zero(), (
                f"Main equation row {i} failed: {recovered} != {expected}"
            )
            assert eq.rhs() == b[i, 0], f"Main equation rhs mismatch at row {i}"

        return True


class TransportationEmbeddingResult(_TransportationConstructionResult):
    _fields = (
        "tensor_M",
        "zero_set_S",
        "active_set_V",
        "marginals_u",
        "marginals_v",
        "marginals_w",
        "sigma_2",
        "constraints",
        "A",
        "b",
        "vars_list",
        "r_values",
        "R_partition",
        "k_plus",
        "k_minus",
        "slack_level",
        "copied_vars",
        "complements",
        "complement_rules",
        "copy_identification_rules",
        "U",
    )

    def verify(self):
        r"""
        Check the Theorem 3.2 embedding invariants.

        Each non-slack horizontal plane must recover one equation of `A*y = b`
        (after substituting `bar_y = U - y` and the stage-2 copies), and every
        `sigma_2` image must land in the enabled set `V`. Return ``True`` or
        raise ``AssertionError``.
        """
        A, b, vars_list = self.A, self.b, self.vars_list

        for k in range(A.nrows()):
            expected = sum(A[k, j] * vars_list[j]
                           for j in range(A.ncols())) - b[k, 0]
            recovered = (sum(self.tensor_M[k].list())
                         .subs(self.complement_rules)
                         .subs(self.copy_identification_rules))
            neg_sum = sum(abs(A[k, j]) for j in range(A.ncols()) if A[k, j] < 0)
            actual = (recovered - self.U * neg_sum - b[k, 0]).expand()
            assert (actual - expected).expand().is_trivial_zero(), (
                f"Plane {k} mismatch"
            )

        for y, coord in self.sigma_2.items():
            assert coord in self.active_set_V, f"sigma_2[{y}] does not land in V"

        return True


class TransportationFaceEmbeddingResult(_TransportationConstructionResult):
    _fields = (
        "tensor_M",
        "zero_set_S",
        "active_set_V",
        "marginals_u",
        "marginals_v",
        "marginals_w",
        "sigma",
        "reduction",
        "embedding",
    )

    def verify(self):
        r"""
        Check the full Theorem 1 result.

        Runs both stage verifications and checks the composition
        `sigma = sigma_2 after sigma_1`. Return ``True`` or raise
        ``AssertionError``.
        """
        self.reduction.verify()
        self.embedding.verify()

        for y in self.reduction.original_vars_list:
            expected = self.embedding.sigma_2[self.reduction.sigma_1[y]]
            assert self.sigma[y] == expected, f"sigma composition failed for {y}"

        return True


#### Normalizing input ####

def _b_to_column(b) -> list:
    r"""Normalize ``b`` into a flat list of entries."""
    if hasattr(b, "nrows") and hasattr(b, "ncols"):
        if b.ncols() == 1:
            return [b[i, 0] for i in range(b.nrows())]
        if b.nrows() == 1:
            return [b[0, j] for j in range(b.ncols())]
        raise ValueError("b must be a vector, list, row matrix, or column matrix")
    return list(b)


def _normalize_Ab(A, b):
    r"""
    Coerce ``A, b`` to integer matrix form by clearing rational denominators.

    Returns ``(A, b)`` with ``A`` an integer matrix and ``b`` an integer column
    matrix. Each row is scaled independently by the lcm of its denominators.
    """
    A = matrix(QQ, A)
    b_entries = [QQ(e) for e in _b_to_column(b)]

    if len(b_entries) != A.nrows():
        raise ValueError(
            f"length of b must equal number of rows of A: "
            f"got len(b)={len(b_entries)} and A.nrows()={A.nrows()}"
        )

    A_rows_int = []
    b_int = []
    for i in range(A.nrows()):
        row = [QQ(A[i, j]) for j in range(A.ncols())]
        bi = b_entries[i]
        denoms = [e.denominator() for e in row] + [bi.denominator()]
        L = lcm(denoms)
        if L != 1:
            row = [e * L for e in row]
            bi = bi * L
        A_rows_int.append([ZZ(e) for e in row])
        b_int.append(ZZ(bi))
        
    return matrix(ZZ, A_rows_int), matrix(ZZ, len(b_int), 1, b_int)


def _polyhedron_to_Ab(P):
    r"""
    Extract `(A, b)` from a Polyhedron in standard form `{y >= 0 : A*y = b}`.

    This checks standard-form semantically rather than syntactically. Sage may
    rewrite coordinate nonnegativity inequalities using the equality system, so
    inequalities in `P.inequalities_list()` do not necessarily appear literally
    as `y_i >= 0`.
    """
    from sage.geometry.polyhedron.constructor import Polyhedron

    base = P.base_ring()
    if base not in (QQ, ZZ):
        raise ValueError(
            f"Polyhedron base ring must be QQ or ZZ; got {base}. "
            f"This embedding requires a rational polytope."
        )

    if not P.is_compact():
        raise ValueError(
            "Polyhedron must be bounded (compact); this embedding is "
            "designed for rational polytopes."
        )

    eqns = list(P.equations_list())
    if not eqns:
        raise ValueError(
            "Polyhedron has no equality constraints; this embedding expects "
            "standard form {y >= 0 : A y = b} with a non-trivial equation "
            "system."
        )

    ambient_dim = P.ambient_dim()

    # Coordinate non-negativities y_i >= 0 as [constant, e_i].
    nonneg_ieqs = []
    for j in range(ambient_dim):
        row = [ZZ(0)] * ambient_dim
        row[j] = ZZ(1)
        nonneg_ieqs.append([ZZ(0)] + row)

    
    P_standard = Polyhedron(eqns=eqns, ieqs=nonneg_ieqs, base_ring=QQ,)

    if P != P_standard:
        raise ValueError(
            "Polyhedron is not equal to the standard-form polyhedron "
            "{y >= 0 : A y = b} determined by its equality constraints. "
            "General inequalities and upper bounds must be rewritten as "
            "equations using slack variables before calling this function."
        )

    A_rows = [eq[1:] for eq in eqns]
    b_entries = [-eq[0] for eq in eqns]
    return _normalize_Ab(matrix(QQ, A_rows), b_entries)


def _make_input_variables(n: int, prefix: str = "y") -> list:
    r"""Create symbolic variables ``y_1, ..., y_n`` for symbolic display."""
    return [
        SR.symbol(f"{prefix}{j + 1}",  latex_name=rf"{prefix}_{{{j + 1}}}")
        for j in range(n)
    ]


#### Helpers ####

def _k_j(A, j: int) -> int:
    r"""
    Return `k[j] = floor(log2(max_i(abs(A[i,j]))))`.

    If column ``j`` is zero, return ``0`` so the variable still gets one copy.
    """
    m = max((abs(int(A[i, j])) for i in range(A.nrows())), default=0)
    return ZZ(m).nbits() - 1 if m > 0 else 0


def _binary_digits(n: int, padto: int) -> list:
    r"""
    Return base-2 digits of ``n`` padded to length ``padto`` (little-endian).

    Sage returns digits with `n = sum_s d_s * 2^s`, matching the chain
    `x[j,0], ..., x[j,k[j]]`.
    """
    return ZZ(abs(n)).digits(2, padto=padto)


def _r_j(A, j):
    r"""
    Return `r[j] = max(sum(A[k,j] for A[k,j] > 0),
    sum(abs(A[k,j]) for A[k,j] < 0))`.
    """
    col = list(A.column(j))
    positive_sum = sum(a for a in col if a > 0)
    negative_sum = sum(abs(a) for a in col if a < 0)
    return int(max(positive_sum, negative_sum))


def _partition_R(r_values):
    r"""Return the natural partition `R = disjoint union over j of R[j]` with `|R[j]| = r[j]`."""
    R_partition = []
    start = 0
    for rj in r_values:
        R_partition.append(list(range(start, start + rj)))
        start += rj
    return R_partition


def _levels_for_positive_copies(A, j):
    r"""Return the `k^+` level sequence for variable ``j`` (pre-padding)."""
    levels = []
    for k in range(A.nrows()):
        if A[k, j] > 0:
            levels.extend([k] * int(A[k, j]))
    return levels

# If A[k, j] = -2, then level k appears two times.
def _levels_for_negative_copies(A, j):
    r"""Return the `k^-` level sequence for variable ``j`` (pre-padding)."""
    levels = []
    for k in range(A.nrows()):
        if A[k, j] < 0:
            levels.extend([k] * int(-A[k, j]))
    return levels


def _pad_to_length(seq: list, target_len: int, fill_value: int) -> list:
    r"""Return ``seq`` padded to ``target_len`` with ``fill_value``."""
    return list(seq) + [fill_value] * (target_len - len(seq))


def _make_stage1_variables(vars_list, k_values):
    r"""
    Introduce chain variables `x[j,0], ..., x[j,k[j]]`.

    Parenthesized superscripts mark the binary-chain index, e.g.
    `y_chain[0], y_chain[1]`, kept visually distinct from the stage-2 square-bracket
    copy index.
    """
    new_vars_grouped = []
    for j, y_j in enumerate(vars_list):
        name = str(y_j)
        base_latex = latex(y_j)
        group = [
            SR.symbol(f"{name}__{s}",
                      latex_name=rf"{{{base_latex}}}^{{({s})}}")
            for s in range(k_values[j] + 1)
        ]
        new_vars_grouped.append(group)
    return new_vars_grouped


def _make_stage2_copies_and_complements(vars_list, r_values):
    r"""
    Create variable copies and complements for the Theorem 3.2 boxes.

    If `r[j] > 1`, introduce square-bracket copies `y_copy[j,1], ..., y_copy[j,r[j]]`.
    If `r[j] = 1`, reuse the original variable as the single copy.
    """
    copied_vars = []
    complements = []
    copy_identification_rules = {}

    for j, count in enumerate(r_values):
        orig = vars_list[j]
        name = str(orig)
        base_latex = latex(orig)

        if count > 1:
            xs = [SR.symbol(f"{name}__c{s}",
                            latex_name=rf"{{{base_latex}}}^{{[{s}]}}")
                  for s in range(1, count + 1)]
            bxs = [SR.symbol(f"bar_{name}__c{s}",
                             latex_name=rf"\overline{{{base_latex}}}^{{[{s}]}}")
                   for s in range(1, count + 1)]
            for x in xs:
                copy_identification_rules[x] = orig
        else:
            xs = [orig]
            bxs = [SR.symbol(f"bar_{name}",latex_name=rf"\overline{{{base_latex}}}")]

        copied_vars.append(xs)
        complements.append(bxs)

    return copied_vars, complements, copy_identification_rules


#### Lemma 3.1: coefficient reduction ####

def coefficient_reduce_from_matrix(A, b, vars_list=None, original_constraints=None):
    r"""
    Reduce `A*y = b` to a system with `{-1, 0, 1, 2}`-coefficients.

    Lemma 3.1 of [DLO2004]_ converts `P = {y >= 0 : A*y = b}` into
    `Q = {x >= 0 : C*x = d}` with `C in {-1, 0, 1, 2}`. Rational entries
    are cleared row by row, then each coefficient is expanded in binary. The
    new equations use only small coefficients, at the cost of introducing chain
    variables `x[j,0], ..., x[j,k[j]]` linked by `2*x[j,s] - x[j,s+1] == 0`.

    INPUT:

    - ``A`` -- integer or rational matrix
    - ``b`` -- integer or rational column matrix, vector, or list
    - ``vars_list`` -- (optional) symbolic variables for `y`
    - ``original_constraints`` -- (optional) symbolic equations for display

    OUTPUT: a :class:`CoefficientReductionResult`. Call its ``verify()`` method
    to check the reduction invariants.

    EXAMPLES::

        sage: from sage.geometry.polyhedron.transportation_face_embedding \
        ....:     import coefficient_reduce_from_matrix
        sage: A = matrix(QQ, [[3]]); b = vector(QQ, [1])
        sage: red = coefficient_reduce_from_matrix(A, b)
        sage: red.k_values
        [1]
        sage: len(red.new_vars_list)
        2
        sage: red.verify()
        True
    """
    A, b = _normalize_Ab(A, b)
    nrows, ncols = A.nrows(), A.ncols()

    if vars_list is None:
        vars_list = _make_input_variables(ncols)

    # (Lemma 3.1): compute k[j].
    k_values = [_k_j(A, j) for j in range(ncols)]

    # Introduce variables x[j,0], ..., x[j,k[j]].
    new_vars_grouped = _make_stage1_variables(vars_list, k_values)
    new_vars_list = [v for group in new_vars_grouped for v in group]

    # sigma_1(y[j]) = x[j,0].
    sigma_1 = {vars_list[j]: new_vars_grouped[j][0] for j in range(ncols)}

    # Doubling-chain equations 2*x[j,s] - x[j,s+1] == 0.
    new_constraints = []
    n_chain = 0
    for j in range(ncols):
        for s in range(k_values[j]):
            new_constraints.append(
                2 * new_vars_grouped[j][s] - new_vars_grouped[j][s + 1] == 0
            )
            n_chain += 1

    # Rewrite each row using binary expansion of |A[i,j]|.
    for i in range(nrows):
        new_lhs = 0
        for j in range(ncols):
            a = int(A[i, j])
            if a == 0:
                continue
            sign = 1 if a > 0 else -1
            bits = _binary_digits(a, padto=k_values[j] + 1)
            new_lhs += sign * sum(bit * x_js
                                  for bit, x_js
                                  in zip(bits, new_vars_grouped[j]))
        new_constraints.append(new_lhs == b[i, 0])

    result = CoefficientReductionResult(
        new_constraints=new_constraints,
        new_vars_list=new_vars_list,
        new_vars_grouped=new_vars_grouped,
        k_values=k_values,
        sigma_1=sigma_1,
        original_constraints=(list(original_constraints)
                              if original_constraints is not None
                              else None),
        original_vars_list=vars_list,
        A=A,
        b=b,
        n_chain=n_chain,
    )

    return result


def coefficient_reduce(P):
    r"""
    Coefficient-reduce a standard-form Sage ``Polyhedron``.

    Polyhedron-in entry point for :func:`coefficient_reduce_from_matrix`. The
    polyhedron must represent `{y >= 0 : A*y = b}`.

    EXAMPLES::

        sage: from sage.geometry.polyhedron.transportation_face_embedding \
        ....:     import coefficient_reduce
        sage: P = Polyhedron(eqns=[[-1, 3]], ieqs=[[0, 1]])
        sage: red = coefficient_reduce(P)
        sage: red.k_values
        [1]
        sage: red.verify()
        True
    """
    A, b = _polyhedron_to_Ab(P)
    return coefficient_reduce_from_matrix(A, b)


#### Theorem 3.2: plane-sum entry-forbidden 3-way transportation embedding ####

def polytope_3waytransportation_from_matrix(A, b, vars_list=None, U=None,original_constraints=None):
    r"""
    Build the 3-way transportation array and marginals from `A*y = b`.

    Theorem 3.2 of [DLO2004]_ runs the array-construction step directly,
    without coefficient reduction. Each variable `y[j]` and its complement
    `bar_y[j] = U - y[j]` are placed in cyclically arranged enabled entries of
    the box `R[j] x R[j] x H`, and equation `k` is encoded by the
    horizontal plane-sum `w[k] = b[k] + U * sum(abs(A[k,j]) for j with A[k,j] < 0)`.

    INPUT:

    - ``A`` -- integer matrix (typically the Lemma 3.1 output)
    - ``b`` -- integer column matrix, vector, or list
    - ``vars_list`` -- (optional) symbolic variables for `y`
    - ``U`` -- (optional) upper bound on `y[j]`; a positive symbol by default
    - ``original_constraints`` -- (optional) symbolic equations for display

    OUTPUT: a :class:`TransportationEmbeddingResult`. The field ``tensor_M`` is
    a list of ``h`` sparse matrices over ``SR``, one per horizontal plane. Call
    its ``verify()`` method to check the embedding invariants.

    EXAMPLES::

        sage: from sage.geometry.polyhedron.transportation_face_embedding \
        ....:     import polytope_3waytransportation_from_matrix
        sage: A = matrix(QQ, [[1, 1, 0], [1, 0, 1]]); b = vector(QQ, [3, 2])
        sage: emb = polytope_3waytransportation_from_matrix(A, b)
        sage: len(emb.tensor_M)
        3
        sage: emb.tensor_M[0].nrows(), emb.tensor_M[0].ncols()
        (4, 4)
        sage: emb.r_values
        [2, 1, 1]
        sage: emb.verify()
        True
    """
    A, b = _normalize_Ab(A, b)
    nrows, ncols = A.nrows(), A.ncols()

    if vars_list is None:
        vars_list = _make_input_variables(ncols)
    if U is None:
        U = SR.symbol("U")
        assume(U > 0)

    # r[j].
    r_values = [_r_j(A, j) for j in range(ncols)]

    # Nondegenerate-box guard.
    zero_cols = [j for j, rj in enumerate(r_values) if rj == 0]
    if zero_cols:
        offenders = ", ".join(str(vars_list[j]) for j in zero_cols)
        raise ValueError(
            f"variables {{{offenders}}} have zero coefficient in every "
            f"equation, so r_j = 0 and the embedding is ill-defined; "
            f"every variable must appear in at least one equation"
        )

    # Partition R = disjoint union R[j] with |R[j]| = r[j].
    R_partition = _partition_R(r_values)
    r = sum(r_values)
    h = nrows + 1
    slack_level = nrows

    # Copies and complements per variable box.
    copied_vars, complements, copy_identification_rules = (
        _make_stage2_copies_and_complements(vars_list, r_values)
    )
    complement_rules = {
        bx: U - x
        for xs, bxs in zip(copied_vars, complements)
        for x, bx in zip(xs, bxs)
    }

    # k^+ and k^- level sequences, padded by the slack plane.
    k_plus = []
    k_minus = []
    for j in range(ncols):
        k_plus.append(
            _pad_to_length(_levels_for_positive_copies(A, j),
                           r_values[j], slack_level)
        )
        k_minus.append(
            _pad_to_length(_levels_for_negative_copies(A, j),
                           r_values[j], slack_level)
        )

    # Place variable and complement copies in the array M.
    M = _new_tensor_3d((r, r, h))
    active_set_V = set()
    for j in range(ncols):
        R_j = R_partition[j]
        for s in range(r_values[j]):
            i_curr = R_j[s]
            i_next = R_j[(s + 1) % r_values[j]]
            kp = k_plus[j][s]
            km = k_minus[j][s]
            M[kp][i_curr, i_curr] = copied_vars[j][s]
            M[km][i_curr, i_next] = complements[j][s]
            active_set_V.add((i_curr, i_curr, kp))
            active_set_V.add((i_curr, i_next, km))

    # Plane marginals w and vertical marginals u, v.
    marginals_w = [
        b[k, 0] + U * sum(abs(A[k, j])
                          for j in range(ncols) if A[k, j] < 0)
        for k in range(nrows)
    ]
    marginals_w.append(r * U - sum(marginals_w))
    marginals_u = [U] * r
    marginals_v = [U] * r

    # Forbidden set S as the complement of V.
    zero_set_S = set(product(range(r), range(r), range(h))) - active_set_V

    # sigma_2(y[j]) is the first positive-copy coordinate.
    sigma_2 = {
        vars_list[j]: (R_partition[j][0], R_partition[j][0], k_plus[j][0])
        for j in range(ncols)
    }

    result = TransportationEmbeddingResult(
        tensor_M=M,
        zero_set_S=zero_set_S,
        active_set_V=active_set_V,
        marginals_u=marginals_u,
        marginals_v=marginals_v,
        marginals_w=marginals_w,
        sigma_2=sigma_2,
        constraints=(list(original_constraints)
                     if original_constraints is not None else None),
        A=A,
        b=b,
        vars_list=vars_list,
        r_values=r_values,
        R_partition=R_partition,
        k_plus=k_plus,
        k_minus=k_minus,
        slack_level=slack_level,
        copied_vars=copied_vars,
        complements=complements,
        complement_rules=complement_rules,
        copy_identification_rules=copy_identification_rules,
        U=U,
    )

    return result


#### Polytope 3-way transportation embedding ####

def polytope_3waytransportation(P, U=None):
    r"""
    Build a transportation-face embedding from a Sage ``Polyhedron``.

    Polyhedron-in entry point for
    :func:`polytope_3waytransportation_from_matrix`. This skips the binary
    coefficient-reduction preprocessing; use
    :func:`transportation_face_embedding` for the full pipeline.

    EXAMPLES::

        sage: from sage.geometry.polyhedron.transportation_face_embedding \
        ....:     import polytope_3waytransportation
        sage: P = Polyhedron(eqns=[[-3, 1, 1, 0], [-2, 1, 0, 1]],
        ....:                ieqs=[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        sage: emb = polytope_3waytransportation(P)
        sage: len(emb.tensor_M)
        3
        sage: emb.verify()
        True
    """
    A, b = _polyhedron_to_Ab(P)
    return polytope_3waytransportation_from_matrix(A, b, U=U)


#### Theorem 1: coefficient-reduce, embed, then compose ####

def transportation_face_embedding_from_matrix(A, b, vars_list=None, U=None, original_constraints=None):
    r"""
    Embed the standard-form system A*y = b into a face of a 3-way transportation polytope.
    Composes coefficient reduction (Lemma 3.1) and tensor construction (Theorem 3.2)
    from [DLO2004]_. The tensor's plane marginals encode the reduced system via the
    composite coordinate map sigma = sigma_2 after sigma_1.

    INPUT:

    - ``A`` -- integer or rational matrix
    - ``b`` -- integer or rational column matrix, vector, or list
    - ``vars_list`` -- (optional) symbolic variables for `y`
    - ``U`` -- (optional) upper bound on `y[j]`; a positive symbol by default
    - ``original_constraints`` -- (optional) symbolic equations for display

    OUTPUT: a :class:`TransportationFaceEmbeddingResult`. Call its ``verify()``
    method to check both stage invariants and the composition
    `sigma = sigma_2 after sigma_1`.

    EXAMPLES::

        sage: from sage.geometry.polyhedron.transportation_face_embedding \
        ....:     import transportation_face_embedding_from_matrix
        sage: A = matrix(QQ, [[1, 1, 0], [1, 0, 1]]); b = vector(QQ, [3, 2])
        sage: out = transportation_face_embedding_from_matrix(A, b)
        sage: len(out.tensor_M)
        3
        sage: sorted(str(y) for y in out.sigma)
        ['y1', 'y2', 'y3']
        sage: out.verify()
        True
    """
    A, b = _normalize_Ab(A, b)

    if vars_list is None:
        vars_list = _make_input_variables(A.ncols())

    reduction = coefficient_reduce_from_matrix(
        A, b,
        vars_list=vars_list,
        original_constraints=original_constraints,
    )

    embedding = polytope_3waytransportation_from_matrix(
        *_reduced_constraints_to_matrix(reduction),
        vars_list=reduction.new_vars_list,
        U=U,
    )

    sigma = {
        y: embedding.sigma_2[reduction.sigma_1[y]]
        for y in reduction.original_vars_list
    }

    result = TransportationFaceEmbeddingResult(
        tensor_M=embedding.tensor_M,
        zero_set_S=embedding.zero_set_S,
        active_set_V=embedding.active_set_V,
        marginals_u=embedding.marginals_u,
        marginals_v=embedding.marginals_v,
        marginals_w=embedding.marginals_w,
        sigma=sigma,
        reduction=reduction,
        embedding=embedding,
    )

    return result


def transportation_face_embedding(P, U=None):
    r"""
    Build the complete transportation-face embedding from a Sage ``Polyhedron``.

    Recommended entry point for Polyhedron input (Theorem 1 of [DLO2004]_). The
    polyhedron must be bounded, rational, and in standard form
    `{y >= 0 : A*y = b}`.

    EXAMPLES::

        sage: from sage.geometry.polyhedron.transportation_face_embedding \
        ....:     import transportation_face_embedding
        sage: P = Polyhedron(eqns=[[-3, 1, 1, 0], [-2, 1, 0, 1]],
        ....:                ieqs=[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        sage: out = transportation_face_embedding(P)
        sage: len(out.tensor_M)
        3
        sage: out.tensor_M[0].nrows()
        4
        sage: out.verify()
        True
    """
    A, b = _polyhedron_to_Ab(P)
    return transportation_face_embedding_from_matrix(A, b, U=U)


#### Internal: re-extract (A, b) from the reduced symbolic constraints ####

def _reduced_constraints_to_matrix(reduction):
    r"""
    Build `(C, d)` for the system in ``reduction.new_constraints``, using
    ``reduction.new_vars_list`` as the column order. The reduced system has
    integer coefficients in `{-1, 0, 1, 2}`.
    """
    new_constraints = reduction.new_constraints
    new_vars_list = reduction.new_vars_list
    n = len(new_vars_list)

    A_rows = []
    b_entries = []
    for eq in new_constraints:
        expr = (eq.lhs() - eq.rhs()).expand()
        coeffs = [expr.coefficient(v) for v in new_vars_list]
        zero = {v: 0 for v in new_vars_list}
        const = -expr.subs(zero)
        A_rows.append([ZZ(c) for c in coeffs])
        b_entries.append(ZZ(const))

    A = matrix(ZZ, A_rows) if A_rows else matrix(ZZ, 0, n)
    b = matrix(ZZ, len(b_entries), 1, b_entries)
    return A, b
