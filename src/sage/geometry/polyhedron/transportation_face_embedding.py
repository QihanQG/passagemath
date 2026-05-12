r"""Transportation-face embedding utilities for standard-form rational polytopes.

This module builds symbolic data that embeds a rational polytope
``P = {y >= 0 : A*y = b}`` into a face of a 3-way transportation polytope (De Loera--Onn 2004,2006)
The output is useful for inspecting the constructed tensor, its marginals,
and the coordinates where the original variables appear.

Most users should call one of these two functions:

- ``transportation_face_embedding_from_matrix(A, b)`` if they already have a
  matrix ``A`` and right-hand side ``b``.
- ``transportation_face_embedding(P)`` if they already have a Sage
  ``Polyhedron`` in standard form.

Example::

    from sage.all__sagemath_symbolics import *
    from transportation_face_embedding import transportation_face_embedding_from_matrix

    A = matrix(QQ, [[1, 1, 0], [1, 0, 1]])
    b = vector(QQ, [3, 2])

    out = transportation_face_embedding_from_matrix(A, b)
    M = out.tensor_M          # symbolic 3D tensor
    u = out.marginals_u       # first-index plane marginals
    v = out.marginals_v       # second-index plane marginals
    w = out.marginals_w       # third-index plane marginals
    sigma = out.sigma         # original variables -> tensor coordinates

The returned result object keeps the final tensor, forced zero entries,
enabled entries, marginals, and coordinate map at the top level. Intermediate
data from coefficient reduction and tensor construction are available under
``out.reduction`` and ``out.embedding``.

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

class Tensor3D(SageObject):
    r"""Small 3D symbolic tensor used for display and slicing.

    Entries are stored as ``self.data[i][j][k]`` and can be accessed with
    ``T[i, j, k]``. The methods ``slice_i``, ``slice_j``, and ``slice_k`` are
    convenience helpers for inspecting fixed-index slices of the tensor.
    """

    def __init__(self, shape):
        self.shape = shape
        r, c, h = shape
        self.data = [[[0] * h for _ in range(c)] for _ in range(r)]

    def __getitem__(self, key):
        return self.data[key[0]][key[1]][key[2]]

    def __setitem__(self, key, value):
        self.data[key[0]][key[1]][key[2]] = value

    def slice_k(self, k):
        return [[self.data[i][j][k] for j in range(self.shape[1])]
                for i in range(self.shape[0])]

    def slice_i(self, i):
        return self.data[i]

    def slice_j(self, j):
        return [self.data[i][j] for i in range(self.shape[0])]

    def _repr_(self):
        return f"3D tensor of shape {self.shape}"

    def _latex_(self):
        def bracket(parts):
            return r"\left[" + ", ".join(parts) + r"\right]"

        def rows(plane):
            return bracket(bracket(latex(x) for x in row) for row in plane)

        return bracket(rows(plane) for plane in self.data)


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
        "recovery_rules",
        "U",
    )


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

def _b_to_column(b):
    if hasattr(b, "nrows") and hasattr(b, "ncols"):
        if b.ncols() == 1:
            return [b[i, 0] for i in range(b.nrows())]
        if b.nrows() == 1:
            return [b[0, j] for j in range(b.ncols())]
        raise ValueError("b must be a vector, list, row matrix, or column matrix")
    return list(b)

def _normalize_Ab(A, b):
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
            "standard form {y >= 0 : A y = b} with a "
            "non-trivial equation system."
        )

    for ieq in P.inequalities_list():
        c = ieq[0]
        coeffs = ieq[1:]
        nonzero = [(i, v) for i, v in enumerate(coeffs) if v != 0]
        if c != 0 or len(nonzero) != 1 or nonzero[0][1] < 0:
            raise ValueError(
                f"Polyhedron has inequality {ieq}, which is not a coordinate "
                f"non-negativity y_i >= 0. Only standard-form polytopes "
                f"{{y >= 0 : A y = b}} are supported; upper bounds and general "
                f"inequalities must be rewritten as equations using slack "
                f"variables before calling this function."
            )

    A_rows = [eq[1:] for eq in eqns]
    b_entries = [-eq[0] for eq in eqns]
    return _normalize_Ab(matrix(QQ, A_rows), b_entries)

def _make_input_variables(n, prefix="y"):
    return [
        SR.symbol(f"{prefix}{j + 1}",
                  latex_name=rf"{prefix}_{{{j + 1}}}")
        for j in range(n)
    ]

# return `k_j = floor(log_2 max_i |a_{i,j}|)
# If column j is zero, return 0 so that the variable still gets one copy.
def _k_j(A, j):
    m = max((abs(int(A[i, j])) for i in range(A.nrows())), default=0)
    return ZZ(m).nbits() - 1 if m > 0 else 0


#Sage returns digits in little-endian order: n = sum_s digits[s] * 2^s.
#Which is what we needed for  x_{j,0}, ..., x_{j,k_j}.
def _binary_digits(n, padto):
    return ZZ(abs(n)).digits(2, padto=padto)

def _r_j(A, j):
    col = list(A.column(j))
    positive_sum = sum(a for a in col if a > 0)
    negative_sum = sum(abs(a) for a in col if a < 0)
    return int(max(positive_sum, negative_sum))

# R_j = [start, ..., start + r_j - 1].
def _partition_R(r_values):
    R_partition = []
    start = 0
    for rj in r_values:
        R_partition.append(list(range(start, start + rj)))
        start += rj
    return R_partition

#If A[k, j] = 3, then level k appears three times.
def _levels_for_positive_copies(A, j):
    levels = []
    for k in range(A.nrows()):
        if A[k, j] > 0:
            levels.extend([k] * int(A[k, j]))
    return levels

#If A[k, j] = -2, then level k appears two times.
def _levels_for_negative_copies(A, j):
    levels = []
    for k in range(A.nrows()):
        if A[k, j] < 0:
            levels.extend([k] * int(-A[k, j]))
    return levels

def _pad_to_length(seq, target_len, fill_value):
    return list(seq) + [fill_value] * (target_len - len(seq))

#    Parenthesized superscripts mark the binary-chain index, e.g.
#    y^{(0)}, y^{(1)}. This stays visually distinct from the stage-2
#   square-bracket copy index.
def _make_stage1_variables(vars_list, k_values):
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

#    If r_j > 1, introduce square-bracket copies y_j^{[1]}, ..., y_j^{[r_j]}.
#    If r_j = 1, reuse the original variable itself as the single copy.
def _make_stage2_copies_and_complements(vars_list, r_values):
    copied_vars = []
    complements = []
    recovery_rules = {}

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
                recovery_rules[x] = orig
        else:
            xs = [orig]
            bxs = [SR.symbol(f"bar_{name}", latex_name=rf"\overline{{{base_latex}}}")]

        copied_vars.append(xs)
        complements.append(bxs)

    return copied_vars, complements, recovery_rules

def _sum2d(mat):
    return sum(sum(row) for row in mat)

def _verify_reduction(result):
    A = result.A
    b = result.b
    vars_list = result.original_vars_list
    k_values = result.k_values
    new_vars_grouped = result.new_vars_grouped
    new_constraints = result.new_constraints
    n_chain = result.n_chain

    sub_back = {
        new_vars_grouped[j][s]: 2 ** s * vars_list[j]
        for j in range(A.ncols())
        for s in range(k_values[j] + 1)
    }

    for c in range(n_chain):
        eq = new_constraints[c]
        diff = (eq.lhs() - eq.rhs()).subs(sub_back).expand()
        assert diff.is_trivial_zero(), f"Doubling chain check failed: {eq}"

    for i in range(A.nrows()):
        eq = new_constraints[n_chain + i]
        recovered = eq.lhs().subs(sub_back).expand()
        expected = sum(int(A[i, j]) * vars_list[j] for j in range(A.ncols()))
        assert (recovered - expected).expand().is_trivial_zero(), (
            f"Main equation row {i} failed: {recovered} != {expected}"
        )
        assert eq.rhs() == b[i, 0], f"Main equation rhs mismatch at row {i}"

    return True

def _verify_embedding(result):
    M = result.tensor_M
    A = result.A
    b = result.b
    vars_list = result.vars_list
    U = result.U
    complement_rules = result.complement_rules
    recovery_rules = result.recovery_rules

    for k in range(A.nrows()):
        expected = sum(A[k, j] * vars_list[j] for j in range(A.ncols())) - b[k, 0]
        recovered = (_sum2d(M.slice_k(k))
                     .subs(complement_rules)
                     .subs(recovery_rules))
        neg_sum = sum(abs(A[k, j]) for j in range(A.ncols()) if A[k, j] < 0)
        actual = (recovered - U * neg_sum - b[k, 0]).expand()
        assert (actual - expected).expand().is_trivial_zero(), (
            f"Plane {k} mismatch"
        )

    for y, coord in result.sigma_2.items():
        assert coord in result.active_set_V, (
            f"sigma_2[{y}] does not land in V"
        )

    return True

def _verify_transportation_face_embedding(result):
    reduction = result.reduction
    embedding = result.embedding

    _verify_reduction(reduction)
    _verify_embedding(embedding)

    for key in ("tensor_M", "zero_set_S", "active_set_V",
                "marginals_u", "marginals_v", "marginals_w"):
        assert getattr(result, key) == getattr(embedding, key), (
            f"Transportation-face embedding wrapper mismatch at {key}"
        )

    for y in reduction.original_vars_list:
        x_j0 = reduction.sigma_1[y]
        expected_coord = embedding.sigma_2[x_j0]
        assert result.sigma[y] == expected_coord, (
            f"sigma composition failed for {y}"
        )

    return True

def coefficient_reduce_from_matrix(A, b, vars_list=None, original_constraints=None, verify=True):
    r""" Lemma 3.1 converts ``P = {y >= 0 : A*y = b}`` into
    ``Q = {x >= 0 : C*x = d}``, where C in {-1, 0, 1, 2}.

    Use this when you only want the preprocessing step. Rational entries are
    cleared row-by-row, then each coefficient is expanded in binary. The new
    equations use only small coefficients from ``{-1, 0, 1, 2}``, at the cost
    of introducing extra chain variables.

    The result contains ``new_constraints``, ``new_vars_list``,
    ``new_vars_grouped``, ``k_values``, and ``sigma_1``. Set ``verify=False``
    to skip the internal symbolic consistency checks.
    """
    A, b = _normalize_Ab(A, b)
    nrows, ncols = A.nrows(), A.ncols()

    #introduce variables x_{j,0}, ..., x_{j,k_j}.
    if vars_list is None:
        vars_list = _make_input_variables(ncols)

    k_values = [_k_j(A, j) for j in range(ncols)]

    new_vars_grouped = _make_stage1_variables(vars_list, k_values)
    new_vars_list = [v for group in new_vars_grouped for v in group]

    #sigma_1(y_j) = x_{j,0}.
    sigma_1 = {vars_list[j]: new_vars_grouped[j][0] for j in range(ncols)}

    new_constraints = []
    n_chain = 0
    # add doubling-chain equations 2*x_{j,s} - x_{j,s+1} = 0.
    for j in range(ncols):
        for s in range(k_values[j]):
            new_constraints.append(
                2 * new_vars_grouped[j][s] - new_vars_grouped[j][s + 1] == 0
            )
            n_chain += 1
    #rewrite each row using binary expansion of |a_{i,j}|.
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

    if verify:
        _verify_reduction(result)
    return result

def coefficient_reduce(P, verify=True):
    r"""Coefficient-reduce a standard-form Sage ``Polyhedron``.

    The polyhedron must represent ``{y >= 0 : A*y = b}``: equality constraints
    are used as ``A*y = b``, and inequalities must be coordinate
    non-negativity constraints. For matrix input, use
    ``coefficient_reduce_from_matrix`` instead.
    """
    A, b = _polyhedron_to_Ab(P)
    return coefficient_reduce_from_matrix(A, b, verify=verify)

def polytope_3waytransportation_from_matrix(A, b, vars_list=None, U=None,  original_constraints=None, verify=True):
    r"""Build the 3-way transportation tensor and marginals from ``A*y = b``.

    This runs the tensor-construction step directly, without first doing
    coefficient reduction. Use it when ``A`` already has manageable
    coefficients, or when you want to inspect only the transportation-face
    construction.

    The result contains ``tensor_M``, ``marginals_u``, ``marginals_v``,
    ``marginals_w``, ``zero_set_S``, ``active_set_V``, and ``sigma_2``. If
    ``U`` is omitted, a symbolic positive variable ``U`` is created.
    """
    A, b = _normalize_Ab(A, b)
    nrows, ncols = A.nrows(), A.ncols()

    if vars_list is None:
        vars_list = _make_input_variables(ncols)
    if U is None:
        U = SR.symbol("U")
        assume(U > 0)

    r_values = [_r_j(A, j) for j in range(ncols)] #(Theorem 3.2): compute r_j.

    zero_cols = [j for j, rj in enumerate(r_values) if rj == 0]
    if zero_cols:
        offenders = ", ".join(str(vars_list[j]) for j in zero_cols)
        raise ValueError(
            f"variables {{{offenders}}} have zero coefficient in every "
            f"equation, so r_j = 0 and the embedding is ill-defined; "
            f"every variable must appear in at least one equation"
        )

    R_partition = _partition_R(r_values) #partition R = disjoint union R_j, with |R_j| = r_j.
    r = sum(r_values) 
    h = nrows + 1
    slack_level = nrows

    # Introduce copies and complements for each variable box.
    copied_vars, complements, recovery_rules = (
        _make_stage2_copies_and_complements(vars_list, r_values)
    )
    complement_rules = {
        bx: U - x
        for xs, bxs in zip(copied_vars, complements)
        for x, bx in zip(xs, bxs)
    }

    k_plus = []
    k_minus = []
    for j in range(ncols):
        k_plus.append( _pad_to_length(_levels_for_positive_copies(A, j), r_values[j], slack_level) )
        k_minus.append(_pad_to_length(_levels_for_negative_copies(A, j),r_values[j], slack_level))

    M = Tensor3D((r, r, h)) # place variable copies and complement copies in tensor M.
    active_set_V = set()
    for j in range(ncols):
        R_j = R_partition[j]
        for s in range(r_values[j]):
            i_curr = R_j[s]
            i_next = R_j[(s + 1) % r_values[j]]
            pos_coord = (i_curr, i_curr, k_plus[j][s])
            neg_coord = (i_curr, i_next, k_minus[j][s])
            M[pos_coord] = copied_vars[j][s]
            M[neg_coord] = complements[j][s]
            active_set_V.add(pos_coord)
            active_set_V.add(neg_coord)

    marginals_w = [
        b[k, 0] + U * sum(abs(A[k, j])
                          for j in range(ncols) if A[k, j] < 0)
        for k in range(nrows)
    ]
    marginals_w.append(r * U - sum(marginals_w))
    marginals_u = [U] * r
    marginals_v = [U] * r

    zero_set_S = set(product(range(r), range(r), range(h))) - active_set_V
    # Let sigma_2(y_j) be the first positive copy coordinate.
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
        recovery_rules=recovery_rules,
        U=U,
    )

    if verify:
        _verify_embedding(result)
    return result

def polytope_3waytransportation(P, U=None, verify=True):
    r"""Build a transportation-face embedding from a Sage ``Polyhedron``.

    This is the Polyhedron-input version of
    ``polytope_3waytransportation_from_matrix``. It skips the binary
    coefficient-reduction preprocessing, so use ``transportation_face_embedding``
    when you want the full matrix-normalization and tensor-construction workflow.
    """
    A, b = _polyhedron_to_Ab(P)
    return polytope_3waytransportation_from_matrix(A, b, U=U, verify=verify)

def transportation_face_embedding_from_matrix(A, b, vars_list=None, U=None,  original_constraints=None, verify=True):
    r"""Build the complete transportation-face embedding from matrix input.

    This is the recommended entry point when your input is ``A, b``. It first
    rewrites the equation system with small coefficients, then constructs a
    3-way symbolic tensor whose plane marginals encode the reduced system.
    The forbidden entries are listed in ``zero_set_S``; enabled entries are
    listed in ``active_set_V``.

    Example::

        A = matrix(QQ, [[1, 1, 0], [1, 0, 1]])
        b = vector(QQ, [3, 2])
        out = transportation_face_embedding_from_matrix(A, b)
        out.sigma          # original variables -> tensor coordinates
        out.tensor_M       # symbolic 3D tensor
        out.marginals_w    # third-index plane marginals

    Set ``verify=False`` for faster runs after the code has already been tested
    on your examples.
    """
    A, b = _normalize_Ab(A, b)

    if vars_list is None:
        vars_list = _make_input_variables(A.ncols())

    reduction = coefficient_reduce_from_matrix(
        A, b,
        vars_list=vars_list,
        original_constraints=original_constraints,
        verify=False,
    )

    embedding = polytope_3waytransportation_from_matrix(
        *_reduced_constraints_to_matrix(reduction),
        vars_list=reduction.new_vars_list,
        U=U,
        verify=False,
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

    if verify:
        _verify_transportation_face_embedding(result)
    return result

def transportation_face_embedding(P, U=None, verify=True):
    r"""Build the complete transportation-face embedding from a Sage ``Polyhedron``.

    This is the recommended entry point for Polyhedron input. The polyhedron
    must be bounded, rational, and in standard form ``{y >= 0 : A*y = b}``.

    For direct matrix input, use ``transportation_face_embedding_from_matrix``.
    """
    A, b = _polyhedron_to_Ab(P)
    return transportation_face_embedding_from_matrix(A, b, U=U, verify=verify)

def _reduced_constraints_to_matrix(reduction):
    r"""Convert reduced symbolic equations back into a Sage matrix system."""
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
