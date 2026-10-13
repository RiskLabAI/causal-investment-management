from typing import NamedTuple

import networkx as nx
import numpy as np
from scipy.stats import chi2, norm

from RiskLabAI.causal_factor_analysis import (
    CausalDAG,
    causal_role_evidence,
    check_instrument,
    minimal_backdoor_adjustment_sets,
)


class AndersonRubinSet(NamedTuple):
    kind: str
    lo: float
    hi: float
    bounded: bool


class DirectPathTest(NamedTuple):
    coef: float
    se: float
    tstat: float
    fired: bool


class FrontDoorResult(NamedTuple):
    premium: float
    ols_premium: float
    wedge: float
    a_hat: float
    b_hat: float
    sensitivity: dict


class IVPremium(NamedTuple):
    beta: float
    first_stage_f: float
    wald_ci: tuple


class IVWedgeResult(NamedTuple):
    iv: IVPremium
    ar_set: AndersonRubinSet
    strong_instrument: bool
    ols_premium: float
    wedge: float


def _public_dag(graph):
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("graph must be a directed graph")
    return CausalDAG(tuple(graph.nodes), tuple(graph.edges), tuple(graph.nodes))


def classify_causal_role(graph, treatment, outcome, node):
    dag = _public_dag(graph)
    evidence = causal_role_evidence(dag, treatment, outcome, node)
    if evidence.common_cause_paths:
        return "confounder"
    if evidence.mediator_on_directed_paths:
        return "mediator"
    if evidence.collider_on_paths:
        return "collider"
    if check_instrument(dag, node, treatment, outcome).cfi_simple:
        return "instrument"
    descendant_treatment = node in nx.descendants(graph, treatment)
    descendant_outcome = node in nx.descendants(graph, outcome)
    if descendant_treatment and descendant_outcome:
        return "collider"
    if node in nx.ancestors(graph, outcome):
        return "ancestor_of_outcome"
    if descendant_treatment:
        return "descendant_of_treatment"
    if descendant_outcome:
        return "descendant_of_outcome"
    return "independent"


def is_confounder(graph, treatment, outcome, node):
    return classify_causal_role(graph, treatment, outcome, node) == "confounder"


def is_mediator(graph, treatment, outcome, node):
    return classify_causal_role(graph, treatment, outcome, node) == "mediator"


def is_collider(graph, treatment, outcome, node):
    return classify_causal_role(graph, treatment, outcome, node) == "collider"


def is_instrument(graph, treatment, outcome, node):
    return classify_causal_role(graph, treatment, outcome, node) == "instrument"


def _endpoint_matrix_to_digraph(endpoint_matrix, names):
    values = np.asarray(endpoint_matrix, dtype=float)
    node_names = tuple(names)
    if values.shape != (len(node_names), len(node_names)):
        raise ValueError("endpoint matrix and node names must have matching dimensions")
    graph = nx.DiGraph()
    graph.add_nodes_from(node_names)
    ambiguous = []
    bidirected = []
    for left in range(len(node_names)):
        for right in range(left + 1, len(node_names)):
            left_mark = values[left, right]
            right_mark = values[right, left]
            if left_mark == 0 and right_mark == 0:
                continue
            first = node_names[left]
            second = node_names[right]
            if left_mark == -1 and right_mark == 1:
                graph.add_edge(first, second)
            elif left_mark == 1 and right_mark == -1:
                graph.add_edge(second, first)
            elif left_mark == 1 and right_mark == 1:
                ambiguous.append((first, second, "bidirected"))
                bidirected.append((first, second))
            elif left_mark == -1 and right_mark == -1:
                ambiguous.append((first, second, "undirected"))
            else:
                ambiguous.append((first, second, "circle"))
    return graph, ambiguous, bidirected


def backdoor_adjustment_set(graph, treatment, outcome):
    candidates = minimal_backdoor_adjustment_sets(_public_dag(graph), treatment, outcome)
    return set(candidates[0]) if candidates else set()


def flag_controls(graph, treatment, outcome, candidates):
    dag = _public_dag(graph)
    decisions = {}
    for node in candidates:
        evidence = causal_role_evidence(dag, treatment, outcome, node)
        if evidence.common_cause_paths:
            decisions[node] = "valid_confounder"
        elif evidence.mediator_on_directed_paths:
            decisions[node] = "avoid_mediator"
        elif evidence.collider_on_paths:
            decisions[node] = "avoid_collider"
        else:
            decisions[node] = "avoid_collider"
    return decisions


def _vectors(*values, minimum=4):
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("inputs must be one-dimensional")
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("inputs must have equal lengths")
    size = next(iter(lengths))
    if size < minimum:
        raise ValueError("too few observations")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("inputs must be finite")
    return arrays


def _linear_fit(response, *predictors):
    arrays = _vectors(response, *predictors, minimum=len(predictors) + 2)
    outcome = arrays[0]
    design = np.column_stack((np.ones(len(outcome)), *arrays[1:]))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("regression design must have full column rank")
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residual = outcome - design @ coefficients
    degrees = len(outcome) - design.shape[1]
    covariance = (residual @ residual / degrees) * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    return coefficients, standard_errors


def ols_premium(treatment, outcome):
    coefficients, _ = _linear_fit(outcome, treatment)
    return float(coefficients[1])


def _frontdoor_components(treatment, mediator, outcome):
    x, m, y = _vectors(treatment, mediator, outcome)
    first_coefficients, _ = _linear_fit(m, x)
    second_coefficients, second_errors = _linear_fit(y, x, m)
    return float(first_coefficients[1]), float(second_coefficients[2]), float(second_errors[2]), x, m, y


def frontdoor_premium(treatment, mediator, outcome):
    first_slope, second_slope, _, _, _, _ = _frontdoor_components(treatment, mediator, outcome)
    return first_slope * second_slope


def frontdoor_sensitivity_interval(treatment, mediator, outcome, strength):
    if not np.isscalar(strength) or not np.isfinite(strength) or strength < 0:
        raise ValueError("sensitivity strength must be a finite nonnegative scalar")
    first_slope, second_slope, second_error, _, _, _ = _frontdoor_components(treatment, mediator, outcome)
    spread = float(strength) * (abs(first_slope) + second_error)
    endpoints = (first_slope * (second_slope - spread), first_slope * (second_slope + spread))
    return min(endpoints), max(endpoints)


def frontdoor_ols_wedge(treatment, mediator, outcome):
    return abs(frontdoor_premium(treatment, mediator, outcome) - ols_premium(treatment, outcome))


def frontdoor_analysis(treatment, mediator, outcome):
    first_slope, second_slope, _, x, m, y = _frontdoor_components(treatment, mediator, outcome)
    premium = first_slope * second_slope
    ordinary = ols_premium(x, y)
    strengths = (0.0, 0.15, 0.30, 0.45, 0.60)
    sensitivity = {value: frontdoor_sensitivity_interval(x, m, y, value) for value in strengths}
    return FrontDoorResult(premium, ordinary, abs(premium - ordinary), first_slope, second_slope, sensitivity)


def direct_path_test(treatment, mediator, outcome):
    x, m, y = _vectors(treatment, mediator, outcome)
    coefficients, errors = _linear_fit(y, x, m)
    coefficient = float(coefficients[1])
    standard_error = float(errors[1])
    statistic = coefficient / standard_error
    return DirectPathTest(coefficient, standard_error, statistic, abs(statistic) >= 2.0)


def _centered_iv_inputs(treatment, outcome, instrument):
    x, y, z = _vectors(treatment, outcome, instrument)
    return x - x.mean(), y - y.mean(), z - z.mean()


def iv_premium(treatment, outcome, instrument, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    x, y, z = _centered_iv_inputs(treatment, outcome, instrument)
    zz = float(z @ z)
    zx = float(z @ x)
    if zz <= 0 or np.isclose(zx, 0.0):
        raise ValueError("the instrument must vary and predict the treatment")
    beta = float((z @ y) / zx)
    first_slope = zx / zz
    first_residual = x - first_slope * z
    first_variance = float(first_residual @ first_residual) / (len(x) - 2)
    first_stage_f = first_slope * first_slope / (first_variance / zz)
    residual = y - beta * x
    residual_variance = float(residual @ residual) / (len(x) - 1)
    standard_error = np.sqrt(residual_variance * zz / (zx * zx))
    critical = norm.ppf(1.0 - alpha / 2.0)
    wald_ci = (beta - critical * standard_error, beta + critical * standard_error)
    return IVPremium(beta, float(first_stage_f), wald_ci)


def ols_iv_wedge(treatment, outcome, instrument):
    return abs(ols_premium(treatment, outcome) - iv_premium(treatment, outcome, instrument).beta)


def anderson_rubin_set(treatment, outcome, instrument, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    x, y, z = _centered_iv_inputs(treatment, outcome, instrument)
    xx = float(x @ x)
    xy = float(x @ y)
    yy = float(y @ y)
    zx = float(z @ x)
    zy = float(z @ y)
    zz = float(z @ z)
    if zz <= 0:
        raise ValueError("the instrument must vary")
    critical = float(chi2.ppf(1.0 - alpha, 1))
    degrees = len(x) - 1
    first = degrees * zx * zx - critical * zz * xx
    second = -2.0 * degrees * zy * zx + 2.0 * critical * zz * xy
    third = degrees * zy * zy - critical * zz * yy
    scale = max(abs(first), abs(second), abs(third), 1.0)
    tolerance = np.finfo(float).eps * scale * 64.0
    if abs(first) <= tolerance:
        if abs(second) <= tolerance:
            return AndersonRubinSet("allreal", -np.inf, np.inf, False) if third <= tolerance else AndersonRubinSet("empty", np.nan, np.nan, False)
        boundary = -third / second
        if second > 0:
            return AndersonRubinSet("lower", -np.inf, float(boundary), False)
        return AndersonRubinSet("upper", float(boundary), np.inf, False)
    discriminant = second * second - 4.0 * first * third
    if discriminant < -tolerance:
        return AndersonRubinSet("allreal", -np.inf, np.inf, False) if first < 0 else AndersonRubinSet("empty", np.nan, np.nan, False)
    discriminant = max(discriminant, 0.0)
    roots = sorted(((-second - np.sqrt(discriminant)) / (2.0 * first), (-second + np.sqrt(discriminant)) / (2.0 * first)))
    if first > 0:
        return AndersonRubinSet("interval", float(roots[0]), float(roots[1]), True)
    return AndersonRubinSet("disjoint", float(roots[0]), float(roots[1]), False)


def in_anderson_rubin_set(candidate, treatment, outcome, instrument, alpha=0.05):
    if not np.isscalar(candidate) or not np.isfinite(candidate):
        raise ValueError("candidate effect must be a finite scalar")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    x, y, z = _centered_iv_inputs(treatment, outcome, instrument)
    residual = y - float(candidate) * x
    zz = float(z @ z)
    residual_variance = float(residual @ residual) / (len(x) - 1)
    if zz <= 0:
        raise ValueError("the instrument must vary")
    if residual_variance <= 0:
        return bool(np.isclose(z @ residual, 0.0))
    statistic = float(z @ residual) ** 2 / (zz * residual_variance)
    return statistic <= chi2.ppf(1.0 - alpha, 1)


def iv_wedge_analysis(treatment, outcome, instrument):
    instrumental = iv_premium(treatment, outcome, instrument)
    ordinary = ols_premium(treatment, outcome)
    robust_set = anderson_rubin_set(treatment, outcome, instrument)
    return IVWedgeResult(instrumental, robust_set, instrumental.first_stage_f >= 10.0, ordinary, abs(instrumental.beta - ordinary))
