module CausalNotebookTools

using LinearAlgebra, Statistics, Distributions
using RiskLabAI.CausalFactorAnalysis: CausalDAG, causal_role_evidence,
    check_instrument, minimal_backdoor_adjustment_sets

struct DirectedGraph
    nodes::Vector{String}
    edges::Vector{Tuple{String,String}}
    function DirectedGraph(nodes, edges)
        names = String.(collect(nodes))
        links = Tuple{String,String}[(String(first), String(second)) for (first, second) in edges]
        length(unique(names)) == length(names) || throw(ArgumentError("node names must be unique"))
        all(edge -> all(in(names), edge), links) || throw(ArgumentError("unknown graph endpoint"))
        new(names, unique(links))
    end
end

DirectedGraph(edges) = DirectedGraph(unique(vcat([[first, second] for (first, second) in edges]...)), edges)
_public_dag(graph::DirectedGraph) = CausalDAG(graph.nodes, graph.edges, graph.nodes)
_public_dag(graph::CausalDAG) = CausalDAG(graph.nodes, graph.directed_edges, graph.nodes)

function _descendants(dag, start; reverse=false)
    visited, pending = Set{String}(), [String(start)]
    while !isempty(pending)
        current = pop!(pending)
        for (source, destination) in dag.directed_edges
            left, right = reverse ? (destination, source) : (source, destination)
            if left == current && !(right in visited)
                push!(visited, right)
                push!(pending, right)
            end
        end
    end
    return visited
end

function classify_causal_role(graph, treatment, outcome, node)
    dag = _public_dag(graph)
    evidence = causal_role_evidence(dag, treatment, outcome, node)
    !isempty(evidence.common_cause_paths) && return "confounder"
    !isempty(evidence.mediator_on_directed_paths) && return "mediator"
    !isempty(evidence.collider_on_paths) && return "collider"
    check_instrument(dag, node, treatment, outcome).cfi_simple && return "instrument"
    descendant_treatment = node in _descendants(dag, treatment)
    descendant_outcome = node in _descendants(dag, outcome)
    descendant_treatment && descendant_outcome && return "collider"
    node in _descendants(dag, outcome; reverse=true) && return "ancestor_of_outcome"
    descendant_treatment && return "descendant_of_treatment"
    descendant_outcome && return "descendant_of_outcome"
    return "independent"
end

is_confounder(graph, treatment, outcome, node) = classify_causal_role(graph, treatment, outcome, node) == "confounder"
is_mediator(graph, treatment, outcome, node) = classify_causal_role(graph, treatment, outcome, node) == "mediator"
is_collider(graph, treatment, outcome, node) = classify_causal_role(graph, treatment, outcome, node) == "collider"
is_instrument(graph, treatment, outcome, node) = classify_causal_role(graph, treatment, outcome, node) == "instrument"

function _endpoint_matrix_to_digraph(values::AbstractMatrix, names)
    count = length(names)
    size(values) == (count, count) || throw(DimensionMismatch("endpoint matrix and node names must have matching dimensions"))
    edges, ambiguous, bidirected = Tuple{String,String}[], Tuple{String,String,String}[], Tuple{String,String}[]
    for left in 1:count, right in left+1:count
        left_mark, right_mark = values[left, right], values[right, left]
        first, second = String(names[left]), String(names[right])
        if left_mark == 0 && right_mark == 0
            continue
        elseif left_mark == -1 && right_mark == 1
            push!(edges, (first, second))
        elseif left_mark == 1 && right_mark == -1
            push!(edges, (second, first))
        elseif left_mark == 1 && right_mark == 1
            push!(ambiguous, (first, second, "bidirected"))
            push!(bidirected, (first, second))
        elseif left_mark == -1 && right_mark == -1
            push!(ambiguous, (first, second, "undirected"))
        else
            push!(ambiguous, (first, second, "circle"))
        end
    end
    return DirectedGraph(names, edges), ambiguous, bidirected
end

function backdoor_adjustment_set(graph, treatment, outcome)
    candidates = minimal_backdoor_adjustment_sets(_public_dag(graph), treatment, outcome)
    return isempty(candidates) ? Set{String}() : Set(first(candidates))
end

function flag_controls(graph, treatment, outcome, candidates)
    dag = _public_dag(graph)
    decisions = Dict{String,String}()
    for node in candidates
        evidence = causal_role_evidence(dag, treatment, outcome, node)
        decisions[node] = !isempty(evidence.common_cause_paths) ? "valid_confounder" : !isempty(evidence.mediator_on_directed_paths) ? "avoid_mediator" : "avoid_collider"
    end
    return decisions
end

struct AndersonRubinSet
    kind::String
    lo::Float64
    hi::Float64
    bounded::Bool
end

struct DirectPathTest
    coef::Float64
    se::Float64
    tstat::Float64
    fired::Bool
end

struct FrontDoorResult
    premium::Float64
    ols_premium::Float64
    wedge::Float64
    a_hat::Float64
    b_hat::Float64
    sensitivity::Dict{Float64,Tuple{Float64,Float64}}
end

struct IVPremium
    beta::Float64
    first_stage_f::Float64
    wald_ci::Tuple{Float64,Float64}
end

struct IVWedgeResult
    iv::IVPremium
    ar_set::AndersonRubinSet
    strong_instrument::Bool
    ols_premium::Float64
    wedge::Float64
end

function _vectors(values...; minimum=4)
    all(value -> value isa AbstractVector, values) || throw(ArgumentError("inputs must be one-dimensional"))
    length(unique(length.(values))) == 1 || throw(DimensionMismatch("inputs must have equal lengths"))
    length(first(values)) >= minimum || throw(ArgumentError("too few observations"))
    all(value -> all(isfinite, value), values) || throw(ArgumentError("inputs must be finite"))
    return map(value -> Float64.(value), values)
end

function _linear_fit(response, predictors...)
    arrays = _vectors(response, predictors...; minimum=length(predictors)+2)
    outcome = first(arrays)
    design = hcat(ones(length(outcome)), arrays[2:end]...)
    rank(design) == size(design, 2) || throw(ArgumentError("regression design must have full column rank"))
    coefficients = design\outcome
    residual = outcome-design*coefficients
    degrees = length(outcome)-size(design, 2)
    covariance = sum(abs2, residual)/degrees*inv(design'*design)
    return coefficients, sqrt.(diag(covariance))
end

ols_premium(treatment, outcome) = first(_linear_fit(outcome, treatment))[2]

function _frontdoor_components(treatment, mediator, outcome)
    x, m, y = _vectors(treatment, mediator, outcome)
    first_coefficients, _ = _linear_fit(m, x)
    second_coefficients, second_errors = _linear_fit(y, x, m)
    return first_coefficients[2], second_coefficients[3], second_errors[3], x, m, y
end

function frontdoor_premium(treatment, mediator, outcome)
    first_slope, second_slope = _frontdoor_components(treatment, mediator, outcome)
    return first_slope*second_slope
end

function frontdoor_sensitivity_interval(treatment, mediator, outcome, strength)
    strength isa Real && isfinite(strength) && strength >= 0 || throw(ArgumentError("sensitivity strength must be a finite nonnegative scalar"))
    first_slope, second_slope, second_error = _frontdoor_components(treatment, mediator, outcome)
    spread = strength*(abs(first_slope)+second_error)
    return minmax(first_slope*(second_slope-spread), first_slope*(second_slope+spread))
end

frontdoor_ols_wedge(treatment, mediator, outcome) = abs(frontdoor_premium(treatment, mediator, outcome)-ols_premium(treatment, outcome))

function frontdoor_analysis(treatment, mediator, outcome)
    first_slope, second_slope, _, x, m, y = _frontdoor_components(treatment, mediator, outcome)
    premium, ordinary = first_slope*second_slope, ols_premium(x, y)
    sensitivity = Dict(value => frontdoor_sensitivity_interval(x, m, y, value) for value in (0., .15, .30, .45, .60))
    return FrontDoorResult(premium, ordinary, abs(premium-ordinary), first_slope, second_slope, sensitivity)
end

function direct_path_test(treatment, mediator, outcome)
    x, m, y = _vectors(treatment, mediator, outcome)
    coefficients, errors = _linear_fit(y, x, m)
    statistic = coefficients[2]/errors[2]
    return DirectPathTest(coefficients[2], errors[2], statistic, abs(statistic) >= 2.)
end

_centered_iv_inputs(treatment, outcome, instrument) = map(value -> value.-mean(value), _vectors(treatment, outcome, instrument))

function iv_premium(treatment, outcome, instrument, alpha=.05)
    0 < alpha < 1 || throw(ArgumentError("alpha must lie strictly between zero and one"))
    x, y, z = _centered_iv_inputs(treatment, outcome, instrument)
    zz, zx = dot(z, z), dot(z, x)
    zz > 0 && abs(zx) > 1e-8 || throw(ArgumentError("the instrument must vary and predict the treatment"))
    beta, first_slope = dot(z, y)/zx, zx/zz
    first_residual = x-first_slope*z
    first_variance = sum(abs2, first_residual)/(length(x)-2)
    first_stage_f = first_slope^2/(first_variance/zz)
    residual = y-beta*x
    residual_variance = sum(abs2, residual)/(length(x)-1)
    standard_error = sqrt(residual_variance*zz/zx^2)
    critical = quantile(Normal(), 1-alpha/2)
    return IVPremium(beta, first_stage_f, (beta-critical*standard_error, beta+critical*standard_error))
end

ols_iv_wedge(treatment, outcome, instrument) = abs(ols_premium(treatment, outcome)-iv_premium(treatment, outcome, instrument).beta)

function anderson_rubin_set(treatment, outcome, instrument, alpha=.05)
    0 < alpha < 1 || throw(ArgumentError("alpha must lie strictly between zero and one"))
    x, y, z = _centered_iv_inputs(treatment, outcome, instrument)
    xx, xy, yy, zx, zy, zz = dot(x, x), dot(x, y), dot(y, y), dot(z, x), dot(z, y), dot(z, z)
    zz > 0 || throw(ArgumentError("the instrument must vary"))
    critical, degrees = quantile(Chisq(1), 1-alpha), length(x)-1
    first = degrees*zx^2-critical*zz*xx
    second = -2*degrees*zy*zx+2*critical*zz*xy
    third = degrees*zy^2-critical*zz*yy
    tolerance = eps(Float64)*max(abs(first), abs(second), abs(third), 1.)*64
    if abs(first) <= tolerance
        if abs(second) <= tolerance
            return third <= tolerance ? AndersonRubinSet("allreal", -Inf, Inf, false) : AndersonRubinSet("empty", NaN, NaN, false)
        end
        boundary = -third/second
        return second > 0 ? AndersonRubinSet("lower", -Inf, boundary, false) : AndersonRubinSet("upper", boundary, Inf, false)
    end
    discriminant = second^2-4*first*third
    if discriminant < -tolerance
        return first < 0 ? AndersonRubinSet("allreal", -Inf, Inf, false) : AndersonRubinSet("empty", NaN, NaN, false)
    end
    root = sqrt(max(discriminant, 0.))
    lower, upper = minmax((-second-root)/(2*first), (-second+root)/(2*first))
    return AndersonRubinSet(first > 0 ? "interval" : "disjoint", lower, upper, first > 0)
end

function in_anderson_rubin_set(candidate, treatment, outcome, instrument, alpha=.05)
    candidate isa Real && isfinite(candidate) || throw(ArgumentError("candidate effect must be a finite scalar"))
    0 < alpha < 1 || throw(ArgumentError("alpha must lie strictly between zero and one"))
    x, y, z = _centered_iv_inputs(treatment, outcome, instrument)
    residual = y-candidate*x
    zz = dot(z, z)
    residual_variance = sum(abs2, residual)/(length(x)-1)
    zz > 0 || throw(ArgumentError("the instrument must vary"))
    residual_variance <= 0 && return abs(dot(z, residual)) <= 1e-8
    return dot(z, residual)^2/(zz*residual_variance) <= quantile(Chisq(1), 1-alpha)
end

function iv_wedge_analysis(treatment, outcome, instrument)
    instrumental, ordinary = iv_premium(treatment, outcome, instrument), ols_premium(treatment, outcome)
    robust_set = anderson_rubin_set(treatment, outcome, instrument)
    return IVWedgeResult(instrumental, robust_set, instrumental.first_stage_f >= 10., ordinary, abs(instrumental.beta-ordinary))
end

export DirectedGraph, CausalDAG, classify_causal_role, is_confounder, is_mediator,
    is_collider, is_instrument, backdoor_adjustment_set, flag_controls,
    AndersonRubinSet, DirectPathTest, FrontDoorResult, IVPremium, IVWedgeResult,
    ols_premium, frontdoor_premium, frontdoor_sensitivity_interval,
    frontdoor_ols_wedge, frontdoor_analysis, direct_path_test, iv_premium,
    ols_iv_wedge, anderson_rubin_set, in_anderson_rubin_set, iv_wedge_analysis

end
