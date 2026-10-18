module BookMirageStudy

using Random, LinearAlgebra, Statistics, DataFrames
using RiskLabAI.CausalFactorAnalysis
using ..BookIdentificationStudy: normal_block

mutable struct SharedNormalRNG <: AbstractRNG
    values::Matrix{Float64}
    column::Int
end

SharedNormalRNG(values::AbstractMatrix) = SharedNormalRNG(Matrix{Float64}(values),1)

function Random.randn(rng::SharedNormalRNG,count::Int)
    size(rng.values,1)==count || throw(DimensionMismatch("Shared draw length differs"))
    rng.column<=size(rng.values,2) || throw(ArgumentError("Shared draws exhausted"))
    values=copy(rng.values[:,rng.column])
    rng.column+=1
    return values
end

function primitive_sample(store,count=5000,seed=0)
    group="n_"*string(count)
    index=findfirst(==(seed),store.groups[group]["seeds"])
    isnothing(index) && throw(ArgumentError("Seed is absent from the preserved inputs"))
    return normal_block(store,group,index)
end

function generated_values(structure,values)
    if structure=="Fork"
        z=values[:,1]
        return DataFrame("X"=>z+values[:,2],"Y"=>z+values[:,3],"Z"=>z)
    elseif structure=="Collider"
        x,y=values[:,1],values[:,2]
        return DataFrame("X"=>x,"Y"=>y,"Z"=>x+y+values[:,3])
    elseif structure=="Confounded mediator"
        x,w=values[:,1],values[:,2]
        z=x+w+values[:,3]
        return DataFrame("X"=>x,"Y"=>z+w+values[:,4],"Z"=>z,"W"=>w)
    end
    throw(ArgumentError("Unknown structural system"))
end

function shared_experiment(function_value,store,count=5000,seed=0)
    rng=SharedNormalRNG(primitive_sample(store,count,seed))
    result=function_value(count,seed;rng)
    expected=function_value===confounded_mediator_specification_experiment ? 5 : 4
    rng.column==expected || error("The public experiment consumed unexpected draws")
    return result
end

function manual_diagnostics(outcome,predictors,names)
    design=hcat(ones(length(outcome)),predictors...)
    coefficients=design\outcome
    residuals=outcome-design*coefficients
    degrees=length(outcome)-size(design,2)
    residual_variance=dot(residuals,residuals)/degrees
    standard_errors=sqrt.(diag(inv(design'*design))*residual_variance)
    centered=outcome.-mean(outcome)
    r_squared=1-dot(residuals,residuals)/dot(centered,centered)
    return RegressionDiagnostics(Tuple(["intercept";names]),Tuple(coefficients),Tuple(standard_errors),Tuple(coefficients./standard_errors),r_squared,1-(1-r_squared)*(length(outcome)-1)/degrees,residual_variance,length(outcome))
end

coefficient_value(result,name) = result.coefficients[findfirst(==(name),result.coefficient_names)]

function same_record(left,right)
    typeof(left)===typeof(right) || return false
    if isstructtype(typeof(left)) && !(left isa Number) && !(left isa AbstractString) && !(left isa Tuple)
        return all(name->same_record(getproperty(left,name),getproperty(right,name)),fieldnames(typeof(left)))
    end
    return isequal(left,right)
end

export SharedNormalRNG,primitive_sample,generated_values,shared_experiment,
    manual_diagnostics,coefficient_value,same_record

end
