module BookTreatmentStudy

using DataFrames, Statistics, LinearAlgebra
import RiskLabAI
using RiskLabAI.CausalFactorAnalysis
using ..BookIdentificationStudy: normal_block

function generated_did(store,group,index,effect;differential_trend=0.,anticipation=0.)
    values = normal_block(store,group,index)
    assignment = repeat([0,1];inner=size(values,1)÷2)
    ability,before_shock,after_shock = values[:,1],.8 .*values[:,2],.8 .*values[:,3]
    untreated_before = 2.0 .+ .6 .*assignment+.5 .*ability+before_shock
    untreated_after = 2.4 .+ .6 .*assignment+differential_trend.*assignment+.5 .*ability+after_shock
    before = untreated_before+anticipation.*assignment
    after = untreated_after+effect.*assignment
    result = difference_in_differences(mean(before[assignment.==1]),mean(after[assignment.==1]),mean(before[assignment.==0]),mean(after[assignment.==0]))
    return result,DataFrame("Group"=>assignment,"Before"=>before,"After"=>after)
end

function generated_iv(store,group,index,relevance;exclusion_path=0.)
    values = normal_block(store,group,index)
    instrument,common_cause = values[:,1],values[:,2]
    treatment = relevance.*instrument+.9 .*common_cause+values[:,3]
    outcome = .8 .*treatment+1.3 .*common_cause+exclusion_path.*instrument+values[:,4]
    numerator,denominator = cov(instrument,outcome),cov(instrument,treatment)
    return linear_instrumental_variable_effect(numerator,denominator),numerator,denominator
end

function histogram_peak(values,edges;density=false)
    counts = zeros(length(edges)-1)
    for value in values
        first(edges)<=value<=last(edges) || continue
        index = min(searchsortedlast(edges,value),length(counts))
        counts[index]+=1
    end
    density && (counts ./= sum(counts).*diff(edges))
    return maximum(counts)
end

function structural_summary(values)
    return DataFrame("Node"=>names(values),"Mean"=>[mean(values[!,name]) for name in names(values)],"Standard deviation"=>[std(values[!,name]) for name in names(values)],"Minimum"=>[minimum(values[!,name]) for name in names(values)],"Maximum"=>[maximum(values[!,name]) for name in names(values)])
end

is_readonly(values) = values isa RiskLabAI.CausalFactorAnalysis._CausalReadOnlyArray

export generated_did, generated_iv, histogram_peak, structural_summary, is_readonly

end
