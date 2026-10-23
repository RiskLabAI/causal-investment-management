module BookSearchStudy

using DataFrames, Dates, LinearAlgebra, Statistics, Plots
using ..BookSupport
import ..BookSupport: palette
using ..BookCausalStudy: frame

function search_returns()
    lines=readlines(dataset_path("french_49_industries"))
    first=findfirst(line->startswith(line,",Agric"),lines)
    last=findnext(line->isempty(strip(line)),lines,first+1)-1
    labels=String.(split(lines[first],',')[2:end])
    dates=Date[]
    rows=Vector{Float64}[]
    for line in lines[first+1:last]
        fields=split(line,',')
        date=Date(strip(fields[1]),dateformat"yyyymmdd")
        Date(2019,1,1)<=date<=Date(2025,12,31) || continue
        values=parse.(Float64,strip.(fields[2:end]))
        any(value->value in (-99.99,-999.),values) && continue
        push!(dates,date)
        push!(rows,values./100)
    end
    return dates,labels,reduce(vcat,permutedims.(rows))
end

function kkt_oracle(covariance,exposures,targets)
    assets,factors=size(exposures)
    system=[2covariance exposures; exposures' zeros(factors,factors)]
    solution=system\vcat(zeros(assets),targets)
    return solution[1:assets],solution[assets+1:end]
end

portfolio_variance(covariance,weights)=dot(weights,covariance*weights)
normal_density(value,scale=1.)=exp(-.5*(value/scale)^2)/(sqrt(2pi)*scale)
allclose(left,right;rtol=1e-5,atol=1e-8)=all(abs.(left.-right).<=atol.+rtol.*abs.(right))

function contract_result(name,action)
    try
        action()
        return (name,"ArgumentError","Accepted",false)
    catch error
        return (name,"ArgumentError",string(nameof(typeof(error))),error isa ArgumentError)
    end
end

function mutation_rejected(action)
    try
        action()
        return false
    catch
        return true
    end
end

function readonly_evidence(value)
    value isa Tuple && return true
    before=collect(value)
    rejected=mutation_rejected(()->setindex!(value,first(value),1))
    return rejected && collect(value)==before
end

function record_evidence(value;sources=())
    field=first(fieldnames(typeof(value)))
    frozen=mutation_rejected(()->setfield!(value,field,getproperty(value,field)))
    arrays=[getproperty(value,name) for name in fieldnames(typeof(value)) if getproperty(value,name) isa AbstractArray]
    pairwise_separate=all(!Base.mightalias(arrays[left],arrays[right]) for left in eachindex(arrays) for right in left+1:length(arrays))
    source_separate=all(!Base.mightalias(array,source) for array in arrays for source in sources)
    separate=pairwise_separate&&source_separate
    return (string(nameof(typeof(value))),frozen,separate,frozen&&separate)
end

function paired_weights(labels,first,second;first_label="Reference",second_label="Misspecified",title="",size=(900,600))
    locations=collect(1:length(labels))
    figure=bar(locations.-.19,first;bar_width=.38,color=palette.cyan,label=first_label,xticks=(locations,labels),xrotation=length(labels)>4 ? 50 : 0,ylabel="Weight",title=title,size=size)
    bar!(figure,locations.+.19,second;bar_width=.38,color=palette.rose,label=second_label)
    hline!(figure,[0.];color=palette.text,linewidth=.8,label="")
    lower=min(0.,minimum(first),minimum(second))
    upper=max(0.,maximum(first),maximum(second))
    padding=.08*(upper-lower)
    plot!(figure;ylims=(lower-padding,upper+padding))
    return figure
end

export search_returns,kkt_oracle,portfolio_variance,normal_density,allclose,contract_result,mutation_rejected,readonly_evidence,record_evidence,paired_weights

end
