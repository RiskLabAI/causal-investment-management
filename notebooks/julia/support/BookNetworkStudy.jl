module BookNetworkStudy

using CSV, DataFrames, Dates, LinearAlgebra, Statistics, Plots
using RiskLabAI.CausalFactorAnalysis: exposure_mean_contrast
using ..BookSupport
import ..BookSupport: palette
using ..BookCausalStudy: frame

function network_returns()
    lines=readlines(dataset_path("french_49_industries"))
    first=findfirst(line->startswith(line,",Agric"),lines)
    last=findnext(line->isempty(strip(line)),lines,first+1)-1
    labels=String.(split(lines[first],',')[2:end])
    dates=Date[]
    rows=Vector{Float64}[]
    for line in lines[first+1:last]
        fields=split(line,',')
        date=Date(strip(fields[1]),dateformat"yyyymmdd")
        Date(2019,1,1)<=date<=Date(2023,12,31) || continue
        values=parse.(Float64,strip.(fields[2:end]))
        any(value->value in (-99.99,-999.),values) && continue
        push!(dates,date)
        push!(rows,values./100)
    end
    return dates,labels,reduce(vcat,permutedims.(rows))
end

function correlation_graph(correlation,neighbors)
    unit_count=size(correlation,1)
    adjacency=zeros(Int,unit_count,unit_count)
    for row in 1:unit_count
        selected=filter(!=(row),sortperm(correlation[row,:];rev=true))[1:neighbors]
        adjacency[row,selected].=1
    end
    return max.(adjacency,adjacency')
end

function line_graph(size)
    adjacency=zeros(Int,size,size)
    for index in 1:size-1
        adjacency[index,index+1]=adjacency[index+1,index]=1
    end
    return adjacency
end

function ring_graph(size,radius=1)
    adjacency=zeros(Int,size,size)
    for index in 1:size,step in 1:radius
        adjacency[index,mod1(index+step,size)]=1
        adjacency[index,mod1(index-step,size)]=1
    end
    return adjacency
end

function star_graph(size)
    adjacency=zeros(Int,size,size)
    adjacency[1,2:end].=1
    adjacency[2:end,1].=1
    return adjacency
end

complete_graph(size)=ones(Int,size,size)-Matrix{Int}(I,size,size)
exposure_labels(adjacency,assignment,threshold=1)=Int.(adjacency*assignment.>=threshold)

function exposure_probabilities(adjacency,probability,threshold=1)
    degrees=vec(sum(adjacency;dims=2))
    positive=[1-sum(binomial(degree,count)*probability^count*(1-probability)^(degree-count) for count in 0:min(threshold,degree+1)-1) for degree in degrees]
    return hcat(1 .-positive,positive)
end

assignment_probability(assignment,probability)=probability^sum(assignment)*(1-probability)^(length(assignment)-sum(assignment))
all_assignments(size)=[[(code>>(size-index))&1 for index in 1:size] for code in 0:2^size-1]
exposure_result(outcome,exposure,probabilities)=exposure_mean_contrast(outcome,exposure.+1,probabilities;first=2,second=1)

function circle_positions(size)
    angles=pi/2 .+collect(0:size-1).*(2pi/size)
    return hcat(cos.(angles),sin.(angles))
end

function network_figure(adjacency,labels,values,title)
    positions=circle_positions(length(labels))
    figure=plot(;title,aspect_ratio=:equal,axis=false,grid=false,xlims=(-1.25,1.25),ylims=(-1.25,1.25),size=(1000,860),legend=false)
    for first in eachindex(labels),second in first+1:length(labels)
        adjacency[first,second]==1 || continue
        plot!(figure,positions[[first,second],1],positions[[first,second],2];color=palette.text,linewidth=1.5,alpha=.35,label="")
    end
    colors=ifelse.(values.<=median(values),palette.cyan,palette.amber)
    scatter!(figure,positions[:,1],positions[:,2];color=colors,markersize=29,markerstrokecolor=palette.text,markerstrokewidth=1,label="")
    for (index,label) in enumerate(labels)
        annotate!(figure,positions[index,1],positions[index,2],text(label,9,palette.background))
    end
    return figure
end

function network_estimates(graph,baseline,effects,probability,assignments;threshold=1)
    probabilities=exposure_probabilities(graph,probability,threshold)
    records=[]
    for repetition in axes(assignments,1)
        exposure=exposure_labels(graph,Int.(assignments[repetition,:]),threshold)
        outcome=baseline+effects.*exposure
        result=exposure_result(outcome,exposure,probabilities)
        exposed=sum(exposure)
        naive=0<exposed<length(exposure) ? mean(outcome[exposure.==1])-mean(outcome[exposure.==0]) : NaN
        push!(records,(repetition,exposed,result.contrast,naive))
    end
    return frame(["Repetition","Exposed units","Horvitz-Thompson contrast","Naive observed difference"],records)
end

function running_finite_mean(values)
    total=0.
    count=0
    result=Float64[]
    for value in values
        if isfinite(value)
            total+=value
            count+=1
        end
        push!(result,count==0 ? NaN : total/count)
    end
    return result
end

function network_failure(name,action)
    try
        action()
        return (name,false,"Calculation continued")
    catch error
        error isa ArgumentError || rethrow()
        return (name,true,"Stopped as required")
    end
end

function network_histogram(values,bins)
    edges=collect(range(minimum(values),maximum(values);length=bins+1))
    counts=zeros(Int,bins)
    for value in values
        counts[clamp(searchsortedlast(edges,value),1,bins)]+=1
    end
    density=counts./(length(values).*diff(edges))
    return DataFrame("Lower edge"=>edges[1:end-1],"Upper edge"=>edges[2:end],"Count"=>counts,"Density"=>density)
end

export network_returns, correlation_graph, line_graph, ring_graph, star_graph,
    complete_graph, exposure_labels, exposure_probabilities, assignment_probability,
    all_assignments, exposure_result, circle_positions, network_figure,
    network_estimates, running_finite_mean, network_failure, network_histogram

end
