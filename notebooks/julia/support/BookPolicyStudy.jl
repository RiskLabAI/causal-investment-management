module BookPolicyStudy

using DataFrames,Dates,LinearAlgebra,Statistics,Plots
using RiskLabAI.CausalFactorAnalysis: evaluate_policy,select_policy
using ..BookSupport
using ..BookCausalStudy: frame
using ..BookIdentificationStudy: PrimitiveStore,normal_block

logistic(values)=1 ./(1 .+exp.(-clamp.(values,-35,35)))

function public_baskets()
    lines=readlines(dataset_path("french_49_industries"))
    first=findfirst(line->startswith(line,",Agric"),lines)
    last=findnext(line->isempty(strip(line)),lines,first+1)-1
    labels=String.(split(lines[first],',')[2:end])
    defensive=[findfirst(==(name),labels) for name in ["Food","Smoke","Hlth","Util"]]
    cyclical=[findfirst(==(name),labels) for name in ["BldMt","Steel","Mach","Autos"]]
    dates=Date[]
    rows=Vector{Float64}[]
    for line in lines[first+1:last]
        fields=split(line,',')
        date=Date(strip(fields[1]),dateformat"yyyymmdd")
        values=parse.(Float64,strip.(fields[2:end]))
        values[in.(values,Ref([-99.99,-999.]))].=NaN
        values./=100
        groups=[filter(isfinite,values[indices]) for indices in [defensive,cyclical]]
        push!(rows,[isempty(group) ? NaN : mean(group) for group in groups])
        push!(dates,date)
    end
    return dates,reduce(vcat,permutedims.(rows))
end

function build_policy_frame(store;floor=.1)
    dates,baskets=public_baskets()
    spread=baskets[:,2]-baskets[:,1]
    market=vec(mean(baskets;dims=2))
    context=fill(NaN,length(dates),3)
    for index in 2:length(dates)
        context[index,3]=spread[index-1]
        if index>20
            context[index,1]=mean(spread[index-20:index-1])
            context[index,2]=std(market[index-20:index-1];corrected=true)
        end
    end
    complete=vec(all(isfinite.(context);dims=2))
    reference=findall((Date(2018,1,1).<=dates.<=Date(2018,12,31)).&complete)
    center=mean(context[reference,:];dims=1)
    scale=std(context[reference,:];dims=1,corrected=true)
    standardized=(context.-center)./scale
    selected=findall((Date(2019,1,1).<=dates.<=Date(2023,12,31)).&complete.&vec(all(isfinite.(baskets);dims=2)))
    output=hcat(DataFrame("Date"=>dates[selected]),DataFrame(baskets[selected,:],["Defensive return","Cyclical return"]),DataFrame(context[selected,:],["Past mean spread","Past volatility","Past one-day spread"]),DataFrame(standardized[selected,:],["Past mean spread z","Past volatility z","Past one-day spread z"]))
    score=.7output[:,"Past mean spread z"]-.4output[:,"Past volatility z"]+.2output[:,"Past one-day spread z"]
    probability_one=floor.+(1-2floor).*logistic(score)
    uniforms=vec(normal_block(store,"actions"))
    length(uniforms)==nrow(output) || error("The preserved action stream must match the public sample")
    actions=Int.(uniforms.<probability_one)
    output[!,"Behavior probability action 1"]=probability_one
    output[!,"Behavior probability action 0"]=1 .-probability_one
    output[!,"Action"]=actions
    output[!,"Observed reward"]=[baskets[selected[index],actions[index]+1] for index in eachindex(selected)]
    output[!,"Uniform draw"]=uniforms
    return output
end

behavior_matrix(panel)=Matrix(panel[:,["Behavior probability action 0","Behavior probability action 1"]])
potential_matrix(panel)=Matrix(panel[:,["Defensive return","Cyclical return"]])
design_matrix(panel)=hcat(ones(nrow(panel)),Matrix(panel[:,["Past mean spread z","Past volatility z","Past one-day spread z"]]))

function candidate_policies(panel)
    spread=panel[:,"Past mean spread z"]
    volatility=panel[:,"Past volatility z"]
    probabilities=[zeros(nrow(panel)),ones(nrow(panel)),Float64.(spread.>0),logistic(1.5spread),logistic(1.2spread-.6volatility)]
    candidates=zeros(5,nrow(panel),2)
    for (index,values) in enumerate(probabilities)
        candidates[index,:,:]=hcat(1 .-values,values)
    end
    labels=["Always defensive","Always cyclical","Positive spread threshold","Smooth momentum","Volatility aware rule"]
    return labels,candidates
end

full_information_value(panel,policy,rows=1:nrow(panel))=mean(vec(sum(policy[rows,:].*potential_matrix(panel)[rows,:];dims=2)))

function fitted_coefficients(panel,training_rows)
    features=design_matrix(panel)
    actions=panel.Action
    rewards=panel[:,"Observed reward"]
    coefficients=zeros(2,4)
    for action in 0:1
        rows=training_rows[actions[training_rows].==action]
        coefficients[action+1,:]=features[rows,:]\rewards[rows]
    end
    return coefficients
end

function chronological_predictions(panel)
    boundaries=[200,370,540,710,880,nrow(panel)]
    features=design_matrix(panel)
    predictions=zeros(nrow(panel),2)
    folds=[]
    coefficient_rows=[]
    for (fold,(start,ending)) in enumerate(zip(boundaries[1:end-1],boundaries[2:end]))
        coefficients=fitted_coefficients(panel,collect(1:start))
        predictions[start+1:ending,:]=features[start+1:ending,:]*coefficients'
        push!(folds,(string(panel.Date[1]),string(panel.Date[start]),string(panel.Date[start+1]),string(panel.Date[ending]),start,ending-start,panel.Date[start]<panel.Date[start+1]))
        for action in 0:1
            push!(coefficient_rows,(fold,start,action,coefficients[action+1,:]...))
        end
    end
    plan=frame(["Training first date","Training last date","Prediction first date","Prediction last date","Training rows","Prediction rows","Training precedes prediction"],folds)
    coefficients=frame(["Fold","Training rows","Action","Intercept","Spread coefficient","Volatility coefficient","One-day spread coefficient"],coefficient_rows)
    return predictions,plan,coefficients
end

policy_evaluation(actions,rewards,behavior,policy,predictions)=evaluate_policy(actions.+1,rewards,behavior,policy,predictions)
policy_selection(actions,rewards,behavior,candidates,predictions;training_indices,evaluation_indices)=select_policy(actions.+1,rewards,behavior,candidates,predictions;training_indices=training_indices,evaluation_indices=evaluation_indices)

function capture_failure(name,calculation)
    try
        calculation()
        return (name,false,"Calculation continued")
    catch error
        error isa ArgumentError || rethrow()
        return (name,true,"Stopped as required")
    end
end

function histogram_counts(values,bins)
    edges=collect(range(minimum(values),maximum(values);length=bins+1))
    counts=zeros(Int,bins)
    for value in values
        counts[clamp(searchsortedlast(edges,value),1,bins)]+=1
    end
    return DataFrame("Lower edge"=>edges[1:end-1],"Upper edge"=>edges[2:end],"Count"=>counts)
end

function bar_headroom!(figure,values)
    lower=min(0.,minimum(values))
    upper=max(0.,maximum(values))
    padding=.08*(upper-lower)
    plot!(figure;ylims=(lower-padding,upper+padding))
    return figure
end

export build_policy_frame,behavior_matrix,potential_matrix,candidate_policies,full_information_value,design_matrix,fitted_coefficients,chronological_predictions,policy_evaluation,policy_selection,capture_failure,histogram_counts,bar_headroom!

end
