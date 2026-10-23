module BookProtocolStudy

using DataFrames,Dates,LinearAlgebra,Statistics,Plots,Distributions,Printf
using RiskLabAI.CausalFactorAnalysis
using ..BookSupport
import ..BookSupport: palette
using ..BookCausalStudy: frame,violet,muted

function update_record(value;kwargs...)
    fields=fieldnames(typeof(value))
    original=NamedTuple{fields}(Tuple(getfield(value,name) for name in fields))
    return typeof(value)(;merge(original,(;kwargs...))...)
end

function update_record(value::CausalFactorProtocolReport;kwargs...)
    fields=Tuple(name for name in fieldnames(typeof(value)) if name∉(:source_oracle,:source_locator))
    original=NamedTuple{fields}(Tuple(getfield(value,name) for name in fields))
    return typeof(value)(;merge(original,(;kwargs...))...)
end

function human_label(value)
    special=Dict("PC"=>"PC","LINGAM"=>"LiNGAM","R_SQUARED"=>"R squared","MEAN_SQUARED_ERROR"=>"Mean squared error","BENJAMINI_HOCHBERG"=>"Benjamini-Hochberg","DEFLATED_SHARPE_RATIO"=>"Deflated Sharpe ratio")
    return get(special,value,titlecase(lowercase(replace(value,"_"=>" "))))
end

record_name(value)=strip(replace(string(nameof(typeof(value))),r"(?<!^)(?=[A-Z])"=>" "))

function contract_case(name,stage,action)
    try
        action()
        return (name,stage,"Accepted","Record satisfies the released checks",false)
    catch exception
        message=exception isa ArgumentError ? exception.msg : sprint(showerror,exception)
        message=replace(message,"Causal factor protocol is invalid. "=>"")
        message=rstrip(first(split(message,". ")),'.')
        return (name,stage,"Rejected",message,true)
    end
end

case_frame(rows)=frame(["Case","Stage","Result","Message","Rejected"],rows)

function make_horizons(ids)
    intervals=((0.,2.),(1.,3.),(4.,5.),(6.,7.),(8.,9.),(10.,11.))
    return Tuple(EventHorizon(ids[index],start,finish) for (index,(start,finish)) in enumerate(intervals))
end

function make_fold(ids,fold_id,path_id,train_positions,test_positions,refit_id)
    train_ids=Tuple(ids[index] for index in train_positions)
    test_ids=Tuple(ids[index] for index in test_positions)
    return FoldEvidence(fold_id=fold_id,path_id=path_id,train_ids=train_ids,test_ids=test_ids,fit_ids=train_ids,prediction_ids=test_ids,refit_id=refit_id,refit_components=("VARIABLE_SELECTION","CAUSAL_ESTIMATION","PORTFOLIO_CONSTRUCTION"))
end

function temporal_validation(ids;method_labels=("PURGED_CROSS_VALIDATION",),embargo=1.)
    folds=(make_fold(ids,"f1","p1",(3,4,5,6),(1,2),"r1"),make_fold(ids,"f2","p1",(1,2,5,6),(3,4),"r2"),make_fold(ids,"f3","p1",(1,2,3,4),(5,6),"r3"))
    return ValidationEvidence(method_labels=method_labels,event_horizons=make_horizons(ids),folds=folds,embargo=embargo)
end

function cpcv_validation(ids)
    folds=(make_fold(ids,"p1-f1","p1",(3,4,5,6),(1,2),"p1-r1"),make_fold(ids,"p1-f2","p1",(1,2,5,6),(3,4),"p1-r2"),make_fold(ids,"p1-f3","p1",(1,2,3,4),(5,6),"p1-r3"),make_fold(ids,"p2-f1","p2",(4,5,6),(1,3),"p2-r1"),make_fold(ids,"p2-f2","p2",(3,4,6),(2,5),"p2-r2"),make_fold(ids,"p2-f3","p2",(1,2,3),(4,6),"p2-r3"))
    return ValidationEvidence(method_labels=("COMBINATORIAL_PURGED_CROSS_VALIDATION",),event_horizons=make_horizons(ids),folds=folds,embargo=1.)
end

function valid_protocol_report(ids)
    trials=("strategy-a","strategy-b","strategy-c")
    family="declared-family"
    selection=VariableSelectionStage(purpose="RISK_PREMIA_HARVESTING",selected_variables=("X","Z","I","M","C","D"),method_labels=("MUTUAL_INFORMATION","SHAPLEY_VALUES","MEAN_DECREASE_IMPURITY","PERMUTATION_FEATURE_IMPORTANCE"),overlapping_returns=true,strong_time_dependence=true,validation=temporal_validation(ids))
    discovery=CausalDiscoveryStage(method_labels=("PC","ECONOMIC_REASONING","DOMAIN_EXPERTISE","OBSERVED_OUTCOMES"),graph_id="resolved-dag",graph_nodes=("X","Y","Z","I","M","C","D"),directed_edges=(("Z","X"),("Z","Y"),("I","X"),("X","M"),("M","Y"),("X","Y"),("X","C"),("Y","C"),("X","D")),graph_kind="DAG",ambiguous_edges=(),assumptions=("The recorded variables are sufficient for the target effect.","Directions use time order and economic restrictions."))
    adjustment=CausalAdjustmentSetStage(treatment="X",outcome="Y",method_label="BACKDOOR_ADJUSTMENT",identified=true,admissible_adjustment_sets=(("Z",),),selected_adjustment_set=("Z",),confounders=("Z",),descendants=("M","C","D"),mediators=("M",),colliders=("C",),instruments=("I",),control_justifications=(("Z","Common cause of factor and return."),),open_backdoor_paths=())
    performance=CausalExplanatoryAndPredictivePowerStage(task_types=("RETURN_SIZE",),estimator_label="fold-refit causal regression",explanatory_metric_labels=("R_SQUARED",),predictive_metric_labels=("MEAN_SQUARED_ERROR","SPEARMAN_CORRELATION"),explanatory_evidence_id="explanatory-evidence",predictive_evidence_id="predictive-evidence",naive_benchmark_id="training-mean-benchmark",validation=temporal_validation(ids))
    portfolio=CausalPortfolioConstructionStage(method_labels=("POSITION_SIZING","EXPOSURE_CONTROL","ECONOMIC_RATIONALE","FRAGILITY_STRESS_TEST","TRANSACTION_COST_OPTIMIZATION","TRANSFER_COEFFICIENT"),causal_exposures=("X",),controlled_unintended_exposures=("C","D"),cost_model_id="cost-record",constraint_set_id="constraint-record",economic_rationale="Positions target the supplied X to Y effect.",fragility_scenarios=("weaken X to Y","perturb Z to X"),transfer_coefficient=.82)
    backtest=BacktestStage(method_labels=("COMBINATORIAL_PURGED_CROSS_VALIDATION",),trial_family_id=family,declared_trial_ids=trials,validation=cpcv_validation(ids))
    multiple=MultipleTestingAdjustmentsStage(method_labels=("HOLM","DEFLATED_SHARPE_RATIO"),trial_family_id=family,declared_trial_ids=trials,family_partitions=(("primary",("strategy-a","strategy-b")),("robustness",("strategy-c",))),alpha=.05,backtests_independent=false,p_value_estimator="estimator accounting for time dependence",time_dependence_model="purged temporal folds",p_value_inputs_assume_independence=false,sharpe_variance=.04,effective_trials=2.,sample_length=252,skewness=.1,kurtosis=3.2,selection_bias_evidence_id="selection-bias-evidence")
    return CausalFactorProtocolReport(trial_family_id=family,declared_trial_ids=trials,variable_selection=selection,causal_discovery=discovery,causal_adjustment_set=adjustment,causal_explanatory_and_predictive_power=performance,causal_portfolio_construction=portfolio,backtest=backtest,multiple_testing_adjustments=multiple)
end

function protocol_returns()
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

function public_trial_returns(dates,returns)
    base=vec(mean(returns[:,1:12];dims=2))
    direction=vcat(NaN,ifelse.(base[1:end-1].>=0,1.,-1.))
    selected=findall((Date(2023,1,1).<=dates.<=Date(2025,12,31)).&isfinite.(direction))
    values=hcat(base,direction.*base,-direction.*base)
    return dates[selected],values[selected,:]
end

function heatmap_record(values,xlabels,ylabels;title="",xlabel="",ylabel="",colors=[palette.background,palette.cyan],limits=(0,1),labels=nothing,colorbar=false,colorbar_title="",size=(1000,550))
    figure=heatmap(1:length(xlabels),1:length(ylabels),values;c=cgrad(colors),clims=limits,yflip=true,xticks=(1:length(xlabels),xlabels),yticks=(1:length(ylabels),ylabels),xlabel=xlabel,ylabel=ylabel,title=title,colorbar=colorbar,colorbar_title=colorbar_title,size=size,xlims=(.5,length(xlabels)+.5),ylims=(.5,length(ylabels)+.5),grid=false)
    for row in axes(values,1),column in axes(values,2)
        message=isnothing(labels) ? string(Int(values[row,column])) : labels[row,column]
        color=values[row,column]==0 && first(colors)==palette.background ? palette.text : palette.background
        isempty(message) || annotate!(figure,column,row,text(message,9,color,:center))
    end
    return figure
end

function draw_causal_graph(edges)
    positions=Dict("I"=>(-1.7,.7),"Z"=>(-1.7,-.7),"X"=>(-.5,0.),"M"=>(.5,.8),"Y"=>(1.6,0.),"C"=>(.5,-.8),"D"=>(-.4,-1.25))
    colors=Dict("I"=>violet,"Z"=>palette.amber,"X"=>palette.cyan,"M"=>palette.green,"Y"=>palette.rose,"C"=>muted,"D"=>palette.text)
    figure=plot(;xlims=(-2.2,2.1),ylims=(-1.65,1.3),axis=false,grid=false,legend=false,aspect_ratio=:equal,title="Supplied directed graph",size=(1100,700))
    for (source,target) in edges
        start,finish=collect(positions[source]),collect(positions[target])
        unit=(finish-start)/norm(finish-start)
        left,right=start+.16unit,finish-.16unit
        plot!(figure,[left[1],right[1]],[left[2],right[2]];color=muted,linewidth=1.8,arrow=true,label="")
    end
    for node in ["I","Z","X","M","Y","C","D"]
        x,y=positions[node]
        angles=range(0,2pi;length=65)
        plot!(figure,Shape(x.+.15cos.(angles),y.+.15sin.(angles));color=colors[node],linecolor=palette.text,label="")
        annotate!(figure,x,y,text(node,12,palette.background,:center))
    end
    return figure
end

function colored_bars(locations,values,colors;kwargs...)
    figure=plot(;kwargs...)
    for (location,value,color) in zip(locations,values,colors)
        bar!(figure,[location],[value];color=color,bar_width=.8,kwargs...)
    end
    return figure
end

function bar_headroom!(figure,values;fraction=.08)
    lower=min(0.,minimum(values))
    upper=max(0.,maximum(values))
    padding=fraction*max(upper-lower,eps())
    plot!(figure;ylims=(lower-padding,upper+padding))
    return figure
end

function average_ranks(values)
    order=sortperm(values)
    ranks=zeros(length(values))
    first=1
    while first<=length(values)
        last=first
        while last<length(values) && values[order[last+1]]==values[order[first]]
            last+=1
        end
        ranks[order[first:last]].=(first+last)/2
        first=last+1
    end
    return ranks
end

function rolling_mean(values,width)
    result=fill(NaN,length(values))
    for index in width:length(values)
        result[index]=mean(values[index-width+1:index])
    end
    return result
end

function dependence_adjusted_p_value(values)
    centered=values.-mean(values)
    count=length(values)
    lag_limit=max(1,floor(Int,4(count/100)^(2/9)))
    covariances=[dot(centered[lag+1:end],centered[1:end-lag])/count for lag in 1:lag_limit]
    variance=dot(centered,centered)/count+2sum((1-lag/(lag_limit+1))*covariances[lag] for lag in 1:lag_limit)
    standard_error=sqrt(max(variance,floatmin(Float64))/count)
    statistic=mean(values)/standard_error
    return 2ccdf(Normal(),abs(statistic)),statistic,lag_limit,variance
end

function holm_adjusted(values)
    order=sortperm(values)
    ranked=min.(1.,(length(values):-1:1).*values[order])
    adjusted=accumulate(max,ranked)
    result=similar(adjusted)
    result[order]=adjusted
    return result
end

function benjamini_hochberg_adjusted(values)
    order=sortperm(values)
    ranked=min.(1.,length(values).*values[order]./(1:length(values)))
    adjusted=reverse(accumulate(min,reverse(ranked)))
    result=similar(adjusted)
    result[order]=adjusted
    return result
end

function expected_maximum_sharpe(variance,count)
    euler=.5772156649015329
    first=clamp(1-1/count,eps(),1-eps())
    second=clamp(1-1/(exp(1)*count),eps(),1-eps())
    return sqrt(variance)*((1-euler)*quantile(Normal(),first)+euler*quantile(Normal(),second))
end

function probabilistic_sharpe(observed,benchmark,count,skewness,kurtosis)
    denominator=sqrt(1-skewness*observed+((kurtosis-1)/4)*observed^2)
    return cdf(Normal(),(observed-benchmark)*sqrt(count-1)/denominator)
end

export colored_bars,average_ranks,update_record,human_label,record_name,contract_case,case_frame,valid_protocol_report,protocol_returns,public_trial_returns,heatmap_record,draw_causal_graph,bar_headroom!,rolling_mean,dependence_adjusted_p_value,holm_adjusted,benjamini_hochberg_adjusted,expected_maximum_sharpe,probabilistic_sharpe

end
