module BookTransportStudy

using DataFrames, Statistics, Plots
using RiskLabAI.CausalFactorAnalysis: transported_ate
using ..BookSupport
import ..BookSupport: palette
using ..BookCausalStudy: frame
using ..BookSensitivityStudy: sensitivity_counties
using ..BookIdentificationStudy: normal_block

function transport_partition(values,count)
    edges=unique(quantile(values,collect(range(0,1;length=count+1))))
    return [clamp(searchsortedfirst(edges,value)-2,0,length(edges)-2) for value in values]
end

function transport_counties()
    raw,base=sensitivity_counties()
    result=base[:,["County identifier","Treatment group","Log employment in 2003","Log employment in 2007","Log population in 2003","Employment change, 2003 to 2007"]]
    population_median=median(result[:,5])
    result[!,"Domain"]=ifelse.(result[:,5].<=population_median,"Source","Target")
    result[!,"Stratum"]=transport_partition(result[:,3],3)
    labels=["Low baseline employment","Middle baseline employment","High baseline employment"]
    result[!,"Stratum name"]=labels[result.Stratum.+1]
    return raw,result,population_median
end

function transport_summary(values,column="Stratum")
    strata=sort(unique(values[!,column]))
    source=values[values.Domain.=="Source",:]
    target=values[values.Domain.=="Target",:]
    records=[]
    for stratum in strata
        source_rows=source[source[!,column].==stratum,:]
        control=source_rows[source_rows[:,2].==0,6]
        treated=source_rows[source_rows[:,2].==1,6]
        control_mean=isempty(control) ? NaN : mean(control)
        treated_mean=isempty(treated) ? NaN : mean(treated)
        target_count=count(==(stratum),target[!,column])
        push!(records,(stratum,length(control),length(treated),control_mean,treated_mean,treated_mean-control_mean,nrow(source_rows)/nrow(source),target_count,target_count/nrow(target)))
    end
    return frame(["Stratum","Source control count","Source treated count","Source control mean","Source treated mean","Source contrast","Source probability","Target count","Target probability"],records)
end

function transport_failure(name,action)
    try
        action()
        return (name,false,"No error")
    catch error
        error isa ArgumentError || rethrow()
        return (name,true,sprint(showerror,error))
    end
end

function paired_bars(labels,left,right,title,ylabel;left_label="Source",right_label="Target",right_color=palette.amber,size=(1100,620))
    positions=collect(1:length(labels))
    width=.36
    figure=bar(positions.-width/2,left;bar_width=width,color=palette.cyan,label=left_label,xticks=(positions,labels),ylabel,title,size)
    bar!(figure,positions.+width/2,right;bar_width=width,color=right_color,label=right_label)
    hline!(figure,[0.];color=palette.text,linewidth=.9,label="")
    lower,upper=extrema(vcat(left,right,[0.]))
    span=max(upper-lower,eps())
    plot!(figure;ylims=(lower<0 ? lower-.12span : 0.,upper+.2span))
    return figure
end

function generated_transport(store)
    source_rows=[]
    target_rows=[]
    effects=.2 .+.4 .*collect(0:3)
    for stratum in 0:3
        block=normal_block(store,"source"*string(stratum))
        for pair in axes(block,1)
            baseline=-.3+.25stratum+.2block[pair,1]
            orientation=Int(block[pair,2])
            for member in 0:1
                treatment=member==0 ? orientation : 1-orientation
                push!(source_rows,(stratum,pair-1,treatment,baseline,baseline+effects[stratum+1],baseline+treatment*effects[stratum+1]))
            end
        end
    end
    for stratum in 0:3
        for innovation in normal_block(store,"target"*string(stratum))[:,1]
            baseline=-.3+.25stratum+.2innovation
            push!(target_rows,(stratum,baseline,baseline+effects[stratum+1]))
        end
    end
    source=frame(["Stratum","Pair","Treatment","Untreated potential outcome","Treated potential outcome","Observed outcome"],source_rows)
    target=frame(["Stratum","Untreated potential outcome","Treated potential outcome"],target_rows)
    return source,target,effects
end

export transport_partition, transport_counties, transport_summary,
    transport_failure, paired_bars, generated_transport

end
