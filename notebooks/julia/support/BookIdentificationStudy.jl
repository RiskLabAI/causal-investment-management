module BookIdentificationStudy

using CSV, DataFrames, LinearAlgebra, Statistics, SHA, TOML, Plots
using ..BookSupport
import ..BookSupport: palette
using ..CausalNotebookTools

struct PrimitiveStore
    path::String
    groups::Dict{String,Any}
    function PrimitiveStore(directory,section)
        manifest = TOML.parsefile(joinpath(directory,section*"_inputs.toml"))
        path = joinpath(directory,manifest["file"]["name"])
        bytes2hex(open(sha256,path))==manifest["file"]["sha256"] || error("Primitive stream hash differs")
        Base.ENDIAN_BOM==0x04030201 || error("This preserved stream requires little-endian Float64")
        new(path,manifest["groups"])
    end
end

function normal_block(store::PrimitiveStore,group,index=1)
    specification = store.groups[group]
    rows,columns = specification["rows"][index],specification["columns"]
    return open(store.path) do stream
        seek(stream,specification["offset_bytes"][index])
        values = Vector{Float64}(undef,rows*columns)
        read!(stream,values)
        reshape(values,rows,columns)
    end
end

function ols_slope(outcome,treatment,controls=nothing)
    design = hcat(ones(length(outcome)),treatment)
    if !isnothing(controls)
        design = hcat(design,controls)
    end
    return (design\outcome)[2]
end

mean_squared_error(values,truth) = mean(abs2,values.-truth)

function simulate_backdoor(store,n;delta=.8,a=.7,direct=.5,b=.6,gamma=.9,seed=240101)
    innovations = normal_block(store,"backdoor_$(seed)_$(n)")
    common = innovations[:,1]
    treatment = delta.*common+innovations[:,2]
    mediator = a.*treatment+innovations[:,3]
    outcome = direct.*treatment+b.*mediator+gamma.*common+innovations[:,4]
    collider = treatment+outcome+innovations[:,5]
    return DataFrame("Treatment"=>treatment,"Common cause"=>common,"Mediator"=>mediator,"Outcome"=>outcome,"Collider"=>collider,"Neutral"=>innovations[:,6])
end

function simulate_frontdoor(store,group,index=1;a=1.,b=1.,direct=0.,c_u=1.,g_u=1.,c_v=0.,g_v=0.)
    innovations = normal_block(store,group,index)
    common_xy,common_my = innovations[:,1],innovations[:,2]
    treatment = c_u.*common_xy+innovations[:,3]
    mediator = a.*treatment+c_v.*common_my+innovations[:,4]
    outcome = b.*mediator+direct.*treatment+g_u.*common_xy+g_v.*common_my+innovations[:,5]
    return treatment,mediator,outcome,common_xy,a*b+direct
end

function simulate_iv(store,group,index=1;gamma=1.,relevance=1.,proxy_noise=3.)
    innovations = normal_block(store,group,index)
    common,instrument = innovations[:,1],innovations[:,2]
    treatment = relevance.*instrument+common+innovations[:,3]
    outcome = .5treatment+gamma.*common+innovations[:,4]
    proxy = common+proxy_noise.*innovations[:,5]
    return treatment,outcome,instrument,proxy
end

function contract_result(name,action,expected)
    try
        action()
        return (name,"Rejected","Accepted",false)
    catch error
        passed = error isa expected
        return (name,"Rejected",passed ? "Rejected" : "Different rejection",passed)
    end
end

function identification_graph_panel(graph,positions,title;latent=String[],labels=Dict())
    xs,ys = [positions[node][1] for node in graph.nodes],[positions[node][2] for node in graph.nodes]
    spanx,spany = max(maximum(xs)-minimum(xs),2.),max(maximum(ys)-minimum(ys),1.5)
    rx,ry = .065spanx,.07spany
    figure = plot(;title,legend=false,axis=false,grid=false,xlims=(minimum(xs)-.22spanx,maximum(xs)+.22spanx),ylims=(minimum(ys)-.28spany,maximum(ys)+.28spany),widen=false)
    for (left,right) in graph.edges
        x1,y1 = positions[left]
        x2,y2 = positions[right]
        dx,dy = x2-x1,y2-y1
        fraction = min(abs(dx)>0 ? rx/abs(dx) : Inf,abs(dy)>0 ? ry/abs(dy) : Inf,.42)
        plot!(figure,[x1+fraction*dx,x2-fraction*dx],[y1+fraction*dy,y2-fraction*dy];color=left in latent ? "#C084FC" : palette.text,linestyle=left in latent ? :dash : :solid,linewidth=1.8,arrow=Plots.arrow(:closed,.14,.12),label="")
    end
    for node in graph.nodes
        x,y = positions[node]
        plot!(figure,Shape([x-rx,x+rx,x+rx,x-rx],[y-ry,y-ry,y+ry,y+ry]);fillcolor=node in latent ? "#C084FC" : palette.cyan,linecolor=palette.text,linewidth=.8,label="")
        label = get(labels,node,node)
        if length(label)<=3
            annotate!(figure,x,y,text(label,10,palette.background,:center))
        else
            offset = y>minimum(ys)+.2spany ? .17spany : -.19spany
            annotate!(figure,x,y+offset,text(label,9,palette.text,:center))
        end
    end
    return figure
end

export PrimitiveStore, normal_block, ols_slope, mean_squared_error,
    simulate_backdoor, simulate_frontdoor, simulate_iv, contract_result,
    identification_graph_panel

end
