module BookCausalStudy

using CSV, DataFrames, Dates, LinearAlgebra, Statistics, SHA, TOML, Plots
using ..BookSupport
import ..BookSupport: palette
using ..CausalNotebookTools

const violet = "#C084FC"
const muted = "#94A3B8"

frame(columns,rows) = DataFrame([name=>[row[i] for row in rows] for (i,name) in enumerate(columns)])
ordered_edges(graph) = [edge for node in graph.nodes for edge in graph.edges if first(edge)==node]
subgraph(graph,nodes) = DirectedGraph(filter(in(nodes),graph.nodes),filter(edge->all(in(nodes),edge),graph.edges))
acyclic(graph) = (CausalDAG(graph.nodes,graph.edges,graph.nodes);true)

function primitive_input(directory,seed)
    name = "primitive_$(seed).csv"
    path = joinpath(directory,name)
    manifest = TOML.parsefile(joinpath(directory,"inputs.toml"))
    bytes2hex(sha256(read(path)))==manifest["files"][name]["sha256"] || error("Primitive input hash differs")
    return Matrix{Float64}(CSV.read(path,DataFrame))
end

function reachable(graph,start;reverse=false)
    visited,pending = Set{String}(),[String(start)]
    while !isempty(pending)
        current = pop!(pending)
        for edge in graph.edges
            left,right = reverse ? (edge[2],edge[1]) : edge
            if left==current && !(right in visited)
                push!(visited,right)
                push!(pending,right)
            end
        end
    end
    return visited
end

function graph_paths(graph,source,target;directed=false)
    adjacency = Dict(node=>String[] for node in graph.nodes)
    for (left,right) in ordered_edges(graph)
        right in adjacency[left] || push!(adjacency[left],right)
        !directed && !(left in adjacency[right]) && push!(adjacency[right],left)
    end
    result = Vector{String}[]
    function visit(path)
        if last(path)==target
            push!(result,path)
            return
        end
        for neighbor in adjacency[last(path)]
            neighbor in path || visit(vcat(path,neighbor))
        end
    end
    visit([String(source)])
    return result
end

function path_records(graph,source,target,conditioned=String[])
    activated = Set{String}(conditioned)
    for node in conditioned
        union!(activated,reachable(graph,node;reverse=true))
    end
    rows = Tuple[]
    for path in graph_paths(graph,source,target)
        colliders = [path[i] for i in 2:length(path)-1 if (path[i-1],path[i]) in graph.edges && (path[i+1],path[i]) in graph.edges]
        active = all(node->node in colliders ? node in activated : !(node in conditioned),path[2:end-1])
        push!(rows,(join(path," - "),isempty(conditioned) ? "None" : join(sort(collect(conditioned)),", "),isempty(colliders) ? "None" : join(colliders,", "),active))
    end
    return frame(["Path","Conditioning set","Collider nodes","Active"],rows)
end

function partial_correlation(x,y,controls=nothing)
    isnothing(controls) && return cor(x,y)
    design = hcat(ones(length(x)),Matrix(controls))
    return cor(x-design*(design\x),y-design*(design\y))
end

function regression_summary(values,outcome,treatment,controls)
    response = values[!,outcome]
    design = hcat(ones(nrow(values)),Matrix(values[:,vcat(treatment,controls)]))
    coefficients = design\response
    residual = response-design*coefficients
    return coefficients[2],1-sum(abs2,residual)/sum(abs2,response.-mean(response))
end

function fit_sem_covariance(covariance,graph,nodes)
    coefficients,residual = zeros(size(covariance)),zeros(length(nodes))
    for (index,node) in enumerate(nodes)
        parents = [findfirst(==(left),nodes) for (left,right) in graph.edges if right==node]
        if isempty(parents)
            residual[index] = covariance[index,index]
        else
            beta = vec(covariance[index,parents]'*inv(covariance[parents,parents]))
            coefficients[index,parents] = beta
            residual[index] = covariance[index,index]-dot(beta,covariance[parents,index])
        end
    end
    transform = inv(I-coefficients)
    return coefficients,residual,transform*Diagonal(residual)*transform'
end

role_color(role) = get(Dict("confounder"=>palette.amber,"mediator"=>palette.green,"collider"=>palette.rose,"instrument"=>violet,"ancestor_of_outcome"=>palette.cyan,"descendant_of_outcome"=>muted,"descendant_of_treatment"=>muted,"independent"=>palette.grid),role,palette.cyan)

function graph_panel(graph,positions,title;colors=Dict(),labels=Dict(),directed=true,dashed=false,widths=nothing)
    xs,ys = [positions[node][1] for node in graph.nodes],[positions[node][2] for node in graph.nodes]
    spanx,spany = max(maximum(xs)-minimum(xs),2.),max(maximum(ys)-minimum(ys),2.)
    rx,ry = .085spanx,.085spany
    figure = plot(;title,legend=false,axis=false,grid=false,xlims=(minimum(xs)-2rx,maximum(xs)+2rx),ylims=(minimum(ys)-2ry,maximum(ys)+2ry),widen=false)
    for (index,(left,right)) in enumerate(graph.edges)
        x1,y1 = positions[left]
        x2,y2 = positions[right]
        dx,dy = x2-x1,y2-y1
        fraction = min(abs(dx)>0 ? rx/abs(dx) : Inf,abs(dy)>0 ? ry/abs(dy) : Inf,.42)
        plot!(figure,[x1+fraction*dx,x2-fraction*dx],[y1+fraction*dy,y2-fraction*dy];color=palette.text,linewidth=isnothing(widths) ? 1.8 : widths[index],linestyle=dashed ? :dash : :solid,arrow=directed ? Plots.arrow(:closed,.14,.12) : nothing,label="")
    end
    for node in graph.nodes
        x,y = positions[node]
        plot!(figure,Shape([x-rx,x+rx,x+rx,x-rx],[y-ry,y-ry,y+ry,y+ry]);fillcolor=get(colors,node,palette.cyan),linecolor=palette.text,linewidth=.8,label="")
        ink = get(colors,node,palette.cyan)==palette.grid ? palette.text : palette.background
        annotate!(figure,x,y,text(get(labels,node,node),9,ink,:center))
    end
    return figure
end

function matrix_panel(values,columns,title;rows=columns,limits=(-1,1),colors=:RdBu,annotate=false)
    figure = heatmap(1:length(columns),1:length(rows),values;title,color=colors,clims=limits,colorbar=true,yflip=true,xticks=(1:length(columns),columns),yticks=(1:length(rows),rows),xrotation=35,xlims=(.5,length(columns)+.5),ylims=(.5,length(rows)+.5),widen=false,legend=false)
    if annotate
        for row in eachindex(rows),column in eachindex(columns)
            annotate!(figure,column,row,text(string(Int(values[row,column])),10,values[row,column]==0 ? palette.text : palette.background))
        end
    end
    return figure
end

function spring_positions(adjacency,initial;iterations=50,threshold=1e-4)
    positions = copy(initial)
    count = size(adjacency,1)
    k = sqrt(1/count)
    temperature = maximum(maximum(positions;dims=1)-minimum(positions;dims=1))*.1
    cooling = temperature/(iterations+1)
    for iteration in 1:iterations
        displacement = zeros(size(positions))
        for i in 1:count,j in 1:count
            delta = positions[i,:]-positions[j,:]
            distance = max(norm(delta),.01)
            displacement[i,:] .+= delta.*(k*k/distance^2-adjacency[i,j]*distance/k)
        end
        lengths = max.(sqrt.(sum(abs2,displacement;dims=2)),.01)
        step = displacement.*(temperature./lengths)
        positions .+= step
        temperature -= cooling
        norm(step)/count<threshold && break
    end
    positions .-= mean(positions;dims=1)
    limit = maximum(abs,positions)
    limit>0 && (positions ./= limit)
    return positions
end

export frame, ordered_edges, subgraph, acyclic, primitive_input, reachable, graph_paths,
    path_records, partial_correlation, regression_summary, fit_sem_covariance,
    role_color, graph_panel, matrix_panel, spring_positions, violet, muted

end
