module BookConditionDiagnostics

using LinearAlgebra
using LinearAlgebra.BLAS: @blasfunc

function generate!(matrix, position, right)
    rows, columns = size(matrix)
    column = right ? position + 1 : position
    count = right ? columns - position : rows - position + 1
    alpha = Ref{Float64}(matrix[position, column])
    first = (column - 1) * rows + position
    next = count == 1 ? first : first + (right ? rows : 1)
    tau = Ref{Float64}(0)
    ccall((@blasfunc(dlarfg_), LinearAlgebra.LAPACK.liblapack), Cvoid,
        (Ref{LinearAlgebra.BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ref{Float64}),
        count, alpha, pointer(matrix, next), right ? rows : 1, tau)
    matrix[position, column] = alpha[]
    return alpha[], tau[]
end

function apply!(matrix, position, right, tau)
    tau == 0 && return nothing
    rows, columns = size(matrix)
    block = right ? view(matrix, position+1:rows, position+1:columns) : view(matrix, position:rows, position+1:columns)
    vector = right ? view(matrix, position, position+2:columns) : view(matrix, position+1:rows, position)
    leading = right ? view(block, :, 1) : view(block, 1, :)
    remaining = right ? view(block, :, 2:size(block, 2)) : view(block, 2:size(block, 1), :)
    work = zeros(length(leading))
    if isempty(vector)
        BLAS.scal!(1-tau, leading)
        return nothing
    end
    BLAS.gemv!(right ? 'N' : 'T', 1., remaining, vector, 0., work)
    BLAS.axpy!(1., leading, work)
    BLAS.axpy!(-tau, work, leading)
    right ? BLAS.ger!(-tau, work, vector, remaining) : BLAS.ger!(-tau, vector, work, remaining)
    return nothing
end

"""Compute the existing 4-by-3 condition diagnostic with an implicit-unit reflector update and the unchanged Julia BLAS/LAPACK backend."""
function exposure_condition(input::AbstractMatrix{Float64})
    size(input) == (4, 3) || return cond(input)
    all(isfinite, input) || throw(ArgumentError("The exposure matrix must be finite."))
    scale = maximum(abs, input)
    scale == 0 && return Inf
    1e-100 <= scale <= 1e100 || return cond(input)
    matrix = Matrix(input)
    diagonal, offdiagonal = zeros(3), zeros(3)
    for position in 1:3
        diagonal[position], tau = generate!(matrix, position, false)
        if position < 3
            apply!(matrix, position, false, tau)
            offdiagonal[position], tau = generate!(matrix, position, true)
            apply!(matrix, position, true, tau)
        end
    end
    values = first(LAPACK.bdsdc!('U', 'N', diagonal, offdiagonal))
    return last(values) == 0 ? Inf : first(values)/last(values)
end

end
