export PSDMatrix, gradient

struct PSDMatrix{T} <: FunctionApproxmiator
    n::Integer
    θ::Vector{T}
    nin::Integer
end

function PSDMatrix(T::DataType, n, nin)
    Nθ = Int(n*(n+1)/2)
    θ = Flux.glorot_normal(Nθ)
    PSDMatrix{T}(n, θ, nin)
end

function (m::PSDMatrix)(x, θ=m.θ) 
    L = vec2tril(θ)
    return L*L' + eltype(x)(1e-4)*I(m.n)
end

function gradient(m::PSDMatrix, x, θ=m.θ)
    return [zeros(eltype(x), (m.n,m.n)) for _=1:m.nin]
end
