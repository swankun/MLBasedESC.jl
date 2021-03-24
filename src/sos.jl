export SOSPoly, gradient, set_params

mutable struct SOSPoly{VA, BA, CO<:AbstractVector{<:Real}}
    dim::Integer
    degree::Integer
    vars::VA
    mono::BA
    θ::CO
end

function SOSPoly(n::Int, degree::Int)
    @polyvar q[1:n]
    mono = monomials(q, 0:degree)
    m = length(mono)
    θ = glorot_uniform(Int(m*(m+1)/2))
    SOSPoly{typeof(q), typeof(mono), typeof(θ)}(n, degree, q, mono, θ)
end

function (P::SOSPoly)(x, θ=P.θ) 
    L = vec2tril(θ)
    X = P.mono
    sos = dot(X, (L*L')*X)(P.vars=>x)
end

function gradient(P::SOSPoly, x, θ=P.θ)
    L = vec2tril(θ)
    X = P.mono
    sos = dot(X, (L*L')*X)
    expr = differentiate.(sos, P.vars)
    val = reduce(hcat, p(P.vars=>x) for p in expr)
end

function set_params(P::SOSPoly, p::Vector{<:Real})
    P.θ = p
end
