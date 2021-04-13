export SOSPoly, SymmetricSOSPoly, gradient, set_params, hessian

abstract type AbstractSOSPoly end

mutable struct SOSPoly{VA, BA, CO<:AbstractVector{<:Real}} <: AbstractSOSPoly
    dim::Int
    degree::Int
    vars::VA
    mono::BA
    θ::CO
end

function SOSPoly(n::Int, degree::Int)
    @polyvar q[1:n]
    mono = monomials(q, 1:degree)
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

function hessian(P::SOSPoly, x, θ=P.θ)
    L = vec2tril(θ)
    X = P.mono
    sos = dot(X, (L*L')*X)
    gs = differentiate.(sos, P.vars)
    expr = [differentiate.(g, P.vars) for g in gs]
    val = [v(P.vars=>x) for v in reduce(hcat, expr)]    
end

function set_params(P::AbstractSOSPoly, p::Vector{<:Real})
    P.θ = p
end


mutable struct SymmetricSOSPoly{VA, BA, CO} <: AbstractSOSPoly
    dim::Int
    degree::Int
    vars::VA
    mono::BA
    θ::CO
end

function SymmetricSOSPoly(n::Int, degree::Int)
    # Currently only works for 2 dimension, 4th degree poly 
    @polyvar q[1:2]
    mono = monomials(q, 1:2) |> reverse
    mono = mono[[2,1,5,4,3]]
    m = length(mono)
    θ = glorot_uniform(9)
    SymmetricSOSPoly{typeof(q), typeof(mono), typeof(θ)}(n, degree, q, mono, θ)
end

function (P::SymmetricSOSPoly)(x, θ=P.θ)
    T = eltype(x)
    L = [
        vec2tril(θ[1:3]) zeros(T,2,3)
        zeros(T,3,2) vec2tril(θ[4:end])
    ]
    X = P.mono
    sos = dot(X, (L*L')*X)(P.vars=>x)
end

function gradient(P::SymmetricSOSPoly, x, θ=P.θ) 
    T = eltype(x)
    L = [
        vec2tril(θ[1:3]) zeros(T,2,3)
        zeros(T,3,2) vec2tril(θ[4:end])
    ]
    X = P.mono
    sos = dot(X, (L*L')*X)
    expr = differentiate.(sos, P.vars)
    val = reduce(hcat, p(P.vars=>x) for p in expr)
end
