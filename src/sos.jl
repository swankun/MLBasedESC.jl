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
    mono = monomials(q, 0:degree)
    m = length(mono)
    θ = glorot_uniform(Int(m*(m+1)/2))
    SOSPoly{typeof(q), typeof(mono), typeof(θ)}(n, degree, q, mono, θ)
end

function (P::SOSPoly)(x, θ=P.θ) 
    L = vec2tril(θ)
    X = P.mono
    v = L'*X
    sos = dot(v, v)(P.vars=>x)
end

function coeff_matrix(P::SOSPoly, θ)
    return vec2tril(θ)
end

function gradient(P::SOSPoly, x, θ=P.θ)
    L = coeff_matrix(P, θ)
    X = P.mono
    v = L'*X
    sos = dot(v, v)
    expr = differentiate.(sos, P.vars)
    val = reduce(hcat, p(P.vars=>x) for p in expr)
end

function hessian(P::SOSPoly, x, θ=P.θ)
    L = coeff_matrix(P, θ)
    X = P.mono
    v = L'*X
    sos = dot(v, v)
    gs = differentiate.(sos, P.vars)
    expr = [differentiate.(g, P.vars) for g in gs]
    val = [v(P.vars=>x) for v in reduce(hcat, expr)]    
end

function set_params(P::AbstractSOSPoly, p::Vector{<:Real})
    P.θ = p
end

nmons(n,d) = binomial(n+d-1,d)

mutable struct SymmetricSOSPoly{VA, BA, CO} <: AbstractSOSPoly
    dim::Int
    degree::Int
    vars::VA
    mono::BA
    θ::CO
    i::Vector{Int}
    j::Vector{Int}
end

function SymmetricSOSPoly(n::Int, degree::Int)
    @polyvar q[1:2]
    
    if degree != 8
        mono = monomials(q, 1:2) 
        θ = glorot_uniform(9)
        rows = [1, 2, 2, 3, 3, 3, 4, 5, 5]
        cols = [1, 1, 2, 1, 2, 3, 4, 4, 5]
    else
        mono = monomials(q, 1:4)
        θ = glorot_uniform(57)
        rows = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 11, 11, 12, 12, 12, 13, 14, 14, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14]
        cols = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 6, 7, 6, 7, 8, 6, 7, 8, 9, 10, 10, 11, 10, 11, 12, 13, 13, 14, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9]
    end

    SymmetricSOSPoly{typeof(q), typeof(mono), typeof(θ)}(n, degree, q, mono, θ, rows, cols)
end

function coeff_matrix(P::SymmetricSOSPoly, θ)
    T = eltype(θ)
    L = sparse(P.i, P.j, θ)
end

function (P::SymmetricSOSPoly)(x, θ=P.θ)
    T = eltype(x)
    L = coeff_matrix(P, θ)
    X = P.mono
    v = L'*X
    sos = dot(v, v)(P.vars=>x)
end

function gradient(P::SymmetricSOSPoly, x, θ=P.θ) 
    T = eltype(x)
    L = coeff_matrix(P, θ)
    X = P.mono
    v = L'*X
    sos = dot(v, v)
    expr = differentiate(sos, P.vars)
    val = reduce(hcat, p(P.vars=>x) for p in expr)
end

function test_symmetry(P::SymmetricSOSPoly)
    val = randn(Float32,2)
    # P(val) == P(-val)
    [P(val) P(-val)]
end

function test_gradient(P::SymmetricSOSPoly)
    val = randn(Float32,2)
    # ReverseDiff.gradient(P,val) ≈ gradient(P,val)[:]
    [ReverseDiff.gradient(P,val) gradient(P,val)[:]]
end
