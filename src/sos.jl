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
    
    # deg=4
    # mono = monomials(q, 1:2) |> reverse
    # mono = mono[[2,1,5,4,3]]
    # m = length(mono)
    # θ = glorot_uniform(9)

    # deg=8
    mono = monomials(q, 1:4)
    mono = reduce(vcat, [getindex(mono, i) for i in [13:14, 10:12, 6:9, 1:5]])
    # m = length(mono)
    θ = glorot_uniform(57)
    row = [1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 11, 11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14]
    col = [1, 1, 2, 3, 3, 4, 3, 4, 5, 6, 6, 7, 6, 7, 8, 6, 7, 8, 9, 10, 10, 11, 10, 11, 12, 10, 11, 12, 13, 10, 11, 12, 13, 14, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]

    SymmetricSOSPoly{typeof(q), typeof(mono), typeof(θ)}(n, degree, q, mono, θ, row, col)
end

function coeff_matrix(P::SymmetricSOSPoly, θ)
    T = eltype(θ)
    L = sparse(P.i, P.j, θ)
    # partlen = cumsum(2:5) .+ 1
    # start = [1; partlen[1:end-1]]
    # idx = [range(a, step=1, length=b) for (a,b) in zip(cumsum(start), partlen)]
    # A,B,C,D = [vec2tril(θ[i]) for i in idx]
    # L = [
    #     A zeros(T,(2,3+4+5))
    #     zeros(T,(3,2)) B zeros(T,(3,4+5))
    #     zeros(T,(4,2+3)) C zeros(T,(4,5))
    #     zeros(T,(5,2+3+4)) D
    # ]
    # L = zeros(14, 14)
    # for (ci, fi) in zip(start,idx)
    #     l = vec2tril(θ[fi])
    #     n = size(l,1)
    #     L[ci:ci+n, ci:ci+n] = l 
    # end
end

function (P::SymmetricSOSPoly)(x, θ=P.θ)
    T = eltype(x)

    # deg=4
    # L = [
    #     vec2tril(θ[1:3]) zeros(T,2,3)
    #     zeros(T,3,2) vec2tril(θ[4:end])
    # ]

    # deg=8
    L = coeff_matrix(P, θ)

    X = P.mono
    sos = dot(X, (L'*L)*X)(P.vars=>x)
end

function gradient(P::SymmetricSOSPoly, x, θ=P.θ) 
    T = eltype(x)

    # deg=4
    # L = [
    #     vec2tril(θ[1:3]) zeros(T,2,3)
    #     zeros(T,3,2) vec2tril(θ[4:end])
    # ]

    # deg=8
    L = coeff_matrix(P, θ)

    X = P.mono
    sos = dot(X, (L'*L)*X)
    expr = differentiate.(sos, P.vars)
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
