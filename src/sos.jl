export SOSPoly, SymmetricSOSPoly, gradient, hessian
export IWPSOSPoly

abstract type AbstractSOSPoly <: FunctionApproxmiator end

struct SOSPoly{VA, BA, BA2, CO<:AbstractVector{<:Real}} <: AbstractSOSPoly
    dim::Int
    degree::Int
    vars::VA
    mono::BA
    ∇x_mono::BA2
    θ::CO
end

function SOSPoly(n::Int, degrees::UnitRange{Int})
    @polyvar q[1:n]
    mono = monomials(q, degrees)
    ∇x_mono = differentiate(mono, q)
    m = length(mono)
    θ = glorot_uniform(Int(m*(m+1)/2))
    SOSPoly{typeof(q), typeof(mono), typeof(∇x_mono), typeof(θ)}(n, last(degrees), q, mono, ∇x_mono, θ)
end
SOSPoly(n::Int, degree::Int) = SOSPoly(n, 0:degree)

function (P::SOSPoly)(x, θ=P.θ) 
    L = vec2tril(θ)
    X = P.mono
    v = L'*X
    sos = dot(v, v)(P.vars=>x)
end

function coeff_matrix(P::SOSPoly, θ)
    return vec2tril(θ)
end

monsub(P::SOSPoly, x) = [v(P.vars=>x) for v in P.mono]
function ∂mon∂q(P::SOSPoly, x)
    [d(P.vars=>x) for d in P.∇x_mono]
end

function gradient(P::SOSPoly, x, θ=P.θ)
    L = coeff_matrix(P, θ)
    ∂X = ∂mon∂q(P,x)
    expr = 2 * ∂X' * (L*L') * monsub(P,x)
    transpose(expr)
    # reduce(hcat, e for e in expr)
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

nmons(n,d) = binomial(n+d-1,d)

###############################################################################

struct IWPSOSPoly{VA,BA,CO} <: AbstractSOSPoly
    dim::Int
    degree::Int
    vars::VA
    mono::BA
    θ::CO
end

function IWPSOSPoly()
    n = 4
    degrees = 1:1
    @polyvar q[1:n]
    mono = monomials(q, degrees)
    m = length(mono)
    θ = glorot_uniform(Int(m*(m+1)/2))
    θ[[2,4,7,8,9]] .= zero(eltype(θ))
    IWPSOSPoly{typeof(q), typeof(mono), typeof(θ)}(n, last(degrees), q, mono, θ)
end

function (P::IWPSOSPoly)(x, θ=P.θ) 
    L = coeff_matrix(P,θ)
    X = P.mono
    v = L'*X
    sos = dot(v, v)(P.vars=>x)
end

function coeff_matrix(P::IWPSOSPoly, θ)
    L = vec2tril(θ)
    L[2:4,1] .= zero(eltype(L))
    L[end,1:3] .= zero(eltype(L))
    L[end,end] = L[1,1]
    return L
end

monsub(P::IWPSOSPoly, x) = [v(P.vars=>x) for v in P.mono]

function ∂mon∂q(P::IWPSOSPoly, x)
    ∂X = transpose(differentiate(P.mono, P.vars))
    [d(P.vars=>x) for d in ∂X]
end

function gradient(P::IWPSOSPoly, x, θ=P.θ)
    L = coeff_matrix(P, θ)
    ∂X = ∂mon∂q(P,x)
    expr = 2 * ∂X * (L*L') * monsub(P,x)
    transpose(expr)
    # reduce(hcat, e for e in expr)
end


###############################################################################

struct SymmetricSOSPoly{VA, BA, CO} <: AbstractSOSPoly
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
