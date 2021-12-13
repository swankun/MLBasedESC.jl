export SOSPoly, fsubs, jacobian

abstract type AbstractSOSPoly <: Function end

struct SOSPoly{N,M,D,LT,X,MON,JMON} <: AbstractSOSPoly where {N<:Integer,D<:Integer,LT<:Union{AbstractMatrix,Function}}
    vars::X
    mon::MON
    dmon::JMON
    initial_params::LT
end
function SOSPoly(n::Int,d::UnitRange{Int};initL=Flux.glorot_uniform)
    @polyvar x[1:n]
    m = monomials(x,d)
    dm = differentiate(m,x)
    M = length(m)
    L() = vec(initL(M,M))
    SOSPoly{n,M,last(d),typeof(L),typeof(x),typeof(m),typeof(dm)}(x,m,dm,L)
end
SOSPoly(n::Int,d::Int;initL=Flux.glorot_uniform) = SOSPoly(n,0:d;initL=initL)

fsubs(S::SOSPoly,v,x) = map(m->m(S.vars=>x), v)

function (S::SOSPoly{N,M})(x, p) where {N,M}
    xbar = fsubs(S, S.mon, x)
    L = LowerTriangular(reshape(p,M,M))
    v = L' * xbar
    dot(v,v)
end

function jacobian(S::SOSPoly{N,M}, x, p) where {N,M}
    ∂x = fsubs(S, S.dmon, x)
    xbar = fsubs(S, S.mon, x)
    L = LowerTriangular(reshape(p,M,M))
    (2 * ∂x' * (L*L') * xbar)'
end

DiffEqFlux.initial_params(S::SOSPoly) = S.initial_params()
DiffEqFlux.paramlength(::SOSPoly{N,M}) where {N,M} = M*M 

const SOSPolyLayers = Tuple{<:Function,S} where {S<:SOSPoly}
function jacobian(m::FastChain{T}, x, p) where {T<:SOSPolyLayers}
    xbar = first(m.layers)(x, p)
    jacobian(last(m.layers), xbar, p) * jacobian(first(m.layers), x)
end
