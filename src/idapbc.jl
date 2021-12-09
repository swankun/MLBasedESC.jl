export IDAPBCProblem, PDELossKinetic, PDELossPotential
export kineticpde, potentialpde
export trainable, unstack, paramstack
export hamiltonian, hamiltoniand, interconnection, controller

const UFA = Union{T,C} where {T<:Function, C<:Chain}
const J2Chain = NTuple{N,<:Chain} where {N}
const J2FChain = NTuple{N,<:FastChain} where {N}

struct IDAPBCProblem{TJ2,M,MD,V,VD,GT,GP}
    N::Int
    M⁻¹::M
    Md⁻¹::MD
    V::V
    Vd::VD
    J2::TJ2
    G::GT
    G⊥::GP
    function IDAPBCProblem(N,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥)
        if isa(M⁻¹,Matrix) && isa(Md⁻¹,Matrix) && !isnothing(J2)
            error("Constant M and Md requires J2=nothing.")
        end
        if isa(M⁻¹,Function) && isa(Md⁻¹,Matrix) 
            error("M is a function, so Md cannot be constant.")
        end
        for f in (isnothing(J2) ? (Md⁻¹, Vd) : (Md⁻¹, Vd, J2...))
            if isa(f, Chain)
                valid = (
                    isa(f.layers, DenseLayers), 
                    isa(front(f.layers), DenseLayers), 
                    isa(front(tail(f.layers)), DenseLayers)
                )
                if !any(valid)
                    error("Unsupported type $(f)")
                end
            elseif isa(f, FastChain)
                valid = (
                    isa(f.layers, FastDenseLayers), 
                    isa(front(f.layers), FastDenseLayers), 
                    isa(front(tail(f.layers)), FastDenseLayers)
                )
                if !any(valid)
                    error("Unsupported type $(f)")
                end
            end
        end
        new{typeof(J2),typeof(M⁻¹),typeof(Md⁻¹),typeof(V),typeof(Vd),typeof(G),typeof(G⊥)}(
            N,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥
        )
    end
end

function IDAPBCProblem(N,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    IDAPBCProblem(N,M⁻¹,Md⁻¹,V,Vd,nothing,G,G⊥)
end

function Base.show(io::IO, p::IDAPBCProblem)
    println(io, "IDAPBCProblem [q ∈ ℜ^$(p.N)]");
    println(io, "M⁻¹  => $(typeof(p.M⁻¹).name.name)");
    println(io, "Md⁻¹ => $(typeof(p.Md⁻¹).name.name)");
    println(io, "Vd   => $(typeof(p.Vd).name.name)");
    println(io, "J2   => $(typeof(p.J2).name.name)");
end

hasfreevariables(::IDAPBCProblem{J2}) where {J2} = J2===Nothing

function kineticpde(M⁻¹, Md⁻¹, ∇M⁻¹, ∇Md⁻¹, G⊥, J2=0) 
    return G⊥ * (∇M⁻¹' - (Md⁻¹\M⁻¹)*∇Md⁻¹' + J2*Md⁻¹)
end

function potentialpde(M⁻¹, Md⁻¹, ∇V, ∇Vd, G⊥)
    return first(G⊥ * (∇V - (Md⁻¹\M⁻¹)*∇Vd))
end

trainable(p::IDAPBCProblem{Nothing,M,MD,V,VD}) where
    {M,V,MD<:UFA,VD<:UFA} = (p.Md⁻¹, p.Vd)
trainable(p::IDAPBCProblem{Nothing,M,MD,V,VD}) where
    {M,V,MD<:Matrix,VD<:UFA} = (p.Vd,)
trainable(p::IDAPBCProblem{J,M,MD,V,VD}) where
    {M,V,MD<:UFA,VD<:UFA,N,J<:NTuple{N,<:UFA}} = (p.Md⁻¹, p.Vd, p.J2...)
function unstack(ts::NTuple{N,Function}, ps) where N
    n = DiffEqFlux.paramlength(first(ts))
    res = ps[1:n]
    return (res, unstack(tail(ts), ps[n+1:end])...)
end
unstack(::Tuple{}, ::Any) = ()
unstack(p::IDAPBCProblem, ps) = unstack(trainable(p), ps)
unstack(::IDAPBCProblem, ::Nothing) = nothing
paramstack(p::IDAPBCProblem) = vcat(DiffEqFlux.initial_params.(trainable(p))...)

_M⁻¹(P::IDAPBCProblem{J2,M}, ::Any) where {J2,M<:Matrix} = P.M⁻¹
_M⁻¹(P::IDAPBCProblem{J2,M}, q) where {J2,M<:Function} = P.M⁻¹(q)
_Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q, ::Any=nothing) where {J2,M,MD<:Matrix} = P.Md⁻¹
_Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q) where {J2,M,MD<:Chain} = P.Md⁻¹(q)
function _Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q, ps) where {J2,M,MD<:Function} 
    θMd = first(unstack(P, ps))
    P.Md⁻¹(q, θMd)
end
_∇M⁻¹(P::IDAPBCProblem{J2,M}, q) where {J2,M<:Matrix} = collect(0P.M⁻¹ for _=1:P.N)
_∇M⁻¹(P::IDAPBCProblem{J2,M}, q) where {J2,M<:Function} = jacobian(P.M⁻¹,q)
_∇Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q, ::Any=nothing) where {J2,M,MD<:Matrix} = collect(0P.M⁻¹ for _=1:P.N)
_∇Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q) where {J2,M,MD<:Chain} = jacobian(P.Md⁻¹,q)
function _∇Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q, ps) where {J2,M,MD<:Function} 
    θMd = first(unstack(P, ps))
    jacobian(P.Md⁻¹, q, θMd)
end
function _∇Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q) where {J2,M,MD,V,VD<:Chain}
    if isequal(length(last(P.Vd.layers).bias), P.N)
        return jacobian(P.Vd, q)[:]
    else
        return P.Vd(q)
    end
end
function _∇Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q, ps) where {J2,M,V,MD<:Matrix,VD<:Function}
    if VD<:FastChain && isequal(last(P.Vd.layers).out, P.N)
        return P.Vd(q, ps)
    else
        return jacobian(P.Vd, q, ps)[:]
    end
end
function _∇Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q, ps) where {J2,M,V,MD<:Function,VD<:Function}
    _, θVd = unstack(P,ps)
    if VD<:FastChain && isequal(last(P.Vd.layers).out, P.N)
        return P.Vd(q, ps)
    else
        return jacobian(P.Vd, q, θVd)[:]
    end
end

interconnection(P::IDAPBCProblem{Nothing}, x, ::Any=nothing) = 0
function interconnection(P::IDAPBCProblem{<:J2Chain}, x)
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    mapreduce(+, P.J2, p) do Uk, pk
        Uk(q)*pk/2
    end
end
function interconnection(P::IDAPBCProblem{<:J2FChain}, x, ps)
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    θ = last(unstack(P, ps), 2)
    mapreduce(+, P.J2, p, θ) do Uk, pk, θk
        Uk(q,θk)*pk/2
    end
end

function hamiltonian(P::IDAPBCProblem, x)
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    dot(p, _M⁻¹(P, q)*p)/2 + first(P.V(q))
end
function ∇H(P::IDAPBCProblem, x)
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    ∇M⁻¹ = _∇M⁻¹(P, q)
    (mapreduce(*,+,∇M⁻¹,p)'*p)/2 + jacobian(P.V, q)[:]
end

function hamiltoniand(P::IDAPBCProblem, x)
    !isone(length(last(P.Vd.layers).bias)) && return NaN
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    dot(p, _Md⁻¹(P, q)*p)/2 + first(P.Vd(q))
end
function ∇Hd(P::IDAPBCProblem, x)
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    ∇Md⁻¹ = _∇Md⁻¹(P, q)
    (mapreduce(*,+,∇Md⁻¹,p)'*p)/2 + _∇Vd(P, q)
end
function ∇Hd(P::IDAPBCProblem, x, ps)
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    ∇Md⁻¹ = _∇Md⁻¹(P, q, ps)
    (mapreduce(*,+,∇Md⁻¹,p)'*p)/2 + _∇Vd(P, q, ps)
end

function controller(P::IDAPBCProblem, x; kv=1)
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    M⁻¹ = _M⁻¹(P,q)
    Md⁻¹ = _Md⁻¹(P,q)
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    u_es = G⁻¹ * ( ∇H(P,x) - MdM⁻¹*∇Hd(P,x) + interconnection(P,x)*Md⁻¹*p )
    u_di = -kv * dot(P.G, Md⁻¹*p)
    return u_es + u_di
end
function controller(P::IDAPBCProblem, x, ps; kv=1)
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    M⁻¹ = _M⁻¹(P,q)
    Md⁻¹ = _Md⁻¹(P,q,ps)
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    u_es = G⁻¹ * ( ∇H(P,x) - MdM⁻¹*∇Hd(P,x,ps) + interconnection(P,x,ps)*Md⁻¹*p )
    u_di = -kv * dot(P.G, Md⁻¹*p)
    return u_es + u_di
end

abstract type IDAPBCLoss end

struct PDELossKinetic{hasJ2,P} <: IDAPBCLoss
    prob::P
    function PDELossKinetic(prob::P) where {J2,P<:IDAPBCProblem{J2}}
        new{J2!==Nothing,P}(prob)
    end
end
function (L::PDELossKinetic{false})(q)
    p = L.prob
    M⁻¹ = _M⁻¹(p, q)
    ∇M⁻¹ = _∇M⁻¹(p, q)
    Md⁻¹ = _Md⁻¹(p, q)
    ∇Md⁻¹ = _∇Md⁻¹(p, q)
    r = map(∇M⁻¹, ∇Md⁻¹) do _1, _2
        sum(abs, kineticpde(M⁻¹, Md⁻¹, _1, _2, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{true})(q)
    p = L.prob
    M⁻¹ = _M⁻¹(p, q)
    ∇M⁻¹ = _∇M⁻¹(p, q)
    Md⁻¹ = _Md⁻¹(p, q)
    ∇Md⁻¹ = _∇Md⁻¹(p, q)
    r = map(∇M⁻¹, ∇Md⁻¹, p.J2) do _1, _2, Uk
        sum(abs, kineticpde(M⁻¹, Md⁻¹, _1, _2, p.G⊥, Uk(q)))
    end
    sum(r)
end
function (L::PDELossKinetic{false})(q,ps)
    p = L.prob
    M⁻¹ = _M⁻¹(p, q)
    ∇M⁻¹ = _∇M⁻¹(p, q)
    Md⁻¹ = _Md⁻¹(p, q, ps)
    ∇Md⁻¹ = _∇Md⁻¹(p, q, ps)
    r = map(∇M⁻¹, ∇Md⁻¹) do _1, _2
        sum(abs, kineticpde(M⁻¹, Md⁻¹, _1, _2, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{true})(q,ps)
    p = L.prob
    M⁻¹ = _M⁻¹(p, q)
    ∇M⁻¹ = _∇M⁻¹(p, q)
    Md⁻¹ = _Md⁻¹(p, q, ps)
    ∇Md⁻¹ = _∇Md⁻¹(p, q, ps)
    J2ps = last(unstack(p, ps), 2)
    r = map(∇M⁻¹, ∇Md⁻¹, p.J2, J2ps) do _1, _2, Uk, Ukps
        sum(abs, kineticpde(M⁻¹, Md⁻¹, _1, _2, p.G⊥, Uk(q,Ukps)))
    end
    sum(r)
end


struct PDELossPotential{P} <: IDAPBCLoss
    prob::P
end
function (L::PDELossPotential)(q) 
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = _∇Vd(p, q)
    M⁻¹ = _M⁻¹(p, q)
    Md⁻¹ = _Md⁻¹(p, q)
    potentialpde(M⁻¹, Md⁻¹, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential)(q,ps)
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = _∇Vd(p, q, ps)
    M⁻¹ = _M⁻¹(p, q)
    Md⁻¹ = _Md⁻¹(p, q, ps)
    potentialpde(M⁻¹, Md⁻¹, ∇V, ∇Vd, p.G⊥)
end

struct PotentialHessianSymLoss{P} <: IDAPBCLoss
    prob::P 
    function PotentialHessianSymLoss(p::IDAPBCProblem{J,M,MD,V,VD}) where 
        {J,M,MD,V,VD<:Chain}
        if isequal(length(last(p.Vd.layers).bias), p.N)
            new{typeof(p)}(p)
        else
            error("Not applicable for this type of IDAPBCProblem")
        end
    end
    function PotentialHessianSymLoss(p::IDAPBCProblem{J,M,MD,V,VD}) where 
        {J,M,MD,V,VD<:FastChain}
        if isequal(last(p.Vd.layers).out, p.N)
            new{typeof(p)}(p)
        else
            error("Not applicable for this type of IDAPBCProblem")
        end
    end
end
function (L::PotentialHessianSymLoss)(q)
    J = jacobian(L.prob.Vd, q)
    mapreduce(abs, +, J - J')
end
function (L::PotentialHessianSymLoss{P})(q, ps) where {J,M,MD<:Matrix,P<:IDAPBCProblem{J,M,MD}}
    J = jacobian(L.prob.Vd, q, ps)
    mapreduce(abs, +, J - J')
end
function (L::PotentialHessianSymLoss{P})(q, ps) where {J,M,MD<:Function,P<:IDAPBCProblem{J,M,MD}}
    _, θVd = unstack(P,ps)
    J = jacobian(L.prob.Vd, q, θVd)
    mapreduce(abs, +, J - J')
end
