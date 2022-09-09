export IDAPBCProblem, PDELossKinetic, PDELossPotential
export kineticpde, potentialpde
export trainable, unstack, paramstack
export hamiltonian, hamiltoniand, interconnection, controller
export gradient

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
        if isa(M⁻¹,Matrix) && (isa(Md⁻¹,Matrix) || isa(Md⁻¹,PSDMatrix)) && !isnothing(J2)
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
function Base.getindex(P::IDAPBCProblem, sym::Symbol) 
    if sym === :Md
        if isimplicit(P)
            return (q,::Any=missing)->inv(_Md⁻¹(P,q))
        else
            return (q,ps)->inv(_Md⁻¹(P,q,ps))
        end
    elseif sym === :Mdinv
        if isimplicit(P)
            return (q,::Any=missing)->_Md⁻¹(P,q)
        else
            return (q,ps)->_Md⁻¹(P,q,ps)
        end
    elseif sym === :Vd
        if isimplicit(P)
            return (q,::Any=missing)->_Vd(P,q)
        else
            return (q,ps)->_Vd(P,q,ps)
        end
    elseif sym === :Hd
        if isimplicit(P)
            function (x,::Any=missing)
                q, qdot = x[1:P.N], x[P.N+1:end]
                momentum = _M⁻¹(P,q) \ qdot
                hamiltoniand(P, [q; momentum])
            end
        else
            function (x,ps)
                q, qdot = x[1:P.N], x[P.N+1:end]
                momentum = _M⁻¹(P,q) \ qdot
                hamiltoniand(P, [q; momentum], ps)
            end
        end
    elseif sym === :J2
        isnothing(P.J2) && return nothing
        if isimplicit(P)
            return (q,::Any=missing)->interconnection(P,q)
        else
            return (q,ps)->interconnection(P,q,ps)
        end
    end
end

hasfreevariables(::IDAPBCProblem{J2}) where {J2} = J2===Nothing
isimplicit(P::IDAPBCProblem) = any(f->isa(f,Chain), trainable(P))

function kineticpde(M⁻¹, Md⁻¹, ∇M⁻¹, ∇Md⁻¹, G⊥, J2=0) 
    return G⊥ * (∇M⁻¹' - (Md⁻¹\M⁻¹)*∇Md⁻¹' + J2*Md⁻¹)
end

function potentialpde(M⁻¹, Md⁻¹, ∇V, ∇Vd, G⊥)
    return G⊥ * (∇V - (Md⁻¹\M⁻¹)*∇Vd)
end

trainable(p::IDAPBCProblem{Nothing,M,MD,V,VD}) where
    {M,V,MD<:UFA,VD<:UFA} = (p.Md⁻¹, p.Vd)
trainable(p::IDAPBCProblem{Nothing,M,MD,V,VD}) where
    {M,V,MD<:AbstractVecOrMat,VD<:UFA} = (p.Vd,)
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

_M⁻¹(P::IDAPBCProblem{J2,M}, ::Any) where {J2,M<:AbstractVecOrMat} = P.M⁻¹
_M⁻¹(P::IDAPBCProblem{J2,M}, q) where {J2,M<:Function} = P.M⁻¹(q)
_Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q, ::Any=nothing) where {J2,M,MD<:AbstractVecOrMat} = P.Md⁻¹
_Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q) where {J2,M,MD<:Chain} = P.Md⁻¹(q)
function _Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q) where {J2,M,MD<:Function} 
    P.Md⁻¹(q)
end
function _Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q, ps) where {J2,M,MD<:Function} 
    θMd = first(unstack(P, ps))
    P.Md⁻¹(q, θMd)
end
_∇M⁻¹(P::IDAPBCProblem{J2,M}, q) where {J2,M<:AbstractVecOrMat} = collect(0P.M⁻¹ for _=1:P.N)
_∇M⁻¹(P::IDAPBCProblem{J2,M}, q) where {J2,M<:Function} = jacobian(P.M⁻¹,q)
_∇Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q, ::Any=nothing) where {J2,M,MD<:AbstractVecOrMat} = collect(0P.M⁻¹ for _=1:P.N)
_∇Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q) where {J2,M,MD<:Chain} = jacobian(P.Md⁻¹,q)
function _∇Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q, ps) where {J2,M,MD<:Function} 
    θMd = first(unstack(P, ps))
    jacobian(P.Md⁻¹, q, θMd)
end
function _∇Md⁻¹(P::IDAPBCProblem{J2,M,MD}, q) where {J2,M,MD<:Function} 
    jacobian(P.Md⁻¹, q)
end
_Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q) where {J2,M,MD,V,VD<:Chain} = P.Vd(q)
function _Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q, ps) where {J2,M,MD<:AbstractVecOrMat,V,VD<:Function} 
    P.Vd(q, ps)
end
function _Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q, ps) where {J2,M,MD<:Function,V,VD<:Function} 
    _, θVd = unstack(P, ps)
    P.Vd(q, θVd)
end
function _∇Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q) where {J2,M,MD,V,VD<:Chain}
    if isa(last(P.Vd.layers), Flux.Dense) && isequal(size(last(P.Vd.layers).weight,1), P.N)
        return P.Vd(q)
    else
        return jacobian(P.Vd, q)[:]
    end
end
function _∇Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q, ps) where {J2,M,V,MD<:AbstractVecOrMat,VD<:FastChain}
    if isa(last(P.Vd.layers), DiffEqFlux.FastDense) && isequal(last(P.Vd.layers).out, P.N)
        return P.Vd(q, ps)
    else
        return jacobian(P.Vd, q, ps)[:]
    end
end
function _∇Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q, ps) where {J2,M,V,MD<:Function,VD<:FastChain}
    _, θVd = unstack(P,ps)
    if isa(last(P.Vd.layers), DiffEqFlux.FastDense) && isequal(last(P.Vd.layers).out, P.N)
        return P.Vd(q, θVd)
    else
        return jacobian(P.Vd, q, θVd)[:]
    end
end
function _∇Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q, ps) where {J2,M,V,MD<:AbstractVecOrMat,VD<:Function}
    return jacobian(P.Vd, q, ps)[:]
end
function _∇Vd(P::IDAPBCProblem{J2,M,MD,V,VD}, q, ps) where {J2,M,V,MD<:Function,VD<:Function}
    _, θVd = unstack(P,ps)
    return jacobian(P.Vd, q, θVd)[:]
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

function hamiltonian(P::IDAPBCProblem, qp)
    N = P.N
    q, p = qp[1:N], qp[N+1:2N]
    dot(p, _M⁻¹(P, q)*p)/2 + first(P.V(q))
end
function ∇H(P::IDAPBCProblem, qp)
    N = P.N
    q, p = qp[1:N], qp[N+1:2N]
    ∇M⁻¹ = _∇M⁻¹(P, q)
    (mapreduce(*,+,∇M⁻¹,p)'*p)/2 + jacobian(P.V, q)[:]
end

function hamiltoniand(P::IDAPBCProblem, qp)
    N = P.N
    q, p = qp[1:N], qp[N+1:2N]
    dot(p, _Md⁻¹(P, q)*p)/2 + first(_Vd(P,q))
end
function hamiltoniand(P::IDAPBCProblem, qp, ps)
    N = P.N
    q, p = qp[1:N], qp[N+1:2N]
    dot(p, _Md⁻¹(P, q, ps)*p)/2 + first(_Vd(P,q,ps))
end
function ∇Hd(P::IDAPBCProblem, qp)
    N = P.N
    q, p = qp[1:N], qp[N+1:2N]
    ∇Md⁻¹ = _∇Md⁻¹(P, q)
    (mapreduce(*,+,∇Md⁻¹,p)'*p)/2 + _∇Vd(P, q)
end
function ∇Hd(P::IDAPBCProblem, qp, ps)
    N = P.N
    q, p = qp[1:N], qp[N+1:2N]
    ∇Md⁻¹ = _∇Md⁻¹(P, q, ps)
    (mapreduce(*,+,∇Md⁻¹,p)'*p)/2 + _∇Vd(P, q, ps)
end

function controller(P::IDAPBCProblem, x; kv=1)
    N = P.N
    q, qdot = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    M⁻¹ = _M⁻¹(P,q)
    Md⁻¹ = _Md⁻¹(P,q)
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    momentum = M⁻¹ \ qdot
    qp = [q; momentum]
    u_es = G⁻¹ * ( ∇H(P,qp) - MdM⁻¹*∇Hd(P,qp) + interconnection(P,qp)*Md⁻¹*momentum )
    u_di = -kv * dot(P.G, Md⁻¹*momentum)
    return u_es + u_di
end
function controller(P::IDAPBCProblem, x, ps; kv=1)
    N = P.N
    q, qdot = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    M⁻¹ = _M⁻¹(P,q)
    Md⁻¹ = _Md⁻¹(P,q,ps)
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    momentum = M⁻¹ \ qdot
    qp = [q; momentum]
    u_es = G⁻¹ * ( ∇H(P,qp) - MdM⁻¹*∇Hd(P,qp,ps) + interconnection(P,qp,ps)*Md⁻¹*momentum )
    u_di = -kv * dot(P.G, Md⁻¹*momentum)
    return u_es + u_di
end

abstract type IDAPBCLoss{P} end

function gradient(l::IDAPBCLoss{P}, q, ps::AbstractVector) where 
    {J2,M,MD,V,VD<:Function,P<:IDAPBCProblem{J2,M,MD,V,VD}}
    ReverseDiff.gradient(_2->l(q,_2), ps)
end
function gradient(l::IDAPBCLoss{P}, q, ps::Flux.Params) where 
    {J2,M,MD,V,VD<:Chain,P<:IDAPBCProblem{J2,M,MD,V,VD}}
    Flux.gradient(()->l(q), ps)
end

struct PDELossKinetic{hasJ2,P} <: IDAPBCLoss{P}
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


struct PDELossPotential{P} <: IDAPBCLoss{P}
    prob::P
end
function (L::PDELossPotential)(q) 
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = _∇Vd(p, q)
    M⁻¹ = _M⁻¹(p, q)
    Md⁻¹ = _Md⁻¹(p, q)
    norm(potentialpde(M⁻¹, Md⁻¹, ∇V, ∇Vd, p.G⊥))
end
function (L::PDELossPotential)(q,ps)
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = _∇Vd(p, q, ps)
    M⁻¹ = _M⁻¹(p, q)
    Md⁻¹ = _Md⁻¹(p, q, ps)
    norm(potentialpde(M⁻¹, Md⁻¹, ∇V, ∇Vd, p.G⊥))
end

struct PotentialHessianSymLoss{P} <: IDAPBCLoss{P}
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
function (L::PotentialHessianSymLoss{P})(q, ps) where {J2,M,MD<:AbstractVecOrMat,P<:IDAPBCProblem{J2,M,MD}}
    J = jacobian(L.prob.Vd, q, ps)
    mapreduce(abs, +, J - J')
end
function (L::PotentialHessianSymLoss{P})(q, ps) where {J2,M,MD<:Function,P<:IDAPBCProblem{J2,M,MD}}
    _, θVd = unstack(P,ps)
    J = jacobian(L.prob.Vd, q, θVd)
    mapreduce(abs, +, J - J')
end
