export IDAPBCProblem, PDELossKinetic, PDELossPotential
export kineticpde, potentialpde
export trainable, unstack, paramstack
export interconnection, controller

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
function unstack(ts::NTuple{N,FastChain}, ps) where N
    n = DiffEqFlux.paramlength(first(ts))
    res = ps[1:n]
    return (res, unstack(tail(ts), ps[n+1:end])...)
end
unstack(ts::Tuple{}, ps) = ()
unstack(p::IDAPBCProblem, ps) = unstack(trainable(p), ps)
unstack(p::IDAPBCProblem, ::Nothing) = nothing
paramstack(p::IDAPBCProblem) = vcat(DiffEqFlux.initial_params.(trainable(p))...)

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

function ∇H(P::IDAPBCProblem{J2,M}, x) where {J2,M<:Matrix}
    q = x[1:P.N]
    jacobian(P.V, q)[:]
end
function ∇H(P::IDAPBCProblem{J2,M}, x) where {J2,M<:Function}
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    JM⁻¹ = jacobian(P.M⁻¹, q)
    (sum(JM⁻¹.*p)'*p)/2 + jacobian(P.V, q)[:]
end

function ∇Hd(P::IDAPBCProblem{J2,M,MD,V,VD}, x) where {J2,M,V,MD<:Matrix,VD<:Chain}
    q = x[1:P.N]
    jacobian(P.Vd, q)[:]
end
function ∇Hd(P::IDAPBCProblem{J2,M,MD,V,VD}, x) where {J2,M,V,MD<:Chain,VD<:Chain}
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    JMd⁻¹ = jacobian(P.Md⁻¹, q)
    (sum(JMd⁻¹.*p)'*p)/2 + jacobian(P.Vd, q)[:]
end
function ∇Hd(P::IDAPBCProblem{J2,M,MD,V,VD}, x, ps) where {J2,M,V,MD<:Function,VD<:Function}
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    θMd, θVd = first(unstack(P,ps), 2)
    JMd⁻¹ = jacobian(P.Md⁻¹, q, θMd)
    (sum(JMd⁻¹.*p)'*p)/2 + jacobian(P.Vd, q, θVd)[:]
end

function controller(P::IDAPBCProblem{J2,M,MD}, x; kv=1) where {J2,M<:Matrix,MD<:Matrix}
    N = P.N
    _, p = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    Md⁻¹, M⁻¹ = P.Md⁻¹, P.M⁻¹
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    u_es = G⁻¹ * ( ∇H(P,x) - MdM⁻¹*∇Hd(P,x) + interconnection(P,x)*Md⁻¹*p )
    u_di = -kv * dot(P.G, Md⁻¹*p)
    return u_es + u_di
end
function controller(P::IDAPBCProblem{J2,M,MD}, x; kv=1) where {J2,M<:Matrix,MD<:Chain}
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    Md⁻¹, M⁻¹ = P.Md⁻¹(q), P.M⁻¹
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    u_es = G⁻¹ * ( ∇H(P,x) - MdM⁻¹*∇Hd(P,x) + interconnection(P,x)*Md⁻¹*p )
    u_di = -kv * dot(P.G, Md⁻¹*p)
    return u_es + u_di
end
function controller(P::IDAPBCProblem{J2,M,MD}, x; kv=1) where {J2,M<:Function,MD<:Chain}
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    Md⁻¹, M⁻¹ = P.Md⁻¹(q), P.M⁻¹(q)
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    u_es = G⁻¹ * ( ∇H(P,x) - MdM⁻¹*∇Hd(P,x) + interconnection(P,x)*Md⁻¹*p )
    u_di = -kv * dot(P.G, Md⁻¹*p)
    return u_es + u_di
end
function controller(P::IDAPBCProblem{J2,M,MD}, x, ps; kv=1) where {J2,M<:Matrix,MD<:Function}
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    θMd = first(unstack(P,ps))
    Md⁻¹, M⁻¹ = P.Md⁻¹(q,θMd), P.M⁻¹
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    u_es = G⁻¹ * ( ∇H(P,x) - MdM⁻¹*∇Hd(P,x,ps) + interconnection(P,x,ps)*Md⁻¹*p )
    u_di = -kv * dot(P.G, Md⁻¹*p)
    return u_es + u_di
end
function controller(P::IDAPBCProblem{J2,M,MD}, x, ps; kv=1) where {J2,M<:Function,MD<:Function}
    N = P.N
    q, p = x[1:N], x[N+1:2N]
    G⁻¹ = (P.G'*P.G)\(P.G')
    θMd = first(unstack(P,ps))
    Md⁻¹, M⁻¹ = P.Md⁻¹(q,θMd), P.M⁻¹(q)
    MdM⁻¹ = Md⁻¹ \ M⁻¹
    u_es = G⁻¹ * ( ∇H(P,x) - MdM⁻¹*∇Hd(P,x,ps) + interconnection(P,x,ps)*Md⁻¹*p )
    u_di = -kv * dot(P.G, Md⁻¹*p)
    return u_es + u_di
end

abstract type IDAPBCLoss end

struct PDELossKinetic{isexplicit,callM,callMD,callJ2,P} <: IDAPBCLoss
    prob::P
    function PDELossKinetic(prob::P) where {J2,M,MD,P<:IDAPBCProblem{J2,M,MD}}
        callJ2 = J2!==Nothing
        callM = !(M<:Matrix)
        callMD = !(MD<:Matrix)
        isexplicit = (MD<:Function)
        new{isexplicit,callM,callMD,callJ2,P}(prob)
    end
end
function (L::PDELossKinetic{E,false,false,false})(q,::Any=nothing) where {E}
    # M⁻¹::Const, Md⁻¹::Const, J2::No
    return 0.0
end
function (L::PDELossKinetic{false,false,true,false})(q)  
    # M⁻¹::Const, Md⁻¹::Chain, J2::No
    p = L.prob
    Md⁻¹ = p.Md⁻¹(q)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q)
    r = map(∇Md⁻¹) do _1
        sum(abs, kineticpde(p.M⁻¹, Md⁻¹, 0*p.M⁻¹, _1, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{false,false,true,true})(q)  
    # M⁻¹::Const, Md⁻¹::Chain, J2::NTuple{Chain}
    p = L.prob
    Md⁻¹ = p.Md⁻¹(q)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q)
    r = map(∇Md⁻¹, p.J2) do _1,Uk
        sum(abs, kineticpde(p.M⁻¹, Md⁻¹, 0*p.M⁻¹, _1, p.G⊥, Uk(q)))
    end
    sum(r)
end
function (L::PDELossKinetic{true,false,true,false})(q,ps)
    # M⁻¹::Const, Md⁻¹::Function, J2::No
    p = L.prob
    θMd⁻¹ = first(unstack(p,ps))
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q,θMd⁻¹)
    r = map(∇Md⁻¹) do _1
        sum(abs, kineticpde(p.M⁻¹, Md⁻¹, 0*p.M⁻¹, _1, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{true,false,true,true})(q,ps)
    # M⁻¹::Const, Md⁻¹::Function, J2::NTuple{Function}
    p = L.prob
    θ = unstack(p,ps)
    θMd⁻¹ = first(θ)
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q,θMd⁻¹)
    r = map(∇Md⁻¹, p.J2, tail(θ[2:end])) do _1,Uk,Ukps
        sum(abs, kineticpde(p.M⁻¹, Md⁻¹, 0*p.M⁻¹, _1, p.G⊥, Uk(q,Ukps)))
    end
    sum(r)
end
function (L::PDELossKinetic{false,true,true,false})(q)
    # M⁻¹::Function, Md⁻¹::Chain, J2::No
    p = L.prob
    M⁻¹ = p.M⁻¹(q)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    Md⁻¹ = p.Md⁻¹(q)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q)
    r = map(∇M⁻¹, ∇Md⁻¹) do _1, _2
        sum(abs, kineticpde(M⁻¹, Md⁻¹, _1, _2, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{false,true,true,true})(q)
    # M⁻¹::Function, Md⁻¹::Chain, J2::NTuple{Chain}
    p = L.prob
    M⁻¹ = p.M⁻¹(q)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    Md⁻¹ = p.Md⁻¹(q)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q)
    r = map(∇M⁻¹, ∇Md⁻¹, p.J2) do _1, _2, Uk
        sum(abs, kineticpde(M⁻¹, Md⁻¹, _1, _2, p.G⊥, Uk(q)))
    end
    sum(r)
end
function (L::PDELossKinetic{true,true,true,false})(q,ps)
    # M⁻¹::Function, Md⁻¹::Function, J2::No
    p = L.prob
    M⁻¹ = p.M⁻¹(q)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    θMd⁻¹ = first(unstack(p,ps))
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q,θMd⁻¹)
    r = map(∇M⁻¹, ∇Md⁻¹) do _1, _2
        sum(abs, kineticpde(M⁻¹, Md⁻¹, _1, _2, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{true,true,true,true})(q,ps)
    # M⁻¹::Function, Md⁻¹::Function, J2::NTuple{Function}
    p = L.prob
    M⁻¹ = p.M⁻¹(q)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    θ = unstack(p,ps)
    θMd⁻¹ = first(θ)
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q,θMd⁻¹)
    r = map(∇M⁻¹, ∇Md⁻¹, p.J2, tail(θ[2:end])) do _1, _2, Uk, Ukps
        sum(abs, kineticpde(M⁻¹, Md⁻¹, _1, _2, p.G⊥, Uk(q,Ukps)))
    end
    sum(r)
end


struct PDELossPotential{isexplicit,callM,callMD,P} <: IDAPBCLoss
    prob::P
    function PDELossPotential(prob::P) where {J2,M,MD,V,VD,P<:IDAPBCProblem{J2,M,MD,V,VD}}
        callM = !(M<:Matrix)
        callMD = !(MD<:Matrix)
        isexplicit = (VD<:Function)
        new{isexplicit,callM,callMD,P}(prob)
    end
end
function (L::PDELossPotential{false,false,false})(q)
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q)[:]
    potentialpde(p.M⁻¹, p.Md⁻¹, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{true,false,false})(q,ps)
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q, ps)[:]
    potentialpde(p.M⁻¹, p.Md⁻¹, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{false,false,true})(q)
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q)[:]
    Md⁻¹ = p.Md⁻¹(q)
    potentialpde(p.M⁻¹, Md⁻¹, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{true,false,true})(q,ps)
    p = L.prob
    θMd⁻¹, θVd = unstack(p,ps)
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q, θVd)[:]
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    potentialpde(p.M⁻¹, Md⁻¹, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{false,true,true})(q) 
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q)[:]
    M⁻¹ = p.M⁻¹(q)
    Md⁻¹ = p.Md⁻¹(q)
    potentialpde(M⁻¹, Md⁻¹, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{true,true,true})(q,ps)
    p = L.prob
    θMd⁻¹, θVd = unstack(p,ps)
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q, θVd)[:]
    M⁻¹ = p.M⁻¹(q)
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    potentialpde(M⁻¹, Md⁻¹, ∇V, ∇Vd, p.G⊥)
end
