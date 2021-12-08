export IDAPBCProblem, PDELossKinetic, PDELossPotential
export kineticpde, potentialpde
export trainable, unstack, paramstack

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

function kineticpde(M⁻¹::M, Md::M, Md⁻¹::M, ∇M⁻¹::M, ∇Md⁻¹::M, G⊥, J2=0) where 
    {M<:Matrix}
    return G⊥ * (∇M⁻¹' - Md*M⁻¹*∇Md⁻¹' + J2*Md⁻¹)
end

function potentialpde(M⁻¹::M, Md::M, ∇V::V, ∇Vd::V, G⊥) where 
    {M<:Matrix, V<:Vector} 
    return first(G⊥ * (∇V - Md*M⁻¹*∇Vd))
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


abstract type IDAPBCLoss end

struct PDELossKinetic{P} <: IDAPBCLoss
    prob::P
end
function (L::PDELossKinetic{P})(q,::Any=nothing) where 
    {M<:Matrix,MD<:Matrix,P<:IDAPBCProblem{Nothing,M,MD}} # M⁻¹::Const, Md⁻¹::Const, J2::No
    return 0.0
end
function (L::PDELossKinetic{P})(q) where 
    {M<:Matrix,MD<:Chain,P<:IDAPBCProblem{Nothing,M,MD}} # M⁻¹::Const, Md⁻¹::Chain, J2::No
    p = L.prob
    Md⁻¹ = p.Md⁻¹(q)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q)
    r = map(∇Md⁻¹) do _1
        sum(abs, kineticpde(p.M⁻¹, Md, Md⁻¹, 0*p.M⁻¹, _1, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{P})(q) where 
    {J,M<:Matrix,MD<:Chain,P<:IDAPBCProblem{J,M,MD}} # M⁻¹::Const, Md⁻¹::Chain, J2::NTuple{Chain}
    p = L.prob
    Md⁻¹ = p.Md⁻¹(q)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q)
    r = map(∇Md⁻¹, p.J2) do _1,Uk
        sum(abs, kineticpde(p.M⁻¹, Md, Md⁻¹, 0*p.M⁻¹, _1, p.G⊥, Uk(q)))
    end
    sum(r)
end
function (L::PDELossKinetic{P})(q,ps) where 
    {M<:Matrix,MD<:Function,P<:IDAPBCProblem{Nothing,M,MD}} # M⁻¹::Const, Md⁻¹::Function, J2::No
    p = L.prob
    θMd⁻¹ = first(unstack(p,ps))
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q,θMd⁻¹)
    r = map(∇Md⁻¹) do _1
        sum(abs, kineticpde(p.M⁻¹, Md, Md⁻¹, 0*p.M⁻¹, _1, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{P})(q,ps) where 
    {J,M<:Matrix,MD<:Function,P<:IDAPBCProblem{J,M,MD}} # M⁻¹::Const, Md⁻¹::Function, J2::NTuple{Function}
    p = L.prob
    θ = unstack(p,ps)
    θMd⁻¹ = first(θ)
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q,θMd⁻¹)
    r = map(∇Md⁻¹, p.J2, tail(θ[2:end])) do _1,Uk,Ukps
        sum(abs, kineticpde(p.M⁻¹, Md, Md⁻¹, 0*p.M⁻¹, _1, p.G⊥, Uk(q,Ukps)))
    end
    sum(r)
end
function (L::PDELossKinetic{P})(q) where 
    {M<:Function,MD<:Chain,P<:IDAPBCProblem{Nothing,M,MD}} # M⁻¹::Function, Md⁻¹::Chain, J2::No
    p = L.prob
    M⁻¹ = p.M⁻¹(q)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    Md⁻¹ = p.Md⁻¹(q)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q)
    r = map(∇M⁻¹, ∇Md⁻¹) do _1, _2
        sum(abs, kineticpde(M⁻¹, Md, Md⁻¹, _1, _2, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{P})(q) where 
    {J,M<:Function,MD<:Chain,P<:IDAPBCProblem{J,M,MD}} # M⁻¹::Function, Md⁻¹::Chain, J2::NTuple{Chain}
    p = L.prob
    M⁻¹ = p.M⁻¹(q)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    Md⁻¹ = p.Md⁻¹(q)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q)
    r = map(∇M⁻¹, ∇Md⁻¹, p.J2) do _1, _2, Uk
        sum(abs, kineticpde(M⁻¹, Md, Md⁻¹, _1, _2, p.G⊥, Uk(q)))
    end
    sum(r)
end
function (L::PDELossKinetic{P})(q,ps) where 
    {M<:Function,MD<:Function,P<:IDAPBCProblem{Nothing,M,MD}} # M⁻¹::Function, Md⁻¹::Function, J2::No
    p = L.prob
    M⁻¹ = p.M⁻¹(q)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    θMd⁻¹ = first(unstack(p,ps))
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q,θMd⁻¹)
    r = map(∇M⁻¹, ∇Md⁻¹) do _1, _2
        sum(abs, kineticpde(M⁻¹, Md, Md⁻¹, _1, _2, p.G⊥))
    end
    sum(r)
end
function (L::PDELossKinetic{P})(q,ps) where 
    {J,M<:Function,MD<:Function,P<:IDAPBCProblem{J,M,MD}} # M⁻¹::Function, Md⁻¹::Function, J2::NTuple{Function}
    p = L.prob
    M⁻¹ = p.M⁻¹(q)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    θ = unstack(p,ps)
    θMd⁻¹ = first(θ)
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹,q,θMd⁻¹)
    r = map(∇M⁻¹, ∇Md⁻¹, p.J2, tail(θ[2:end])) do _1, _2, Uk, Ukps
        sum(abs, kineticpde(M⁻¹, Md, Md⁻¹, _1, _2, p.G⊥, Uk(q,Ukps)))
    end
    sum(r)
end


struct PDELossPotential{P} <: IDAPBCLoss
    prob::P
end
function (L::PDELossPotential{P})(q) where 
    {J,M<:Matrix,MD<:Matrix,V,VD<:Chain,P<:IDAPBCProblem{J,M,MD,V,VD}}
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q)[:]
    potentialpde(p.M⁻¹, inv(p.Md⁻¹), ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{P})(q,ps) where 
    {J,M<:Matrix,MD<:Matrix,V,VD<:Function,P<:IDAPBCProblem{J,M,MD,V,VD}}
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q, ps)[:]
    potentialpde(p.M⁻¹, inv(p.Md⁻¹), ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{P})(q) where 
    {J,M<:Matrix,MD<:Chain,V,VD<:Chain,P<:IDAPBCProblem{J,M,MD,V,VD}}
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q)[:]
    Md⁻¹ = p.Md⁻¹(q)
    Md = inv(Md⁻¹)
    potentialpde(p.M⁻¹, Md, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{P})(q,ps) where 
    {J,M<:Matrix,MD<:Function,V,VD<:Function,P<:IDAPBCProblem{J,M,MD,V,VD}}
    p = L.prob
    θMd⁻¹, θVd = unstack(p,ps)
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q, θVd)[:]
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    Md = inv(Md⁻¹)
    potentialpde(p.M⁻¹, Md, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{P})(q) where 
    {J,M<:Function,MD<:Chain,V,VD<:Chain,P<:IDAPBCProblem{J,M,MD,V,VD}}
    p = L.prob
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q)[:]
    M⁻¹ = p.M⁻¹(q)
    Md⁻¹ = p.Md⁻¹(q)
    Md = inv(Md⁻¹)
    potentialpde(M⁻¹, Md, ∇V, ∇Vd, p.G⊥)
end
function (L::PDELossPotential{P})(q,ps) where 
    {J,M<:Function,MD<:Function,V,VD<:Function,P<:IDAPBCProblem{J,M,MD,V,VD}}
    p = L.prob
    θMd⁻¹, θVd = unstack(p,ps)
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q, θVd)[:]
    M⁻¹ = p.M⁻¹(q)
    Md⁻¹ = p.Md⁻¹(q,θMd⁻¹)
    Md = inv(Md⁻¹)
    potentialpde(M⁻¹, Md, ∇V, ∇Vd, p.G⊥)
end
