export NeuralPBC, unstack, controller
export SetDistanceLoss, gradient
export CustomDagger

struct NeuralPBC{N,HD} 
    Hd::HD
    function NeuralPBC(N::Int, Hd)
        new{N,typeof(Hd)}(Hd)
    end
end

(P::NeuralPBC)(x, ps) = controller(P, x, ps)

function DiffEqFlux.initial_params(P::NeuralPBC{N,HD}) where {N,HD<:FastChain}
    gains = Flux.glorot_uniform(N)
    vcat(DiffEqFlux.initial_params(P.Hd), gains)
end

function unstack(P::NeuralPBC{N}, ps::AbstractVector) where {N}
    return (ps[1:end-N], ps[end-N+1:end])
end

function controller(P::NeuralPBC{N}, x, ps) where {N}
    Hdθ, gains = unstack(P,ps)
    dot(gains, jacobian(P.Hd, x, Hdθ))
end



abstract type NeuralPBCLoss end

function gradient(l::NeuralPBCLoss, prob::ODEProblem, x0::VX, θ; dt=0.1) where 
    {T<:Real, VX<:Union{Vector{T}, SubArray{T}}}
    loss(ps) = l(trajectory(prob, x0, ps; saveat=dt, sensealg=DiffEqFlux.ReverseDiffAdjoint()))
    val, back = Zygote.pullback(loss, θ)
    return first(back(1)), val
end
function gradient(l::NeuralPBCLoss, prob::ODEProblem, batch::VB, θ; dt=0.1) where 
    {T<:Real, VX<:Union{Vector{T}, SubArray{T}}, VB<:Vector{VX}}
    function loss(ps)
        L = mapreduce(+, batch) do x0
            traj = trajectory(prob, x0, ps; saveat=dt, sensealg=DiffEqFlux.ReverseDiffAdjoint())
            l(traj)
        end
        L / length(batch)
    end
    val, back = Zygote.pullback(loss, θ)
    return first(back(1)), val
end
function gradient(ls::Tuple{Vararg{L}}, prob::ODEProblem, x0, θ; dt=0.1) where {L<:NeuralPBCLoss}
    loss(ps) = begin
        x = trajectory(prob, x0, ps; saveat=dt, sensealg=DiffEqFlux.ReverseDiffAdjoint())
        sum(l(x) for l in ls)
    end
    val, back = Zygote.pullback(loss, θ)
    return first(back(1)), val
end


struct SetDistanceLoss{T,F} <: NeuralPBCLoss
    xstar::Vector{T}
    radius::T
    f::F
end
function SetDistanceLoss(f::Function,xstar,r)
    SetDistanceLoss{eltype(xstar),typeof(f)}(xstar,r,f)
end
function (l::SetDistanceLoss)(x::AbstractVector)
    delta = l.f(x)
    return delta < l.radius ? zero(eltype(x)) : delta - l.radius
end
function (l::SetDistanceLoss)(x::AbstractMatrix)
    delta = mapreduce(l.f, min, eachcol(x))
    return delta < l.radius ? zero(eltype(x)) : delta - l.radius
end


abstract type AbstractNeuralPBCSampler{N} <: Function end

function (d::AbstractNeuralPBCSampler{N})() where {N}
    return randn(N)
end
function (d::AbstractNeuralPBCSampler{N})(batchsize::Int) where {N}
    return collect(d() for _=1:batchsize)
end

struct CustomDagger{N,ODE,F,D} <: AbstractNeuralPBCSampler{N}
    pode::ODE
    transformation::F
    dxstar::D
end
function CustomDagger(pode::ODEProblem, xstar::AbstractVector, f::Function=identity; 
    Σ=LinearAlgebra.I
)
    N = length(xstar)
    dxstar = MvNormal(xstar, Σ)
    CustomDagger{N,typeof(pode),typeof(f),typeof(dxstar)}(pode,f,dxstar)
end

function (d::CustomDagger)(batchsize::Int)
    if rand(Bool)
        xbar = rand(d.dxstar)
        xbar[1:2] *= pi
        x0 = d.transformation(xbar)
        forward = eachcol(trajectory(d.pode, x0))
        return collect.(rand(forward, batchsize))
    else
        samples = rand(d.dxstar, batchsize)
        return map(d.transformation, eachcol(samples))
    end
end
function (d::CustomDagger)(batchsize::Int, θ::AbstractVector)
    if rand(Bool)
        xbar = rand(d.dxstar)
        xbar[1:2] *= pi
        x0 = d.transformation(xbar)
        forward = collect(eachcol(trajectory(d.pode, x0, θ)))
        return collect.(rand(forward, batchsize))
    else
        samples = rand(d.dxstar, batchsize)
        return map(d.transformation, eachcol(samples))
    end
end
