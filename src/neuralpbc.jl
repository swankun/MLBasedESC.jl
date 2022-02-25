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
    # gains = ones(Float32,N)
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
    batchsize = length(batch)
    gs = Vector{typeof(θ)}(undef, batchsize)
    ls = Vector{eltype(batch[1])}(undef, batchsize)
    Threads.@threads for id in 1:batchsize
        thisgs, thisls = gradient(l, prob, batch[id], θ, dt=dt)
        if !isnothing(thisgs)
            gs[id] = thisgs
        else
            gs[id] = 0θ
        end
        ls[id] = thisls
    end
    (sum(gs)/batchsize, sum(ls)/batchsize)
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
    return ifelse(delta < l.radius, zero(eltype(x)), delta - l.radius)
end
function (l::SetDistanceLoss)(x::AbstractMatrix)
    delta = mapreduce(l.f, min, eachcol(x))
    return ifelse(delta < l.radius, zero(eltype(x)), delta - l.radius)
end

