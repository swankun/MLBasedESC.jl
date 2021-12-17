export NeuralPBC, unstack, controller
export SetDistanceLoss, gradient

struct NeuralPBC{N,HD} 
    Hd::HD
    function NeuralPBC(N::Int, Hd)
        new{N,typeof(Hd)}(N,Hd)
    end
end

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

function gradient(l::NeuralPBCLoss, prob::ODEProblem, x0, θ; dt=0.1)
    x = trajectory(prob, x0, θ; saveat=dt, sensealg=DiffEqFlux.ReverseDiffAdjoint())
    loss(ps) = l(x)
    val, back = Zygote.pullback(loss, θ)
    return first(back(1)), val
end
function gradient(ls::Tuple{Vararg{L}}, prob::ODEProblem, x0, θ; dt=0.1) where {L<:NeuralPBCLoss}
    loss(ps) = begin
        x = trajectory(prob, x0, θ; saveat=dt, sensealg=DiffEqFlux.ReverseDiffAdjoint())
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
function (l::SetDistanceLoss)(x::Matrix)
    delta = mapreduce(l.f, min, eachcol(x))
    return delta < l.radius ? zero(eltype(x)) : delta - l.radius
end
