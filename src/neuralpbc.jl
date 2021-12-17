export NeuralPBC, unstack, controller

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

