export HyperParameters

mutable struct HyperParameters{T<:Real}
    epochs_per_minibatch::T
    traj_sampling_size::T
    replay_buffer_size::T
    time_horizon::T
    step_size::T
    regularization_coeff::T
end
function HyperParameters(
    T::DataType ;
    epochs_per_minibatch::Integer=1,
    traj_sampling_size::Integer=0,
    replay_buffer_size::Integer=256,
    time_horizon::Real=3.0,
    step_size::Real=1/20,
    regularization_coeff::Real=0.1
)
    HyperParameters(
        T(epochs_per_minibatch),
        T(traj_sampling_size),
        T(replay_buffer_size),
        T(time_horizon),
        T(step_size),
        T(regularization_coeff)
    )
end
function Base.show(io::IO, hyper::HyperParameters)
    for f in fieldnames(HyperParameters)
        print(io, f)
        print(io, " => ")
        print(io, getfield(hyper, f))
        println()
    end
end


dtanh(x) = one(x) - tanh(x)*tanh(x)
elu(x::Real, α=one(x)) = x > zero(x) ? x : α * (exp(x) - one(1))
delu(x::Real, α=one(x)) = ifelse(x > 0.0, one(x), α*exp(x) )
drelu(x::Real) = ifelse(x > 0.0, one(x), zero(x) )
const pif0 = Float32(pi)


function vec2tril(v::AbstractVector)
    N = length(v)
    M = floor(Int, (-1 + sqrt(1 + 4*2*N))/2)
    diag_ind = [Int(i*(i+1)/2) for i=1:M]
    sum( [diagm(-i => getindex(v, diag_ind[1+i:end] .- i)) for i=0:M-1] )
end
function vec2tril(v::AbstractVector, ::Bool)
    N = length(v)
    M = floor(Int, (1 + sqrt(1 + 4*2*N))/2)
    diag_ind = [Int(i*(i-1)/2) for i=2:M]
    sum( [diagm(-i-1 => getindex(v, diag_ind[1+i:end] .- i)) for i=0:M-1] )
end


