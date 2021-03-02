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
delu(x::Real, α=one(x)) = ifelse(x > 0, one(x), α*exp(x) )
const pif0 = Float32(pi)


function flat_to_tril(k::Integer, n::Integer)
    #=
      Takes the index k of a vector and convert to the
      correponding indices for a lower triangular matrix
      Based on https://stackoverflow.com/questions/242711/
      Modified to work with one-based indices.
    =#
    k = (0 < k <= n*(n+1)/2) ? k - 1 : return;
    J = n*(n+1)/2-1-k
    K = floor((sqrt(8J+1)-1)/2)
    column_index = n - 1 - K
    row_index = k - n*(n+1)/2 + (K+1)*(K+2)/2
    return Int(row_index + 1), Int(column_index + 1)
end

function LinearAlgebra.tril(v::AbstractVector)
    N = length(v)
    M = floor(Integer, (-1 + sqrt(1 + 4*2*N))/2)
    diag_ind = [Int(i*(i+1)/2) for i=1:M]
    sum( diagm(-i => getindex(v, diag_ind[1+i:end] .- i)) for i=0:M-1 )
end
