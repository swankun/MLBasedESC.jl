export PSDMatrix, jacobian

struct PSDMatrix{N,T,F} <: Function where {N<:Integer,T<:DataType} 
    data::Vector{T}
    initial_params::F
end

function PSDMatrix(T::DataType, N::Integer, init::Function)
    PSDMatrix{N,T,typeof(init)}( T.(init()), init )
end
PSDMatrix(N::Integer, init::Function) = PSDMatrix(Float32,N,()->vec(init()))
PSDMatrix(N::Integer) = PSDMatrix(N,()->vec(Flux.glorot_uniform(N,N)))

Flux.trainable(P::PSDMatrix) = (P.data,)
function DiffEqFlux.initial_params(P::PSDMatrix{N,T}) where {T,N}
    T.(P.initial_params())
end
DiffEqFlux.paramlength(::PSDMatrix{N}) where {N} = N*N

(m::PSDMatrix)(::AbstractVecOrMat) = posdef(m.data)
(m::PSDMatrix)(::AbstractVecOrMat, θ) = posdef(θ)
function jacobian(m::PSDMatrix{N,T}, ::AbstractVecOrMat, ::Any=nothing) where {N,T}
    collect(zeros(T,N,N) for _=1:N)
end

