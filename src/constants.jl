export PSDMatrix, jacobian

struct PSDMatrix{N,T} <: Function where {N<:Integer,T<:DataType} 
    data::Vector{T}
end

PSDMatrix(T::DataType, N::Integer) = PSDMatrix{N,T}(T.(vec(Flux.glorot_uniform(N,N))))
PSDMatrix(N::Integer) = PSDMatrix(Float32,N)

function DiffEqFlux.initial_params(::PSDMatrix{N,T}) where {T,N}
    T.(vec(Flux.glorot_uniform(N,N)))
end
DiffEqFlux.paramlength(::PSDMatrix{N}) where {N} = N*N

(m::PSDMatrix)(::AbstractVecOrMat) = posdef(m.data)
(m::PSDMatrix)(::AbstractVecOrMat, θ) = posdef(θ)
function jacobian(m::PSDMatrix{N,T}, ::AbstractVecOrMat, ::Any=nothing) where {N,T}
    collect(zeros(T,N,N) for _=1:N)
end

