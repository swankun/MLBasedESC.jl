abstract type AbstractStateSpace end
abstract type RealLine <: AbstractStateSpace end

struct S1{T<:Real} <: AbstractStateSpace
    q::T
    x::T
    y::T
    S1{T}(q,x,y) where {T<:Real} = begin
        x^2 + y^2 ≈ 1 ? new(q,x,y) : error("Invalid circle constraint")
    end
end
S1(T::DataType) = S1{T}(π, cos(π), sin(π))