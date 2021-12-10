export jacobian, derivative, @posdef, @odd, @skewsym, square
export makeposdef, makeodd, makeskewsym

using Base: front, tail
using LinearAlgebra

import Flux
import DiffEqFlux
using Flux: Chain, Dense, elu
using DiffEqFlux: FastChain, FastDense


function ltril(v::AbstractVector)
    N = length(v)
    M = isqrt(N)
    LowerTriangular(reshape(v,M,M))
end
function ltril(V::AbstractMatrix)
    cat(map(ltril, eachcol(V))..., dims=(1,2))
end
function posdef(v::AbstractVecOrMat,::Any=nothing)
    L = ltril(v)
    L*L'
end

function skewsym(v::AbstractVector,::Any=nothing)
    N = floor(Int, sqrt(length(v)))
    A = reshape(v,N,N)
    S = A - A'
end
function skewsym(V::AbstractMatrix,::Any=nothing)
    cat(map(skewsym, eachcol(V))..., dims=(1,2))
end

const SubMatrix = SubArray{T,2} where {T}

function hadamard(A::AbstractMatrix,B::M) where {M<:Union{AbstractMatrix,SubMatrix}}
    nx = size(A,2)
    m,n = size(B)
    res = zeros(eltype(A), (m*nx, n*nx))
    for i = 1:nx
        res[m*(i-1)+1:m*i,n*(i-1)+1:n*i] = A[:,i] .* B
    end
    return res
end
hadamard(A::AbstractVector, B::M) where {M<:Union{AbstractMatrix,SubMatrix}} = A .* B

delu(x::Real, α=one(x)) = ifelse(x > 0.0, one(x), α*exp(x) )
dtanh(x) = one(x) - tanh(x)*tanh(x)
square(x,::Any=nothing) = x.^2

derivative(::typeof(elu)) = delu
derivative(::typeof(tanh)) = dtanh
derivative(::typeof(identity)) = (x)->one(eltype(x))
derivative(::typeof(square)) = (x,::Any=nothing)->2x

jacobian(::typeof(square), x::AbstractVector, ::Any=nothing) = diagm(derivative(square)(x))
jacobian(l::Dense, x) = hadamard(derivative(l.σ).(x), l.W)
function jacobian(l::FastDense, x, p)
    W, _ = param2Wb(l, p)
    σbar = derivative(l.σ).(x)
    return hadamard(σbar, W)
end
function jacobian(::F, x, ::Any=nothing) where {F<:Function}
    error("Analytical jacobian of $F required, but not implemented.")
end

struct ActivationMode{T} end
const NoActivation = ActivationMode{:NoActivation}()
const DefaultActivation = ActivationMode{:Default}()

# Flux ####################################################

const DenseLayers = NTuple{N,Dense} where {N}
const InputDense = Tuple{<:Function,Vararg{Dense,N}} where {N}

function (a::Dense)(x::AbstractVecOrMat, s::ActivationMode)
    s === DefaultActivation && return a(x)
    W, b = a.weight, a.bias
    return W*x .+ b
end

function _applychain(fs::DenseLayers, x)
    y = Flux.applychain(front(fs),x)
    last(fs)(y, NoActivation)
end
_applychain(fs::InputDense, x) = _applychain(tail(fs), first(fs)(x))
_applychain(::Tuple{<:Function}, x) = x


chainGrad(::Tuple{}, x) = LinearAlgebra.I
function chainGrad(fs::T, x)  where {T<:Union{DenseLayers,InputDense}}
    y = _applychain(fs,x) 
    jacobian(last(fs), y) * chainGrad(front(fs), x)
end

function jacobian(m::Chain{T}, x) where {T<:Union{DenseLayers,InputDense}}
    chainGrad(m.layers,x)
end
function jacobian(c::Chain, x) 
    fs = front(c.layers)
    J = chainGrad(fs, x)
    fout = typeof(last(c.layers))
    if fout === typeof(posdef)
        y = Flux.applychain(fs, x)
        L = ltril(y)
        ∂L∂x = map(eachcol(J)) do col
            dL = ltril(col)
            (L * dL') + (dL * L')
        end
    elseif fout === typeof(skewsym)
        ∂L∂x = map(eachcol(J)) do col
            dL = skewsym(col)
        end 
    else
        error("Analytical Jacobian not supported for this Chain type. Use AD.")
    end
end

linear(m::Chain, x) = extraChain(Tuple(m.layers), x)
function extraChain(fs::Tuple, x)
    res = first(fs)(x, NoActivation)
    σ = first(fs).σ
    return (res, extraChain(Base.tail(fs), σ.(res))...)
end
extraChain(::Tuple{}, x) = ()

makeodd(m::Chain) = Chain(square, m.layers...)
makeposdef(m::Chain) = Chain(m.layers..., posdef)
makeskewsym(m::Chain) = Chain(m.layers..., skewsym)


# # DiffEqFlux ##############################################

const FastDenseLayers = NTuple{N,FastDense} where {N}
const InputFastDense = Tuple{<:Function,Vararg{FastDense,N}} where {N}

function (a::FastDense)(x::AbstractVecOrMat, p, s::ActivationMode)
    s === DefaultActivation && return a(x, p)
    W, b = param2Wb(a, p)
    return W*x .+ b
end
function _applychain(fs::FastDenseLayers, x, p)
    ps = paramslice(fs,p)
    psvec = reduce(vcat,front(ps))
    y = DiffEqFlux.applychain(front(fs),x,psvec)
    last(fs)(y, last(ps), NoActivation)
end
_applychain(fs::Tuple{FastDense}, x, p) = last(fs)(x, p, NoActivation)
_applychain(fs::InputFastDense, x, p) = _applychain(tail(fs), first(fs)(x), p)
_applychain(::Tuple{<:Function}, x, ::Any) = x

function chainGrad(fs::InputFastDense, x, p)
    y = _applychain(fs,x,p)
    ps = last(paramslice(fs,p))
    return jacobian(last(fs), y, ps) * chainGrad(front(fs), x, p[1:end-length(ps)])
end
chainGrad(::Tuple{}, ::Any, ::Any) = LinearAlgebra.I

function jacobian(m::FastChain{T}, x, p) where {T<:Union{FastDenseLayers,InputFastDense}}
    chainGrad(m.layers,x,p)
end
function jacobian(c::FastChain, x, p) 
    fs = front(c.layers)
    J = chainGrad(fs, x, p)
    fout = typeof(last(c.layers))
    if fout === typeof(posdef)
        y = DiffEqFlux.applychain(fs, x, p)
        L = ltril(y)
        ∂L∂x = map(eachcol(J)) do col
            dL = ltril(col)
            (L * dL') + (dL * L')
        end
    elseif fout === typeof(skewsym)
        ∂L∂x = map(eachcol(J)) do col
            dL = skewsym(col)
        end 
    else
        error("Analytical Jacobian not supported for this Chain type. Use AD.")
    end
end

paramslice(c::FastChain{<:FastDenseLayers}, p::Vector) = paramslice(c.layers,p)
function paramslice(fs::Tuple, p)   # layers fs must be in increasing order
    f = first(fs)
    N = DiffEqFlux.paramlength(f)
    res = @view(p[1:N])
    return (res, paramslice(tail(fs), p[N+1:end])...)
end
paramslice(::Tuple{}, ::Any) = ()

function param2Wb(f::FastDense, p)
    np = DiffEqFlux.paramlength(f)
    W = @view p[reshape(1:(f.out*f.in),f.out,f.in)]
    b = @view p[(f.out*f.in + 1):np]
    W, b
end

makeodd(m::FastChain) = FastChain(square, m.layers...)
makeposdef(m::FastChain) = FastChain(m.layers..., posdef)
makeskewsym(m::FastChain) = FastChain(m.layers..., skewsym)

