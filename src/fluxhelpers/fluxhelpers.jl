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
    M = floor(Int, (-1 + sqrt(1 + 4*2*N))/2)
    P = cumsum(0:M-1)
    @inbounds columns = map(i->v[P.+i], 1:M)
    Q = reduce(hcat, columns)
    L = LowerTriangular(Q)
end

function posdef(v::AbstractVector,::Any=nothing)
    L = ltril(v)
    L*L'
end

function skewsym(v::AbstractVector,::Any=nothing)
    N = floor(Int, sqrt(length(v)))
    A = reshape(v,N,N)
    S = A - A'
end

const SubMatrix = SubArray{T,2} where {T}

function hadamard(A::Matrix,B::M) where {M<:Union{Matrix,SubMatrix}}
    nx = size(A,2)
    m,n = size(B)
    res = zeros(eltype(A), (m*nx, n*nx))
    for i = 1:nx
        res[m*(i-1)+1:m*i,n*(i-1)+1:n*i] = A[:,i] .* B
    end
    return res
end
hadamard(A::Vector, B::M) where {M<:Union{Matrix,SubMatrix}} = A .* B

delu(x::Real, α=one(x)) = ifelse(x > 0.0, one(x), α*exp(x) )
dtanh(x) = one(x) - tanh(x)*tanh(x)
square(x,::Any=nothing) = x.^2

derivative(f::typeof(elu)) = delu
derivative(f::typeof(tanh)) = dtanh
derivative(f::typeof(identity)) = (x)->one(eltype(x))
derivative(::typeof(square)) = (x,::Any=nothing)->2x

struct ActivationMode{T} end
const NoActivation = ActivationMode{:NoActivation}()
const DefaultActivation = ActivationMode{:Default}()

function jacobian(::F, x) where {F<:Function}
    error("Analytical jacobian of $F required, but not implemented.")
    # ∇ = Flux.jacobian(f, x)[1]
    # return ∇
end

# Flux ####################################################

const DenseLayers = NTuple{N,Dense} where {N}
const OddDense = Tuple{typeof(square),Vararg{Dense,N}} where {N}
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

chainGrad(fs::Tuple{}, x) = LinearAlgebra.I
function chainGrad(fs::DenseLayers, x)
    y = _applychain(fs,x)
    σbar = derivative(last(fs).σ).(y)
    hadamard(σbar, last(fs).W) * chainGrad(front(fs), x)
end
function chainGrad(fs::OddDense, x)
    J = chainGrad(tail(fs), first(fs)(x))
    xbar = derivative(first(fs))(x)
    return J .* repeat(xbar', size(J,1))
end
function chainGrad(fs::InputDense, x)
    xbar = first(fs)(x)
    J = chainGrad(tail(fs), xbar)
    return J * jacobian(first(fs), xbar)
end

function jacobian(m::Chain{T}, x) where {T<:Union{DenseLayers,OddDense,InputDense}}
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
            dL = skewsym(-col)
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
const OddFastDense = Tuple{typeof(square),Vararg{FastDense,N}} where {N}
const InputFastDense = Tuple{<:Function,Vararg{FastDense,N}} where {N}

function (a::FastDense)(x::AbstractVecOrMat, p, s::ActivationMode)
    s === DefaultActivation && return a(x, p)
    W, b = param2Wb(a, p)
    return W*x .+ b
end
function _applychain(fs::Tuple, x, p)
    ps = paramslice(fs,p)
    psvec = reduce(vcat,front(ps))
    y = DiffEqFlux.applychain(front(fs),x,psvec)
    last(fs)(y, last(ps), NoActivation)
end
_applychain(fs::Tuple{FastDense}, x, p) = last(fs)(x, p, NoActivation)

function chainGrad(fs::FastDenseLayers, x, p)
    y = _applychain(fs,x,p)
    ps = last(paramslice(fs,p))
    W, _ = param2Wb(last(fs), ps)
    σbar = derivative(last(fs).σ).(y)
    return hadamard(σbar, W) * chainGrad(front(fs), x, p[1:end-length(ps)])
end
function chainGrad(fs::OddFastDense, x, p)
    J = chainGrad(tail(fs), first(fs)(x), p)
    xbar = derivative(first(fs))(x,p)
    return J .* repeat(xbar', size(J,1))
end
function chainGrad(fs::InputFastDense, x, p)
    xbar = first(fs)(x)
    J = chainGrad(tail(fs), xbar, p)
    return J * jacobian(first(fs), xbar)
end
chainGrad(fs::Tuple{}, x, p) = LinearAlgebra.I

function jacobian(m::FastChain{T}, x, p) where {T<:Union{FastDenseLayers,OddFastDense,InputFastDense}}
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
            dL = skewsym(-col)
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
paramslice(l::Tuple{}, p) = ()

function param2Wb(f::FastDense, p)
    np = DiffEqFlux.paramlength(f)
    W = @view p[reshape(1:(f.out*f.in),f.out,f.in)]
    b = @view p[(f.out*f.in + 1):np]
    W, b
end

makeodd(m::FastChain) = FastChain(square, m.layers...)
makeposdef(m::FastChain) = FastChain(m.layers..., posdef)
makeskewsym(m::FastChain) = FastChain(m.layers..., skewsym)

