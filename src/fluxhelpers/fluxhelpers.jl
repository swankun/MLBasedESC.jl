module Fluxhelpers

export jacobian, derivative, @posdef, @odd, @skewsym

using Base: front, tail
using Flux
using LinearAlgebra

function ltril(v::AbstractVector)
    N = length(v)
    M = floor(Int, (-1 + sqrt(1 + 4*2*N))/2)
    P = cumsum(0:M-1)
    @inbounds columns = map(i->v[P.+i], 1:M)
    Q = reduce(hcat, columns)
    L = LowerTriangular(Q)
end

function posdef(v::AbstractVector)
    L = ltril(v)
    L*L'
end

function skewsym(v::AbstractVector)
    N = floor(Int, sqrt(length(v)))
    A = reshape(v,N,N)
    S = A - A'
end

function hadamard(A::Matrix,B::Matrix)
    nx = size(A,2)
    m,n = size(B)
    res = zeros(eltype(A), (m*nx, n*nx))
    for i = 1:nx
        res[m*(i-1)+1:m*i,n*(i-1)+1:n*i] = A[:,i] .* B
    end
    return res
end
hadamard(A::Vector, B::Matrix) = A .* B


delu(x::Real, α=one(x)) = ifelse(x > 0.0, one(x), α*exp(x) )
dtanh(x) = one(x) - tanh(x)*tanh(x)
square(x) = x.^2

derivative(f::typeof(elu)) = delu
derivative(f::typeof(tanh)) = dtanh
derivative(f::typeof(identity)) = (x)->one(eltype(x))
derivative(::typeof(square)) = (x)->2x

struct ActivationMode{T} end
const NoActivation = ActivationMode{:NoActivation}()
const DefaultActivation = ActivationMode{:Default}()

const FullyConnected = NTuple{N,Dense} where {N}
const OddDense = Tuple{typeof(square),Vararg{Dense,N}} where {N}

function (a::Dense)(x::AbstractVecOrMat, s::ActivationMode)
    s === DefaultActivation && return a(x)
    W, b = a.weight, a.bias
    return W*x .+ b
end

function _applychain(fs::FullyConnected, x)
    y = Flux.applychain(front(fs),x)
    last(fs)(y, NoActivation)
end

chainGrad(fs::Tuple{}, x) = LinearAlgebra.I
function chainGrad(fs::FullyConnected, x)
    y = _applychain(fs,x)
    σbar = derivative(last(fs).σ).(y)
    hadamard(σbar, last(fs).W) * chainGrad(front(fs), x)
end
function chainGrad(fs::OddDense, x)
    J = chainGrad(tail(fs), first(fs)(x))
    xbar = derivative(first(fs))(x)
    return J .* repeat(xbar', size(J,1))
end

function jacobian(m::Chain{T}, x) where {T<:Union{FullyConnected,OddDense}}
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

macro odd(model)
    return :( $(esc(model)) = Chain(square, $(esc(model)).layers...) )
end
macro posdef(model)
    return :( $(esc(model)) = Chain($(esc(model)).layers..., $(esc(posdef))) )
end
macro skewsym(model)
    return :( $(esc(model)) = Chain($(esc(model)).layers..., $(esc(skewsym))) )
end


end

using .Fluxhelpers
using Flux

C = Chain(
    Dense(2, 10, elu),
    Dense(10, 5, elu),
    Dense(5, 4),
)
l = C.layers
x = rand(2)
# jacobian(C, x)
