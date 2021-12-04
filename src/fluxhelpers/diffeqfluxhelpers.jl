using Base: front, tail
using LinearAlgebra
using DiffEqFlux
using Zygote

elu(x::Real, α=one(x)) = x > zero(x) ? x : α * (exp(x) - one(1))
delu(x::Real, α=one(x)) = ifelse(x > 0.0, one(x), α*exp(x) )
dtanh(x) = one(x) - tanh(x)*tanh(x)
square(x,_=nothing) = x.^2

derivative(f::typeof(elu)) = delu
derivative(f::typeof(tanh)) = dtanh
derivative(f::typeof(identity)) = (x)->one(eltype(x))
derivative(::typeof(square)) = (x,_=nothing)->2x

Cmodel = FastChain(
    FastDense(2, 10, elu),
    FastDense(10, 5, elu),
    FastDense(5, 3),
)
l = Cmodel.layers
x = rand(2)
p = initial_params(Cmodel);



struct ActivationMode{T} end
const NoActivation = ActivationMode{:NoActivation}()
const DefaultActivation = ActivationMode{:Default}()
const FullyConnected = NTuple{N,FastDense} where {N}

function (a::FastDense)(x::AbstractVecOrMat, p, s::ActivationMode)
    s === DefaultActivation && return a(x, p)
    W, b = param2Wb(a, p)
    return W*x .+ b
end
function _applychain(fs::Tuple, x, p)
    ps = paramslice(fs,p)
    y = DiffEqFlux.applychain(front(fs),x,reduce(vcat, front(ps)))
    last(fs)(y, last(ps), NoActivation)
end
_applychain(fs::Tuple{FastDense}, x, p) = last(fs)(x, p, NoActivation)


function hadamard(A::Matrix, B::Union{Matrix,SubArray})
    nx = size(A,2)
    m,n = size(B)
    res = zeros(eltype(A), (m*nx, n*nx))
    for i = 1:nx
        res[m*(i-1)+1:m*i,n*(i-1)+1:n*i] = A[:,i] .* B
    end
    return res
end
hadamard(A::Vector, B::Union{Matrix,SubArray}) = A .* B

jacobian(m::FastChain{<:FullyConnected}, x, p) = chainGrad(m.layers,x,p)
function chainGrad(fs::Tuple, x, p)
    y = _applychain(fs,x,p)
    ps = last(paramslice(fs,p))
    W, _ = param2Wb(last(fs), ps)
    # σ′ = derivative(last(fs).σ)
    # return (σ′.(y) .* W) * chainGrad(front(fs), x, p[1:end-length(ps)])
    σbar = derivative(last(fs).σ).(y)
    return hadamard(σbar, W) * chainGrad(front(fs), x, p[1:end-length(ps)])
end
chainGrad(fs::Tuple{}, x, p) = LinearAlgebra.I

paramslice(c::FastChain{<:FullyConnected}, p::Vector) = paramslice(c.layers,p)
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


macro symmetry(model)
    if !isa(eval(model).layers, FullyConnected)
        @warn "Imposing symmetry only supports FullyConnected Chain."
        return nothing
    end
    @eval begin
        $model = FastChain(square, $model.layers...)
        function jacobian(c::typeof($model), x, p)
            J = chainGrad(tail(c.layers), first(c.layers)(x), p)
            xbar = derivative(first(c.layers))(x)
            return J .* repeat(xbar', size(J,1))
        end
    end
    nothing
end

