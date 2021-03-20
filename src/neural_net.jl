export NeuralNetwork, get_weights, get_biases, gradient

const IndexType = UInt16

struct LayerParamIndices
    flat :: UnitRange{IndexType}
    W    :: Array{IndexType, 2}
    b    :: UnitRange{IndexType}
end

mutable struct NeuralNetwork{T<:Real, ChainType<:Union{Chain, FastChain}, LayerType<:Union{Dense, FastDense}, F1, F2}
    depth  :: IndexType
    widths :: Vector{IndexType}
    σ      :: F1
    σ′     :: F2
    chain  :: ChainType
    layers :: Tuple{Vararg{LayerType}}
    θ      :: Vector{T}
    inds   :: Vector{LayerParamIndices}    # indices for (flattened θ, weights, biases) of each layer
end

function NeuralNetwork( T::Type,
                        widths::Vector{Int}, 
                        σ::Function=elu, 
                        σ′::Function=delu ;
                        symmetric::Bool=false )
    
    depth = length(widths)-1
    layers = Tuple([
        FastDense(
            widths[i-1], widths[i], i==depth+1 ? identity : σ,
            # widths[i-1], widths[i], i==depth+1 ? relu : σ,
            # widths[i-1], widths[i], i==depth+1 ? x->x.^2 : σ,
            initW=glorot_uniform, initb=glorot_uniform
        ) for i in 2:depth+1
    ])
    chain = FastChain(
        symmetric ? (x,p) -> x.^2 : (x,p) -> identity(x), 
        layers...
    )
    θ = initial_params(chain)

    θ_lens = [l.in * l.out + l.out for l in layers]
    θ_begin = [1; cumsum(θ_lens).+1]
    θ_end = cumsum(θ_lens)
    inds = LayerParamIndices[]
    for (i, j, layer, len) in zip(θ_begin, θ_end, layers, θ_lens)
        nin = layer.in
        nout = layer.out
        flat = i:j 
        W = collect(Int, reshape(1:(nin*nout), nout, nin))
        b = (nin*nout+1):len
        push!( 
            inds, 
            LayerParamIndices(flat, W, b) 
        )
    end
    @assert length(inds) == depth

    NeuralNetwork{T, typeof(chain), eltype(layers), typeof(σ), typeof(σ′)}(
        depth, widths, σ, σ′, chain, layers, θ, inds
    )
end

NeuralNetwork(widths::Vector{Int}) = NeuralNetwork(Float32, widths)

function (net::NeuralNetwork)(x, p=net.θ)
    net.chain(x, p)
    # net.chain([1.0], p)
end

function Base.show(io::IO, net::NeuralNetwork{T,C,D}) where {T,C,D}
    print(io, "NeuralNetwork{$(T),$(C),$(D)}, \n")
    print(io, "widths = $(Int.(net.widths))"); print(io, ", \n")
    print(io, "σ = "); show(io, net.σ); print(io, " ")
end




function set_params(net::NeuralNetwork, p::Vector{<:Real})
    net.θ = p
end

function get_weights(net::NeuralNetwork, θ, layer::Integer)
    θ[ net.inds[layer].flat ][ net.inds[layer].W ]
    # @view θ[ net.inds[layer].flat ][ net.inds[layer].W ]  # breaks ReverseDiff w.r.t θ
end

function get_biases(net::NeuralNetwork, θ, layer::Integer)
    θ[ net.inds[layer].flat ][ net.inds[layer].b ]
    # @view θ[ net.inds[layer].flat ][ net.inds[layer].b ] # breaks ReverseDiff w.r.t θ
end




function _applychain(net::NeuralNetwork, θ, layers::Tuple, input)
    get_weights(net, θ, first(layers)+1) * 
        net.σ.(_applychain(net,θ,Base.tail(layers),input)) + 
        # elu.(_applychain(net,θ,Base.tail(layers),input)) + 
        # _applychain(net,θ,Base.tail(layers), input) + 
        get_biases(net, θ, first(layers)+1)
end
function _applychain(net::NeuralNetwork, θ, ::Tuple{}, input)
    get_weights(net, θ, 1) * 
        net.chain.layers[1](input, θ) + 
        get_biases(net, θ, 1)
end
_applychain(net::NeuralNetwork, θ, i::Int, x) = _applychain(net, θ, Tuple(i-1:-1:1), x)

function _issymmetric(net::NeuralNetwork)
    input = rand(eltype(net.θ), net.widths[1])
    !(net.chain.layers[1](input, net.θ) == input)
end
@nograd _issymmetric




function gradient(net::NeuralNetwork, x, θ=net.θ)
    ∂NNx = identity.( 
        get_weights(net, θ, net.depth) * 
        prod(net.σ′.(_applychain(net, θ, i, x)) .* get_weights(net, θ, i) for i = net.depth-1:-1:1) 
    ) 
    if _issymmetric(net)
        return ∂NNx .* reduce(hcat, [fill(eltype(x)(2)*y, last(net.widths)) for y in x])
    else
        return ∂NNx
        # return drelu.(_applychain(net, θ, net.depth, x)) .* ∂NNx
        # return (2f0 * _applychain(net, θ, net.depth, x)) .* ∂NNx
    end
end




mutable struct PSDNeuralNetwork{N<:NeuralNetwork}
    n::Int
    net::N
end 

function PSDNeuralNetwork( T::Type,
    n::Integer,
    depth::Integer=3, 
    σ::Function=elu, 
    σ′::Function=delu ;
    nin::Integer=n,
    num_hidden_nodes=16,
    symmetric::Bool=false
)
    widths = vcat(
        nin, 
        fill(num_hidden_nodes, depth-1)..., 
        Int(n*(n+1)/2)
    )
    net = NeuralNetwork(T, widths, σ, σ′, symmetric=symmetric)
    PSDNeuralNetwork{typeof(net)}(n, net)
end

function (S::PSDNeuralNetwork)(x, p=S.net.θ)
    L = S.net.chain(x, p) |> vec2tril
    return L*L' + eltype(x)(1e-4)*I(S.n)
end

function Base.show(io::IO, S::PSDNeuralNetwork)
    print(io, "$(S.n) × $(S.n) positive semidefinite matrix NeuralNetwork: ")
    print(io, "widths = "); show(io, S.net.widths); print(io, ", ")
    print(io, "σ = "); show(io, S.net.σ); print(io, " ")
end

set_params(S::PSDNeuralNetwork, p::Vector{<:Real}) = set_params(S.net, p)
get_weights(S::PSDNeuralNetwork, θ, layer::Integer) = get_weights(S.net, θ, layer)
get_biases(S::PSDNeuralNetwork, θ, layer::Integer) = get_biases(S.net, θ, layer)

function gradient(S::PSDNeuralNetwork, x, θ=S.net.θ)
    """
    Returns Array{Matrix{T}} with 'nin' elements, and the ith matrix is
    the gradient of output w.r.t. the ith input
    """
    L = S.net.chain(x, θ) |> vec2tril
    ∂L∂x = [ vec2tril(col) for col in eachcol(gradient(S.net, x, θ)) ]
    return [ (L * dL') + (dL * L') for dL in ∂L∂x ]
end