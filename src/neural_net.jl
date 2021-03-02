struct LayerParamIndices{T<:Integer} 
    flat :: UnitRange{T}
    W    :: Array{T,2}
    b    :: UnitRange{T}
end


mutable struct NeuralNetwork{T<:Real}
    depth  :: Int
    widths :: Vector{Int}
    σ      :: Function
    σ′     :: Function
    _net   :: Union{Chain, DiffEqFlux.FastChain, Function}
    layers :: Tuple{Vararg{Union{Dense, DiffEqFlux.FastDense}}}
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
            initW=glorot_uniform, initb=glorot_uniform
        ) for i in 2:depth+1
    ])
    _net = FastChain(
        symmetric ? (x,p) -> x.^2 : (x,p) -> identity(x), 
        layers...
    )
    θ = initial_params(_net)

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
            LayerParamIndices{Int}(flat, W, b) 
        )
    end
    @assert length(inds) == depth

    NeuralNetwork{T}(depth, widths, σ, σ′, _net, layers, θ, inds)
end
NeuralNetwork(widths::Vector{Int}) = NeuralNetwork(Float32, widths)
function (net::NeuralNetwork)(x, p=net.θ)
    net._net(x, p)[:]
end
function Base.show(io::IO, net::NeuralNetwork)
    print(io, "NeuralNetwork: ")
    print(io, "widths = "); show(io, net.widths); print(io, ", ")
    print(io, "σ = "); show(io, net.σ); print(io, " ")
end


function set_params(net::NeuralNetwork, p::Vector{<:Real})
    net.θ = p
end

function get_weights(net::NeuralNetwork, θ, layer::Integer)
    @view θ[ net.inds[layer].flat ][ net.inds[layer].W ]
end


function get_biases(net::NeuralNetwork, θ, layer::Integer)
    @view θ[ net.inds[layer].flat ][ net.inds[layer].b ]
end


function _applychain(net::NeuralNetwork, θ, layers::Tuple, input)
    if layers isa Tuple{Integer}
        get_weights(net, θ, first(layers)) * net._net.layers[1](input, θ) + 
            get_biases(net, θ, first(layers))
    else
        get_weights(net, θ, first(layers)) * net.σ.(_applychain(net,θ,first(Base.tail(layers)), input)) + 
            get_biases(net, θ, first(layers))
    end
end
_applychain(net::NeuralNetwork, θ, i::Int, x) = _applychain(net, θ, Tuple(i:-1:1), x)


function _issymmetric(net::NeuralNetwork)
    input = rand(eltype(net.θ), net.widths[1])
    !(net._net.layers[1](input, net.θ) == input)
end
@nograd _issymmetric


function ∇NN(net::NeuralNetwork, x, θ=net.θ)
    net.layers[end].σ.( 
        get_weights(net, θ, net.depth) * 
        prod(net.σ′.(_applychain(net, θ, i, x)) .* get_weights(net, θ, i) for i = net.depth-1:-1:1) 
    )[:] .* (_issymmetric(net) ? 2x : one.(x))
end
