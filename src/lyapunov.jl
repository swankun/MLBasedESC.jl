mutable struct LyapunovNN{T<:Real, S1, S2}
    depth  :: IndexType
    widths :: Vector{IndexType}
    σ      :: S1
    σ′     :: S2
    θ      :: Vector{T}
    inds   :: Vector{LayerParamIndices}    # indices for (flattened θ, weights, biases) of each layer
end

function LyapunovNN(T::Type, widths)
    @assert issorted(widths) "Each layer must not decrease input dimension"

    Wshape = Tuple(
        (widths[i], widths[i-1]) for i in 2:length(widths)
    )
end