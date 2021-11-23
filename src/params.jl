struct IDAPBCParams{T,PROB}
    data::Vector{T}
    prob::PROB
end

function IDAPBCParams(prob::IDAPBCProblem, vals=MLBasedESC.params(prob))
    return IDAPBCParams{typeof(vals),typeof(prob)}(prob, vals)
end

function Base.getproperty(p::IDAPBCParams, sym::Symbol)
    if sym ∈ p.prob.ps_index.keys
        return getindex(p.data, p.prob.ps_index[sym])
    else # fallback to getfield
        return getfield(obj, sym)
    end
end
