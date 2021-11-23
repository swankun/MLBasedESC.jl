export InterconnectionMatrix, IDAPBCProblem, set_constant_Md!, params, controller


struct InterconnectionMatrix{isstatic,T,UK}
    uk_collection::Vector{UK}
    init_params::Vector{T}
    ps_index::Dict{Symbol,UnitRange{Int}}
    function InterconnectionMatrix{isstatic}(uk_collection, init_params, ps_index) where {isstatic}
        new{isstatic, eltype(init_params), eltype(uk_collection)}(
            uk_collection, init_params, ps_index
        )
    end
end

function InterconnectionMatrix(uks::Vararg{SkewSymNeuralNetwork,N}) where {N}
    uk_collection = typeof(first(uks))[]
    init_params = eltype(first(uks).net.θ)[]
    ps_index = Dict{Symbol,UnitRange{Int}}()
    prevlen = 0
    for (k, uk) in enumerate(uks)
        push!(uk_collection, uk)
        append!(init_params, uk.net.θ)
        thislen = length(uk.net.θ)
        push!(ps_index, 
            Symbol("interconnection_u",k) => 
            prevlen+1+(k-1)*thislen : prevlen+k*thislen
        )
    end
    InterconnectionMatrix{false}(uk_collection, init_params, ps_index)
end

function InterconnectionMatrix(uks::Vararg{Function,N}) where {N}
    uk_collection = typeof(first(uks))[]
    init_params = Float32[]
    ps_index = Dict{Symbol,UnitRange{Int}}()
    prevlen = 0
    for (k, uk) in enumerate(uks)
        push!(uk_collection, uk)
        append!(init_params, 0)
        thislen = 1
        push!(ps_index, 
            Symbol("interconnection_u",k) => 
            prevlen+1+(k-1)*thislen : prevlen+k*thislen
        )
    end
    InterconnectionMatrix{true}(uk_collection, init_params, ps_index)
end

function InterconnectionMatrix(uks::Vararg{Matrix{T},N}) where {T,N}
    _func(q,_=nothing) = first(uks)
    uk_collection = typeof(_func)[]
    init_params = T[]
    ps_index = Dict{Symbol,UnitRange{Int}}()
    prevlen = 0
    for (k, uk) in enumerate(uks)
        push!(uk_collection, _func)
        append!(init_params, 0)
        thislen = 1
        push!(ps_index, 
            Symbol("interconnection_u",k) => 
            prevlen+1+(k-1)*thislen : prevlen+k*thislen
        )
    end
    InterconnectionMatrix{true}(uk_collection, init_params, ps_index)
end

function (J2::InterconnectionMatrix)(q,p)
    sum(eltype(q)(1/2)*uk(q)*p[k] for (k,uk) in enumerate(J2.uk_collection))
end
Base.getindex(J2::InterconnectionMatrix, i::Int) = getindex(J2.uk_collection, i)


struct IDAPBCProblem{T,H1,H2,J2}
    ham::H1
    hamd::H2
    input::VecOrMat{T}
    input_annihilator::VecOrMat{T}
    interconnection::J2
    init_params::Vector{T}
    ps_index::Dict{Symbol,UnitRange{Int}}
end

function IDAPBCProblem(ham::Hamiltonian{true}, hamd::Hamiltonian{false}, 
    input, input_annihilator, interconnection::InterconnectionMatrix
)
    if isa(hamd.mass_inv, FunctionApproxmiator) && !isa(hamd.mass_inv, PSDMatrix)
        θMd = hamd.mass_inv.net.θ
        inferred_precision = precisionof(hamd.mass_inv.net)
    elseif isa(hamd.mass_inv, PSDMatrix)
        θMd = hamd.mass_inv.θ
        inferred_precision = eltype(hamd.mass_inv([0f0]))
    else
        inferred_precision = eltype(hamd.mass_inv([0f0]))
        θMd = inferred_precision[0.0]
    end
    init_params = [θMd; hamd.potential.θ]
    ps_index = Dict(
        :mass_inv => 1 : length(θMd), 
        :potential => length(θMd)+1 : length(θMd)+length(hamd.potential.θ)
    )
    prevlen = length(init_params)
    append!(init_params, interconnection.init_params)
    for (k, uk_ps_index) in enumerate(interconnection.ps_index)
        thislen = length(uk_ps_index)
        push!(ps_index, 
            Symbol("j2_u",k) => 
            prevlen .+ uk_ps_index.second
        )
    end
    IDAPBCProblem{inferred_precision, typeof(ham), typeof(hamd), typeof(interconnection)}(
        ham, hamd, input, input_annihilator, interconnection, init_params, ps_index)
end

function IDAPBCProblem(ham::Hamiltonian{true}, hamd::Hamiltonian{true}, 
    input, input_annihilator, interconnection::InterconnectionMatrix
)
    inferred_precision = eltype(hamd.mass_inv([0.]))    # TODO: Infer this in some other way
    init_params = zeros(inferred_precision,2)
    ps_index = Dict(:mass_inv =>1:1, :potential=>2:2)
    prevlen = length(init_params)
    append!(init_params, interconnection.init_params)
    for (k, uk_ps_index) in enumerate(interconnection.ps_index)
        thislen = length(uk_ps_index)
        push!(ps_index, 
            Symbol("j2_u",k) => 
            prevlen .+ uk_ps_index.second
        )
    end
    IDAPBCProblem{inferred_precision, typeof(ham), typeof(hamd), typeof(interconnection)}(
        ham, hamd, input, input_annihilator, interconnection, init_params, ps_index)
end

function IDAPBCProblem(ham::Hamiltonian{true}, hamd, input, input_annihilator)
    nq = isstatic(hamd) || !isa(hamd.mass_inv, FunctionApproxmiator) ? size(hamd.mass_inv([0.]),1) : hamd.mass_inv.n     # TODO: Infer this in some other way
    J = zeros(eltype(input), (nq,nq))
    J2 = InterconnectionMatrix( (J for _=1:nq)... )
    IDAPBCProblem(ham, hamd, input, input_annihilator, J2)
end

function Base.show(io::IO, prob::IDAPBCProblem)
    print(io, "IDAPBCProblem{$(typeof(prob).parameters[1])}\n")
    print(io, "Mass matrix\n\t"); show(io, prob.hamd.mass_inv); print(io, "\n")
    print(io, "Potential energy\n\t"); show(io, prob.hamd.potential)
end

function Base.getproperty(p::IDAPBCProblem, sym::Symbol)
    if sym === :Vd
        return p.hamd.potential
    elseif sym === :∇Vd
        return p.hamd.jac_pe
    elseif sym === :Md
        return p.hamd.mass_inv
    elseif sym === :∇Md
        return p.hamd.jac_mass_inv
    elseif sym === :θ
        return p.init_params
    elseif sym === :θMd
        return getindex(p.init_params, p.ps_index[:mass_inv])
    elseif sym === :θVd
        return getindex(p.init_params, p.ps_index[:potential])
    elseif contains(string(sym), "θU") && isdigit(last(string(sym)))
        return getindex(p.init_params, p.ps_index[Symbol( string("j2_u", last(string(sym))) )])
    else # fallback to getfield
        return getfield(p, sym)
    end
end

function params(prob::IDAPBCProblem)
    deepcopy(prob.init_params)
end

function controller(prob::IDAPBCProblem{T}, θ=prob.init_params; damping_gain=T(1.0)) where {T}
    G = prob.input
    Gtop = transpose(G)
    massd_inv_ps = getindex(θ, prob.ps_index[:mass_inv])
    ped_ps = getindex(θ, prob.ps_index[:potential])
    H = prob.ham
    Hd = prob.hamd
    Uk = prob.interconnection
    u(q,p) = begin
        mass_inv = H.mass_inv(q)
        massd_inv = Hd.mass_inv(q,massd_inv_ps)
        massd = inv(massd_inv)
        J2 = sum(T(1/2)*Uk[j](q)*p[j] for j in 1:lastindex(p))
        Gu_es = gradient(H, q, p) .- 
            (massd*mass_inv)*gradient(Hd, q, p, massd_inv_ps, ped_ps) .+ (J2*massd_inv*p)
        u_es = dot( inv(Gtop*G)*Gtop, Gu_es )
        u_di = -damping_gain*dot(G, T(1)*massd_inv*p)
        return u_es + u_di
    end
end 

function set_constant_Md!(prob::IDAPBCProblem, θ::Vector, target::Matrix)
    @assert size(target,1) == prob.hamd.mass_inv.n
    L = tril2vec(cholesky(inv(target)).L)
    Mdinv = prob.hamd.mass_inv
    if isa(Mdinv, PSDNeuralNetwork)
        net = Mdinv.net
        zeroweight!(θ, net, :mass_inv, prob.ps_index)
        last_layer_ps_idx = net.inds[end].flat
        last_layer_bias_idx = last_layer_ps_idx[net.inds[end].b]
        θ[ prob.ps_index[:mass_inv][last_layer_bias_idx] ] .= eltype(θ).(L)
        nothing
    elseif isa(Mdinv, PSDMatrix)
        θ[prob.ps_index[:mass_inv]] .= eltype(θ).(L)
        nothing
    else
        @warn "Only PSDNeuralNetwork or PSDMatrix can be set to a constant."
    end
end
