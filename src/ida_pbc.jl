export InterconnectionMatrix, IDAPBCProblem, params, controller, 
    loss_massd, ∂loss_massd, loss_ped, ∂loss_ped, solve_sequential!,
    pmap, pmap!


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
    sum(1/2*uk(q)*p[k] for (k,uk) in enumerate(J2.uk_collection))
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
    init_params = [hamd.mass_inv.net.θ; hamd.potential.θ]
    ps_index = Dict(
        :mass_inv => 1 : length(hamd.mass_inv.net.θ), 
        :potential => length(hamd.mass_inv.net.θ)+1 : length(hamd.mass_inv.net.θ)+length(hamd.potential.θ)
    )
    inferred_precision = precisionof(hamd.mass_inv.net)
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
    inferred_precision = eltype(hamd.mass_inv([0.]))
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
    nq = hamd.mass_inv.n
    J = zeros(eltype(input), (nq,nq))
    J2 = InterconnectionMatrix( (J for _=1:nq)... )
    IDAPBCProblem(ham, hamd, input, input_annihilator, J2)
end

function Base.show(io::IO, prob::IDAPBCProblem)
    print(io, "IDAPBCProblem{$(typeof(prob).parameters[1])}\n")
    print(io, "Mass matrix\n\t"); show(io, prob.hamd.mass_inv); print(io, "\n")
    print(io, "Potential energy\n\t"); show(io, prob.hamd.potential)
end

function params(prob::IDAPBCProblem)
    deepcopy(prob.init_params)
end

function controller(prob::IDAPBCProblem{T}; damping_gain=T(1.0)) where {T}
    G = prob.input
    Gtop = transpose(G)
    u(q,p) = begin
        mass_inv = prob.ham.mass_inv(q)
        massd_inv = prob.hamd.mass_inv(q)
        massd = inv(massd_inv)
        J2 = sum(1/2*prob.interconnection[j](q)*p[j] for j in 1:lastindex(p))
        Gu_es = gradient(prob.ham, q, p) .- (massd*mass_inv)*gradient(prob.hamd, q, p) .+ (J2*massd_inv*p)
        
        u_es = dot( inv(Gtop*G)*Gtop, Gu_es )
        u_di = -damping_gain*dot(G, 2*massd_inv*p)
        return u_es + u_di
    end
end 


function loss_massd(prob::IDAPBCProblem, q, θ=prob.init_params)
    massd_inv_ps = getindex(θ, prob.ps_index[:mass_inv])
    massd_inv = prob.hamd.mass_inv(q, massd_inv_ps)
    massd_inv_gs = prob.hamd.jac_mass_inv(q, massd_inv_ps)
    massd = inv(massd_inv)
    mass_inv = prob.ham.mass_inv(q)
    mass_inv_gs = prob.ham.jac_mass_inv(q)
    
    nq = prob.hamd.mass_inv.n
    map(1:nq) do j
        uk_ps = getindex(θ, prob.ps_index[Symbol("j2_u", j)])
        uk = prob.interconnection[j](q, uk_ps)
        sum(abs2, prob.input_annihilator * 
            (transpose(mass_inv_gs[j]) - (massd*mass_inv)*transpose(massd_inv_gs[j]) + uk*massd_inv))
    end |> sum
end
∂loss_massd(prob::IDAPBCProblem) = (q,ps) -> first(Zygote.gradient(w->loss_massd(prob,q,w), ps))

function loss_ped(prob::IDAPBCProblem, q, θ=prob.init_params)
    ped_ps = getindex(θ, prob.ps_index[:potential])
    ped_gs = vec(prob.hamd.jac_pe(q, ped_ps))
    pe_gs = prob.ham.jac_pe(q)[1:prob.hamd.mass_inv.n]
    massd_inv_ps = getindex(θ, prob.ps_index[:mass_inv])
    massd = inv(prob.hamd.mass_inv(q,massd_inv_ps))
    mass_inv = prob.ham.mass_inv(q)
    abs2(dot( prob.input_annihilator, pe_gs - (massd*mass_inv)*ped_gs ))
end
∂loss_ped(prob::IDAPBCProblem) = (q,ps) -> ReverseDiff.gradient(w -> loss_ped(prob,q,w), ps)

function pmap(f, batch::Vector{Vector{T}}) where {T<:Real}
    l = Threads.Atomic{eltype(first(batch))}(0)
    n = length(batch)
    Threads.@threads for id in 1:n
        Threads.atomic_add!(l, f(batch[id]))
    end
    return l.value
end

function pmap!(out, f, batch::Vector{Vector{T}}) where {T<:Real}
    n = length(batch)
    Threads.@threads for id in 1:n
        out[id] = f(batch[id])
    end
end

function zeroexcept!(vec::Vector, target, keypairs)
    for (key,val) in keypairs
        if key != target
            vec[val] .= zero(eltype(vec))
        end
    end
end

function solve_sequential!(prob::IDAPBCProblem, paramvec, data, qdesired; batchsize=64, η=0.001, maxiters=1000)
    ℓ1  = loss_massd
    ∂ℓ1 = ∂loss_massd(prob)
    ℓ2(q,θ) = loss_ped(prob,q,θ) + abs2(prob.hamd.potential(qdesired,θ)[1])
    ∂ℓ2 = (q,θ) -> ReverseDiff.gradient(w->ℓ2(q,w), θ)
    batchgs = Vector{typeof(paramvec)}(undef, batchsize)

    dataloader = Data.DataLoader(data; batchsize=batchsize, shuffle=true)
    max_batch = round(Int, dataloader.imax / dataloader.batchsize, RoundUp)
    optimizer = ADAM(η)

    bstatus(b,bs,l) = @printf("Batch %05d/%05d | %10.4e\r", b, bs, l)
    estatus(t,e,l) = @printf("EPOCH %05d/%05d | TRAIN LOSS (%s) = %10.4e\n", e, maxiters, t, l)

    nepoch = 0; 
    train_loss = 1/length(data) * pmap(x->ℓ1(prob,x,paramvec), data)
    while nepoch < maxiters && train_loss > 1e-4
        estatus("Md", nepoch, train_loss)
        nbatch = 1
        for batch in dataloader
            npoints = length(batch)
            batchloss = 1/npoints * pmap(x->ℓ1(prob,x,paramvec), batch)
            bstatus(nbatch, max_batch, batchloss)
            pmap!(batchgs, x->∂ℓ1(x,paramvec), batch)
            grads = 1/npoints * sum(batchgs[1:npoints])
            if !any(isnan.(grads))
                Optimise.update!(optimizer, paramvec, grads)
                nbatch += 1
            end
        end
        nepoch += 1
        train_loss = 1/length(data) * pmap(x->ℓ1(prob,x,paramvec), data)
    end
    estatus("Md", nepoch, 1/length(data) * pmap(x->ℓ1(prob,x,paramvec), data))
    GC.gc()

    nepoch = 0; 
    train_loss = 1/length(data) * pmap(x->ℓ2(x,paramvec), data)
    while nepoch < maxiters && train_loss > 1e-4
        estatus("Vd", nepoch, train_loss)
        nbatch = 1
        for batch in dataloader
            npoints = length(batch)
            batchloss = 1/npoints * pmap(x->ℓ2(x,paramvec), batch)
            bstatus(nbatch, max_batch, batchloss)
            pmap!(batchgs, x->∂ℓ2(x,paramvec), batch)
            grads = 1/npoints * sum(batchgs[1:npoints])
            zeroexcept!(grads, :potential, prob.ps_index)
            if !any(isnan.(grads))
                Optimise.update!(optimizer, paramvec, grads)
                nbatch += 1
            end
        end
        nepoch += 1
        train_loss = 1/length(data) * pmap(x->ℓ2(x,paramvec), data)
    end
    estatus("Vd", nepoch, 1/length(data) * pmap(x->ℓ2(x,paramvec), data))
    GC.gc()

end
