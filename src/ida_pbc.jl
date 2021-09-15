export IDAPBCProblem, pde_constraint_ke, pde_constraint_pe, controller

struct IDAPBCProblem{T,H1,H2,JD}
    ham::H1
    hamd::H2
    input::VecOrMat{T}
    input_annihilator::VecOrMat{T}
    j2_free::JD
    ps::Vector{T}
    ps_index::Dict{Symbol,UnitRange{Int}}
end

function IDAPBCProblem(ham::Hamiltonian{true}, hamd::Hamiltonian{false}, input, input_annihilator; j2_free=nothing)
    ps = [hamd.mass_inv.net.θ; hamd.potential.θ]
    ps_index = Dict(
        :mass_inv => 1 : length(hamd.mass_inv.net.θ), 
        :potential => length(hamd.mass_inv.net.θ)+1 : length(hamd.mass_inv.net.θ)+length(hamd.potential.θ)
    )
    findparam(net::NeuralNetwork{T}) where {T} = T
    inferred_precision = findparam(hamd.mass_inv.net)
    if !isnothing(j2_free)
        j2_free::Vector{SkewSymNeuralNetwork}
        prevlen = length(hamd.mass_inv.net.θ) + length(hamd.potential.θ)
        for (k, uk) in enumerate(j2_free)
            thislen = length(uk.net.θ)
            push!(ps, uk.net.θ)
            push!(_θind, 
                Symbol("j2_u",k) => 
                prevlen+1+(k-1)*thislen : prevlen+k*thislen
            )
        end
    end
    IDAPBCProblem{inferred_precision, typeof(ham), typeof(hamd), typeof(j2_free)}(
        ham, hamd, input, input_annihilator, j2_free, ps, ps_index)
end

function IDAPBCProblem(ham::Hamiltonian{true}, hamd::Hamiltonian{true}, input, input_annihilator)
    inferred_precision = eltype(hamd.mass_inv([0.]))
    ps = zeros(inferred_precision,2)
    ps_index = Dict(:mass_inv =>1:1, :potential=>2:2)
    IDAPBCProblem{inferred_precision, typeof(ham), typeof(hamd), Nothing}(
        ham, hamd, input, input_annihilator, nothing, ps, ps_index)
end

function Base.show(io::IO, prob::IDAPBCProblem)
    print(io, "IDAPBCProblem{$(typeof(prob).parameters[1])}\n")
    print(io, "Mass matrix\n\t"); show(io, prob.hamd.mass_inv); print(io, "\n")
    print(io, "Potential energy\n\t"); show(io, prob.hamd.potential)
end

function pde_constraint_ke(prob::IDAPBCProblem, q, ps=prob.ps)
    massd_inv_ps = getindex(ps, prob.ps_index[:mass_inv])
    massd_inv = prob.hamd.mass_inv(q, massd_inv_ps)
    massd_inv_gs = prob.hamd.jac_mass_inv(q, massd_inv_ps)
    massd = inv(massd_inv)
    mass_inv = prob.ham.mass_inv(q)
    mass_inv_gs = prob.ham.jac_mass_inv(q)
    
    nq = lastindex(q)
    map(1:nq) do j
        pdevec = transpose(mass_inv_gs[j]) - massd*mass_inv*transpose(massd_inv_gs[j])
        if !isnothing(prob.j2_free)
            uk_ps = getindex(ps, prob.ps_index[Symbol("j2_u", j)])
            uk = prob.j2_free[j](q, uk_ps)
            sum(abs2, prob.input_annihilator * (pdevec + uk*massd_inv))
        else
            sum(abs2, prob.input_annihilator * pdevec)
        end
        
    end |> sum
end

function pde_constraint_pe(prob::IDAPBCProblem, q, ps=prob.ps; freeze_mass_ps=true)
    ped_ps = getindex(ps, prob.ps_index[:potential])
    ped_gs = vec(prob.hamd.jac_pe(q, ped_ps))
    pe_gs = prob.ham.jac_pe(q)
    massd_inv_ps = getindex(ps, prob.ps_index[:mass_inv])
    massd = ifelse(freeze_mass_ps, 
        inv(prob.hamd.mass_inv(q)), inv(prob.hamd.mass_inv(q,massd_inv_ps)))
    mass_inv = prob.ham.mass_inv(q)
    dot( prob.input_annihilator, pe_gs - (massd*mass_inv)*ped_gs ) |> abs2
end

function controller(prob::IDAPBCProblem{T}; damping_gain=T(1.0)) where {T}
    G = prob.input
    Gtop = transpose(G)
    u(q,p) = begin
        mass_inv = prob.ham.mass_inv(q)
        massd_inv = prob.hamd.mass_inv(q)
        massd = inv(massd_inv)
        Gu_es = gradient(prob.ham, q, p) .- (massd*mass_inv)*gradient(prob.hamd, q, p)
        if !isnothing(prob.j2_free)
            j2 = zeros(T,length(p),length(p))
            for j in 1:lastindex(p)
                j2 = j2 .+ 1/2*prob.j2_free[j](q)*p[j]
            end
            Gu_es = Gu_es .+ (j2*massd_inv*p)
        end
        u_di = -damping_gain*dot(G, 2*massd_inv*p)
        return dot( inv(Gtop*G)*Gtop, Gu_es ) + u_di
    end
end 
