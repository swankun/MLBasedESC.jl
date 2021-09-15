export Hamiltonian, gradient

struct Hamiltonian{isstatic,MA,PE,JM,JV}
    mass_inv::MA            # M⁻¹(q)
    potential::PE           # V(q)
    jac_mass_inv::JM        # (∂M⁻¹/∂q1, ∂M⁻¹/∂q2, ...)
    jac_pe::JV              # ∂V/∂q
    function Hamiltonian{isstatic}(mass_inv,pe,jac_mass_inv,jac_pe) where {isstatic}
        new{isstatic,typeof(mass_inv),typeof(pe),
            typeof(jac_mass_inv),typeof(jac_pe)}(
            mass_inv, pe, jac_mass_inv, jac_pe
        )
    end
end

function Hamiltonian(mass_inv::Matrix, potential::PE) where {PE<:Function}
    jac_pe(q,_=nothing) = ReverseDiff.gradient(potential, q)
    Hamiltonian{true}(
        (q,_=nothing)->mass_inv, 
        potential, 
        (q,_=nothing)->[zeros(size(mass_inv)...) for _=1:size(mass_inv,1)],
        jac_pe     
    )
end

function Hamiltonian(mass_inv::MA, potential::PE) where {MA<:Function, PE<:Function}
    jac_mass_inv(q,_=nothing) = begin
        n = length(q)
        jac = ReverseDiff.jacobian(mass_inv, q) 
        map(i->reshape(jac[:,i], n, :), 1:n)
    end
    jac_pe(q,_=nothing) = ReverseDiff.gradient(potential, q)
    Hamiltonian{true}(
        (q,_=nothing)->mass_inv(q), 
        potential, 
        jac_mass_inv,
        jac_pe     
    )
end

function Hamiltonian(mass_inv::MA, potential::PE) where {MA<:FunctionApproxmiator, PE<:Function}
    jac_mass_inv(q, θ=mass_inv.θ) = begin
        n = length(q)
        jac = reduce(vcat, gradient(mass_inv, q, θ))
        map(i->reshape(jac[:,i], n, :), 1:n)
    end
    jac_pe(q,_=nothing) = ReverseDiff.gradient(potential, q)
    Hamiltonian{false}(
        mass_inv, 
        potential, 
        jac_mass_inv,
        jac_pe     
    )
end

function Hamiltonian(mass_inv::MA, potential::PE) where {MA<:FunctionApproxmiator, PE<:FunctionApproxmiator}
    jac_mass_inv(q, θ=mass_inv.net.θ) = begin
        n = length(q)
        jac = reduce(vcat, gradient(mass_inv, q, θ))
        map(i->reshape(jac[:,i], n, :), 1:n)
    end
    jac_pe(q, θ=potential.θ) = gradient(potential, q, θ)
    Hamiltonian{false}(
        mass_inv, 
        potential, 
        jac_mass_inv,
        jac_pe     
    )
end

function (H::Hamiltonian)(q,p)
    return 1/2 * dot(p, H.mass_inv(q)*p) + H.potential(q)
end

function gradient(H::Hamiltonian, q, p)
    n = length(q)
    jac = H.jac_mass_inv(q)
    gs = map(i->jac[i]*p[i], 1:n)
    return 1/2 * (sum(gs)' * p) .+ H.jac_pe(q)[:]
end
