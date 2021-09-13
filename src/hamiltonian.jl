export Hamiltonian

struct Hamiltonian{MA,PE,JM,JV}
    mass::MA            # M(q)
    potential::PE       # V(q)
    jac_mass::JM        # (∂M/∂q1, ∂M/∂q2, ...)
    jac_pe::JV          # ∂V/∂q
end

function Hamiltonian(mass::Matrix, potential::PE) where {PE<:Function}
    jac_pe(q) = ReverseDiff.gradient(potential, q)
    Hamiltonian(
        (q)->mass, 
        potential, 
        [zeros(size(mass)...) for _=1:size(mass,1)],
        jac_pe     
    )
end
function Hamiltonian(mass::MA, potential::PE) where {MA<:Function, PE<:Function}
    jac_mass(q) = begin
        n = length(q)
        jac = ReverseDiff.jacobian(Mf, q) 
        map(i->reshape(jac[:,i], n, :), 1:n)
    end
    jac_pe(q) = ReverseDiff.gradient(potential, q)
    Hamiltonian(
        mass, 
        potential, 
        jac_mass,
        jac_pe     
    )
end
function Hamiltonian(mass::MA, potential::PE) where {MA<:FunctionApproxmiator, PE<:Function}
    jac_mass(q, θ=mass.θ) = begin
        n = length(q)
        jac = reduce(vcat, gradient(mass, q, θ))
        map(i->reshape(jac[:,i], n, :), 1:n)
    end
    jac_pe(q) = ReverseDiff.gradient(potential, q)
    Hamiltonian(
        mass, 
        potential, 
        jac_mass,
        jac_pe     
    )
end
function Hamiltonian(mass::MA, potential::PE) where {MA<:FunctionApproxmiator, PE<:FunctionApproxmiator}
    jac_mass(q, θ=mass.net.θ) = begin
        n = length(q)
        jac = reduce(vcat, gradient(mass, q, θ))
        map(i->reshape(jac[:,i], n, :), 1:n)
    end
    jac_pe(q, θ=potential.θ) = gradient(potential, q, θ)
    Hamiltonian(
        mass, 
        potential, 
        jac_mass,
        jac_pe     
    )
end

function (H::Hamiltonian)(q,p)
    return 1/2 * dot(p, H.mass(q)*p) + H.potential(q)
end

function gradient(H::Hamiltonian, q, p)
    n = length(q)
    jac = H.jac_mass(q)
    gs = map(i->jac[i]*p[i], 1:n)
    return 1/2 * (sum(gs)' * p) .+ gradient(H.potential, q)[:]
end
