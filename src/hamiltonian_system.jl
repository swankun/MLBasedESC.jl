struct HamiltonianSystem{T<:Real, F, JT, JV, M, G, GA}
    dim::Int
    dynamics::F
    keq::JT
    peq::JV
    M::M
    G::G
    Gperp::GA
    q0::Vector{T}
    p0::Vector{T}
end
