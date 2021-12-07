struct IDAPBCProblem{J2,M,MD,V,VD,GT,GP}
    N::Int
    M⁻¹::M
    Md⁻¹::MD
    V::V
    Vd::VD
    J2::J2
    G::GT
    G⊥::GP
end

function IDAPBCProblem(N,M⁻¹,Md⁻¹,V,Vd,G::AbstractVecOrMat,G⊥::AbstractVecOrMat)
    IDAPBCProblem(N,M⁻¹,Md⁻¹,V,Vd,nothing,G,G⊥)
end

function Base.show(io::IO, p::IDAPBCProblem)
    println(io, "IDAPBCProblem [q ∈ ℜ^$(p.N)]");
    println(io, "M⁻¹  => $(typeof(p.M⁻¹).name.name)");
    println(io, "Md⁻¹ => $(typeof(p.Md⁻¹).name.name)");
    println(io, "Vd   => $(typeof(p.Vd).name.name)");
    println(io, "J2   => $(typeof(p.J2).name.name)");
end

function kinetic_pde(M⁻¹::M, Md::M, Md⁻¹::M, ∇M⁻¹::M, ∇Md⁻¹, G⊥::VecOrMat, J2=0) where {M<:Matrix}
    return G⊥*(∇M⁻¹' - Md*M⁻¹*∇Md⁻¹' + J2*Md⁻¹)
end
function kinetic_pde(p::IDAPBCProblem{Nothing,<:Matrix,<:Matrix}, q) 
    M⁻¹ = p.M⁻¹
    Md⁻¹ = p.Md⁻¹
    Md = inv(Md⁻¹)
    map(1:p.N) do i
        sum(abs, kinetic_pde(M⁻¹, Md, Md⁻¹, 0*M⁻¹, 0, p.G⊥))
    end |> sum
end
function kinetic_pde(p::IDAPBCProblem{Nothing,<:Matrix}, q) 
    M⁻¹ = p.M⁻¹
    Md⁻¹ = p.Md⁻¹(q)
    Md = inv(Md⁻¹)
    ∇Md⁻¹ = jacobian(p.Md⁻¹, q)
    map(1:p.N) do i
        sum(abs, kinetic_pde(M⁻¹, Md, Md⁻¹, 0*M⁻¹, ∇Md⁻¹[i], p.G⊥))
    end |> sum
end
function kinetic_pde(p::IDAPBCProblem{Nothing}, q)
    M⁻¹ = p.M⁻¹(q)
    Md⁻¹ = p.Md⁻¹(q)
    M = inv(M⁻¹)
    Md = inv(Md⁻¹)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    ∇Md⁻¹ = jacobian(p.Md⁻¹, q)
    map(1:p.N) do i
        sum(abs, kinetic_pde(M⁻¹, Md, Md⁻¹, ∇M⁻¹[i], ∇Md⁻¹[i], p.G⊥))
    end |> sum
end
function kinetic_pde(p::IDAPBCProblem, q)
    M⁻¹ = p.M⁻¹(q)
    Md⁻¹ = p.Md⁻¹(q)
    M = inv(M⁻¹)
    Md = inv(Md⁻¹)
    ∇M⁻¹ = jacobian(p.M⁻¹, q)
    ∇Md⁻¹ = jacobian(p.Md⁻¹, q)
    map(1:p.N) do i
        sum(abs, kinetic_pde(M⁻¹, Md, Md⁻¹, ∇M⁻¹[i], ∇Md⁻¹[i], p.G⊥, p.J2[i](q)))
    end |> sum
end


function potential_pde(M⁻¹::M, Md::M, ∇V::V, ∇Vd::V, G⊥::VecOrMat) where {M<:Matrix, V<:Vector} 
    return dot(G⊥, ∇V - Md*M⁻¹*∇Vd) 
end
function potential_pde(p::IDAPBCProblem{TJ2,<:Matrix}, q) where {TJ2}
    M⁻¹ = p.M⁻¹
    Md⁻¹ = p.Md⁻¹(q)
    M = inv(M⁻¹)
    Md = inv(Md⁻¹)
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q)[:]
    abs( potential_pde(M⁻¹, Md, ∇V, ∇Vd, p.G⊥) )
end
function potential_pde(p::IDAPBCProblem, q)
    M⁻¹ = p.M⁻¹(q)
    Md⁻¹ = p.Md⁻¹(q)
    M = inv(M⁻¹)
    Md = inv(Md⁻¹)
    ∇V = jacobian(p.V, q)[:]
    ∇Vd = jacobian(p.Vd, q)[:]
    abs( potential_pde(M⁻¹, Md, ∇V, ∇Vd, p.G⊥) )
end
