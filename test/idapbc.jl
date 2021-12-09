using MLBasedESC
using Flux
using LinearAlgebra
import DiffEqFlux
using DiffEqFlux: FastChain, FastDense
using ReverseDiff

const N = 2
const M⁻¹ = inv(diagm([0.1, 0.2]))
const G = [-1.0, 1.0]
const G⊥ = [1.0 1.0]

V(q) = [ 10.0*(cos(q[1]) - 1.0) ]
MLBasedESC.jacobian(::typeof(V), q) = [-10.0*sin(q[1]), zero(eltype(q))]

inmap(q,::Any=nothing) = [cos(q[1]); sin(q[1]); cos(q[2]); sin(q[2])]
function MLBasedESC.jacobian(::typeof(inmap), q,::Any=nothing)
    qbar = inmap(q)
    [
        -qbar[2] 0
         qbar[1] 0
         0 -qbar[4] 
         0  qbar[3] 
    ]
end

function runalltests()
    test1_chain()
    test1_fastchain()
    test1_poly()
    # test2_chain()
    # test2_fastchain()
    # test2_poly()
    test3_chain()
    test3_fastchain()
    test3_poly()
    test4_chain()
    test4_fastchain()
    test4_poly()
end

#==============================================================================
Test 1 ===> constant M; constant Md; neural-net Vd; no input mapping; no J2
==============================================================================#
function test1_chain()
    q = rand(2)
    Md⁻¹ = inv(diagm([0.1, 0.2]))
    Vd = Chain(
        Dense(2, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1, square),
    ) 
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    L1(q)
    L2(q)
    θ = Flux.params(MLBasedESC.trainable(P)...)
    dL1(x) = Flux.gradient(()->L1(x), θ)
    dL2(x) = Flux.gradient(()->L2(x), θ)
    dL1(q)[ θ[rand(1:length(θ))] ]
    dL2(q)[ θ[rand(1:length(θ))] ]
end
function test1_fastchain()
    q = rand(2)
    Md⁻¹ = inv(diagm([0.1, 0.2]))
    Vd = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1, square),
    ) 
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    θ = paramstack(P)
    L1(q,θ)
    L2(q,θ)
    dL1(q,ps) = ReverseDiff.gradient(_2->L1(q,_2), ps)
    dL2(q,ps) = ReverseDiff.gradient(_2->L2(q,_2), ps)
    unstack(P, dL1(q,θ))
    unstack(P, dL2(q,θ))
end
function test1_poly()
    q = rand(2)
    Md⁻¹ = inv(diagm([0.1, 0.2]))
    Vd = SOSPoly(2, 1:2)
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    θ = paramstack(P)
    L1(q,θ)
    L2(q,θ)
    dL1(q,ps) = ReverseDiff.gradient(_2->L1(q,_2), ps)
    dL2(q,ps) = ReverseDiff.gradient(_2->L2(q,_2), ps)
    unstack(P, dL1(q,θ))
    unstack(P, dL2(q,θ))
end

#==============================================================================
Test 2 ===> constant M; constant Md; neural-net Vd; no input mapping; yes J2
==============================================================================#
function test2_chain()
    q = rand(2)
    Md⁻¹ = inv(diagm([0.1, 0.2]))
    Vd = Chain(
        Dense(2, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1, square),
    ) 
    J2 = Tuple(
        Chain(
            Dense(2, 5, elu),
            Dense(5, 4),
        ) |> makeskewsym for _=1:2
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    L1(q)
    L2(q)
    θ = Flux.params(MLBasedESC.trainable(P)...)
    dL1(x) = Flux.gradient(()->L1(x), θ)
    dL2(x) = Flux.gradient(()->L2(x), θ)
    dL1(q)[ θ[rand(1:length(θ))] ]
    dL2(q)[ θ[rand(1:length(θ))] ]
end
function test2_fastchain()
    q = rand(2)
    Md⁻¹ = inv(diagm([0.1, 0.2]))
    Vd = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1, square),
    ) 
    J2 = Tuple(
        FastChain(
            FastDense(2, 5, elu),
            FastDense(5, 4),
        ) |> makeskewsym for _=1:2
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    θ = paramstack(P)
    L1(q,θ)
    L2(q,θ)
    dL1(q,ps) = ReverseDiff.gradient(_2->L1(q,_2), ps)
    dL2(q,ps) = ReverseDiff.gradient(_2->L2(q,_2), ps)
    unstack(P, dL1(q,θ))
    unstack(P, dL2(q,θ))
end
function test2_poly()
    q = rand(2)
    Md⁻¹ = inv(diagm([0.1, 0.2]))
    Vd = SOSPoly(2, 1:2)
    J2 = Tuple(
        FastChain(
            FastDense(2, 5, elu),
            FastDense(5, 4),
        ) |> makeskewsym for _=1:2
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    θ = paramstack(P)
    L1(q,θ)
    L2(q,θ)
    dL1(q,ps) = ReverseDiff.gradient(_2->L1(q,_2), ps)
    dL2(q,ps) = ReverseDiff.gradient(_2->L2(q,_2), ps)
    unstack(P, dL1(q,θ))
    unstack(P, dL2(q,θ))
end

#==============================================================================
Test 3 ===> constant M; neural-net Md; neural-net Vd; no input mapping; no J2
==============================================================================#
function test3_chain()
    q = rand(2)
    Md⁻¹ = Chain(
        Dense(2, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 4),
    ) |> makeposdef
    Vd = Chain(
        Dense(2, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1, square),
    ) 
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    L1(q)
    L2(q)
    θ = Flux.params(MLBasedESC.trainable(P)...)
    dL1(x) = Flux.gradient(()->L1(x), θ)
    dL2(x) = Flux.gradient(()->L2(x), θ)
    dL1(q)[ θ[rand(1:length(θ))] ]
    dL2(q)[ θ[rand(1:length(θ))] ]
end
function test3_fastchain()
    q = rand(2)
    Md⁻¹ = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 4),
    ) |> makeposdef
    Vd = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1, square),
    ) 
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    θ = paramstack(P)
    L1(q,θ)
    L2(q,θ)
    dL1(q,ps) = ReverseDiff.gradient(_2->L1(q,_2), ps)
    dL2(q,ps) = ReverseDiff.gradient(_2->L2(q,_2), ps)
    unstack(P, dL1(q,θ))
    unstack(P, dL2(q,θ))
end
function test3_poly()
    q = rand(2)
    Md⁻¹ = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 4),
    ) |> makeposdef
    Vd = SOSPoly(2, 1:2) 
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    θ = paramstack(P)
    L1(q,θ)
    L2(q,θ)
    dL1(q,ps) = ReverseDiff.gradient(_2->L1(q,_2), ps)
    dL2(q,ps) = ReverseDiff.gradient(_2->L2(q,_2), ps)
    unstack(P, dL1(q,θ))
    unstack(P, dL2(q,θ))
end

#==============================================================================
Test 4 ===> constant M; neural-net Md; neural-net Vd; no input mapping; yes J2
==============================================================================#
function test4_chain()
    q = rand(2)
    Md⁻¹ = Chain(
        Dense(2, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 4),
    ) |> makeposdef
    Vd = Chain(
        Dense(2, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1, square),
    ) 
    J2 = Tuple(
        Chain(
            Dense(2, 5, elu),
            Dense(5, 4),
        ) |> makeskewsym for _=1:2
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    L1(q)
    L2(q)
    θ = Flux.params(MLBasedESC.trainable(P)...)
    dL1(x) = Flux.gradient(()->L1(x), θ)
    dL2(x) = Flux.gradient(()->L2(x), θ)
    dL1(q)[ θ[rand(1:length(θ))] ]
    dL2(q)[ θ[rand(1:length(θ))] ]
end
function test4_fastchain()
    q = rand(2)
    Md⁻¹ = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 4),
    ) |> makeposdef
    Vd = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1, square),
    ) 
    J2 = Tuple(
        FastChain(
            FastDense(2, 5, elu),
            FastDense(5, 4),
        ) |> makeskewsym for _=1:2
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    θ = paramstack(P)
    L1(q,θ)
    L2(q,θ)
    dL1(q,ps) = ReverseDiff.gradient(_2->L1(q,_2), ps)
    dL2(q,ps) = ReverseDiff.gradient(_2->L2(q,_2), ps)
    unstack(P, dL1(q,θ))
    unstack(P, dL2(q,θ))
end
function test4_poly()
    q = rand(2)
    Md⁻¹ = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 4),
    ) |> makeposdef
    Vd = SOSPoly(2,1:2)
    J2 = Tuple(
        FastChain(
            FastDense(2, 5, elu),
            FastDense(5, 4),
        ) |> makeskewsym for _=1:2
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    θ = paramstack(P)
    L1(q,θ)
    L2(q,θ)
    dL1(q,ps) = ReverseDiff.gradient(_2->L1(q,_2), ps)
    dL2(q,ps) = ReverseDiff.gradient(_2->L2(q,_2), ps)
    unstack(P, dL1(q,θ))
    unstack(P, dL2(q,θ))
end