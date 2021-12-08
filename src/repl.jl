using MLBasedESC
using Flux
using LinearAlgebra
import DiffEqFlux
using DiffEqFlux: FastChain, FastDense

const M⁻¹ = inv(diagm([0.1, 0.2]))
V(q) = [ 10.0*(cos(q[1]) - 1.0) ]
MLBasedESC.jacobian(::typeof(V), q) = [-10.0*sin(q[1]), zero(eltype(q))]
const G = [-1.0, 1.0]
const G⊥ = [1.0 1.0]

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

Md⁻¹ = M⁻¹
# Vd = SOSPoly(2,1:2)
Vd = FastChain(
    inmap,
    SOSPoly(4,1:2)
)
P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
L1 = PDELossPotential(P)
ps = paramstack(P)
L1(q,ps)
