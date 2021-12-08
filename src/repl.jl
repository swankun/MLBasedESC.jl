using .Tmp
using Flux
using LinearAlgebra
import DiffEqFlux
using DiffEqFlux: FastChain, FastDense

M⁻¹ = inv(diagm([0.1, 0.2]))
V(q) = [ 10.0*(cos(q[1]) - 1.0) ]
Tmp.jacobian(::typeof(V), q) = [-10.0*sin(q[1]), zero(eltype(q))]
G = [-1.0, 1.0]
G⊥ = [1.0 1.0]

inmap(q,::Any=nothing) = [cos(q[1]); sin(q[1]); cos(q[2]); sin(q[2])]
function Tmp.jacobian(::typeof(inmap), qbar)
    [
        -qbar[2] 0
         qbar[1] 0
         0 -qbar[4] 
         0  qbar[3] 
    ]
end

ConstMd⁻¹ = inv(diagm([0.1, 0.2]))
Md⁻¹ = Chain(
    inmap, Dense(4, 10, elu),
    # Dense(2, 10, elu),
    Dense(10, 5, elu),
    Dense(5, 3),
) |> makeposdef
Vd = Chain(
    inmap, Dense(4, 10, elu),
    # Dense(2, 10, elu),
    Dense(10, 5, elu),
    Dense(5, 1, square),
) #|> makeodd
J2 = Tuple(makeskewsym(Chain(
    inmap, Dense(4, 5, elu),
    # Dense(2, 5, elu),
    Dense(5, 4),
)) for _=1:2)

## Implicit params 
P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,J2,G,G⊥)
# P = IDAPBCProblem(2,M⁻¹,ConstMd⁻¹,V,Vd,G,G⊥)
L1 = PDELossPotential(P)
L2 = PDELossKinetic(P)
q = rand(2)
L1(q)
L2(q)
dL1(x) = Flux.gradient(Flux.params(Md⁻¹,Vd)) do
    L1(x)
end
dL2(x) = Flux.gradient(Flux.params(Md⁻¹,J2...)) do
    L2(x)
end
dL1(q)[Md⁻¹[2].W]
dL1(q)[Vd[2].W]
dL2(q)[Md⁻¹[2].W]
dL2(q)[J2[1][2].W]

## Explicit params
FastMd⁻¹ = FastChain(
    inmap, FastDense(4, 10, elu),
    # FastDense(2, 10, elu),
    FastDense(10, 5, elu),
    FastDense(5, 3),
) |> makeposdef
FastVd = FastChain(
    inmap, FastDense(4, 10, elu),
    # FastDense(2, 10, elu),
    FastDense(10, 5, elu),
    FastDense(5, 1, square),
) #|> makeodd
FastJ2 = Tuple(makeskewsym(FastChain(
    inmap, FastDense(4, 5, elu),
    # FastDense(2, 5, elu),
    FastDense(5, 4),
)) for _=1:2)
Pe = IDAPBCProblem(2,M⁻¹,FastMd⁻¹,V,FastVd,FastJ2,G,G⊥)
# Pe = IDAPBCProblem(2,M⁻¹,ConstMd⁻¹,V,FastVd,G,G⊥)
L1e = PDELossPotential(Pe)
L2e = PDELossKinetic(Pe)
ps = paramstack(Pe)
L1e(q,ps)
L2e(q,ps)
dL1e(q,ps) = Flux.gradient(_2->L1e(q,_2), ps)[1]
dL2e(q,ps) = Flux.gradient(_2->L2e(q,_2), ps)[1]
unstack(Pe, dL1e(q,ps))
unstack(Pe, dL2e(q,ps))
