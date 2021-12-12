using MLBasedESC
using Flux
using LinearAlgebra
using ForwardDiff
using ReverseDiff
import DiffEqFlux
using DiffEqFlux: FastChain, FastDense
using OrdinaryDiffEq

const I1 = 0.0455
const I2 = 0.00425
const m3 = 0.183*9.81
const M⁻¹ = inv(diagm([I1, I2]))
V(q) = [ m3*(cos(q[1]) - 1.0) ]
MLBasedESC.jacobian(::typeof(V), q) = [-m3*sin(q[1]), zero(eltype(q))]
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
# Vd = FastChain(
#     inmap,
#     SOSPoly(4,1:2)
# )
Vd = FastChain(
        FastDense(2, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1, square),
)

const trueMd = [1.0 -1.01; -1.01 1.02025]
# const trueMd = [5.0 -10.0; -10.0 20.0+1e-6]
# const trueMd = [0.1I1 -I1*0.11; -I1*0.11 I2*10]
# const trueMd = [0.5I1 -I1*1.51; -I1*1.51 I2*50]
trueMd⁻¹ = inv(trueMd)
trueVd(q,ps) = begin
    a1,a2,a3 = trueMd[[1,2,4]]
    k1 = 1e-3
    γ2 = -I1*(a2+a3)/(I2*(a1+a2))
    z = q[2] + γ2*q[1]
    return [I1*m3/(a1+a2)*cos(q[1]) + 0.5*k1*z^2]
end
function MLBasedESC.jacobian(::typeof(trueVd), q,ps)
    ForwardDiff.jacobian(_1->trueVd(_1,ps), q)
end

# P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
P = IDAPBCProblem(2,M⁻¹,trueMd⁻¹,V,trueVd,G,G⊥)
L1 = PDELossPotential(P)
ps = paramstack(P)
q = [3., 0.]
u_idapbc(x,p) = controller(P,x,p,kv=20)

function eom(x,u)
    q1, q2, q1dot, q2dot = x
    [
        q1dot,
        q2dot,
        m3*sin(q1)/I1 - u/I1,
        u/I2
    ]
end
function eom!(dx,x,u)
    q1, q2, q1dot, q2dot = x
    dx[1] = q1dot
    dx[2] = q2dot
    dx[3] = m3*sin(q1)/I1 - u/I1
    dx[4] = u/I2
end


Hd = FastChain(
    FastDense(4, 10, elu),
    FastDense(10, 5, elu),
    FastDense(5, 1)
)
ps2 = DiffEqFlux.initial_params(Hd)
u_neuralpbc(x,p) = sum(jacobian(Hd,x,p))/length(p)

sys = ParametricControlSystem{true}(eom!,u_idapbc,4)
prob = ODEProblem(sys, ps, (0.0, 3.0))
x0 = [q; 0.0; 0.0]
# sys = ParametricControlSystem{!true}(eom,u_neuralpbc,4)
# prob = ODEProblem(sys, ps2, (0.0, 3.0))


function L3(q,ps)
    #=
    Adjoint method notes:
    - InterpolatingAdjoint() with oop dynamics works w/ IDAPBC
    - ReverseDiffAdjoint() with in-place dynamics works w/ IDAPBC
    - ReverseDiffAdjoint() tends to be the fastest w/ least mem usage
    - For dim(x)=4, oop dynamics is faster
    =#
    sum(abs2, 
        trajectory(
            prob, q, ps; 
            saveat=0.1,
            sensealg=DiffEqFlux.ReverseDiffAdjoint()
        )[[1,3,4],end]
    )
end
dL3(q,ps) = Flux.gradient(_2->L3(q,_2), ps)


# # MWE ReverseDiff breaking
# function fiip(du,u,p,t)
#     du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
#     du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
#   end
# p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
# prob = ODEProblem(fiip,u0,(0.0,10.0),p)
# sol = solve(prob,Tsit5(),rtol=1e-6,atol=1e-6)
# function sum_of_solution(x)
#     _prob = remake(prob,u0=x[1:2],p=x[3:end])
#     sum(solve(_prob,Vern9(),rtol=1e-6,atol=1e-6,saveat=0.1))
# end
# dx = ReverseDiff.gradient(sum_of_solution,[u0;p])
