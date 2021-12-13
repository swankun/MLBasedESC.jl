using MLBasedESC

using LinearAlgebra

import ForwardDiff
import ReverseDiff

import Flux
using Flux: Chain, Dense, elu
import DiffEqFlux
using DiffEqFlux: FastChain, FastDense

using OrdinaryDiffEq


#==============================================================================
Constants
==============================================================================#

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
function MLBasedESC.jacobian(::typeof(trueVd), q, ps=nothing)
    ForwardDiff.jacobian(_1->trueVd(_1,ps), q)
end

function eom!(dx,x,u)
    q1, q2, q1dot, q2dot = x
    dx[1] = q1dot
    dx[2] = q2dot
    dx[3] = m3*sin(q1)/I1 - u/I1
    dx[4] = u/I2
end
function eom(x,u)
    dx = similar(x)
    q1, q2, q1dot, q2dot = x
    eom!(dx,x,u)
    return dx
end

#==============================================================================
IDAPBC
==============================================================================#

function build_idapbc_model()
    Md⁻¹ = trueMd⁻¹
    Vd = Chain(
        Dense(2, 10, elu; bias=false),
        Dense(10, 5, elu; bias=false),
        Dense(5, 1, square; bias=false),
    )
    # Vd = SOSPoly(2,1:2)
    # Vd = FastChain(
    #     inmap,
    #     SOSPoly(4,1:2)
    # )
    # Vd = FastChain(
    #         FastDense(2, 10, elu; bias=false),
    #         FastDense(10, 5, elu; bias=false),
    #         FastDense(5, 1, square; bias=false),
    # )
    P = IDAPBCProblem(2,M⁻¹,trueMd⁻¹,V,Vd,G,G⊥)
    ps = Vd isa Function ? paramstack(P) : Flux.params(MLBasedESC.trainable(P)...)
    return P, ps
end


function train!(P, ps; dq=0.1pi)
    L1 = PDELossPotential(P)
    data = ([q1,q2] for q1 in -2pi:dq:2pi for q2 in -2pi:dq:2pi)
    optimize!(L1,ps,collect(data))
end

function simulate(P, ps::Flux.Params; x0=[3.,0,0,0], tf=3.0, kv=1)
    u_idapbc(x,p) = controller(P,x,kv=kv)
    sys = ParametricControlSystem{true}(eom!,u_idapbc,4)
    prob = ODEProblem(sys, (0.0, tf))
    trajectory(prob, x0; saveat=0.1)
end
function simulate(P, ps::AbstractVector; x0=[3.,0,0,0], tf=3.0, kv=1)
    u_idapbc(x,p) = controller(P,x,p,kv=kv)
    sys = ParametricControlSystem{true}(eom!,u_idapbc,4)
    prob = ODEProblem(sys, ps, (0.0, tf))
    trajectory(prob, x0, ps; saveat=0.1)
end

# P, θ = build_idapbc_model()
# train!(P, θ)
# simulate(P, θ)
