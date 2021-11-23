using MLBasedESC
using LinearAlgebra
using OrdinaryDiffEq
using Plots; pyplot(display=false)

const DO_TRAIN = false

sys = InertiaWheelPendulum(pretrain=true)
DO_TRAIN && train!(sys)
x0 = [pi,0,1.0,0]
tf = 10.0
f! = dynamics!(sys, Kv=1e-3, b1=0.001, b2=0.002, umax=0.8)
ode = ODEProblem{true}(f!, x0, (zero(tf), tf))
sol = OrdinaryDiffEq.solve(ode, BS5(), saveat=tf/1000)
plot(sol.t, Array(sol)[1,:])
