using MLBasedESC
using MLBasedESC: predict
using LinearAlgebra
using LaTeXStrings
using ReverseDiff
using StatsBase

struct Pendulum{T}
    m::T
    l::T
    b::T
    I::T
end

sys = Pendulum(1f0, 0.3f0, 0f0, 1f0*0.3f0^2)

function dynamics(x, u)
    cq, sq, qdot = x
    m  = sys.m
    l  = sys.l
    b  = sys.b
    I  = sys.I
    g  = 9.81f0
    ϵ  = 0.01f0
    ẍ1 = -sq*qdot - ϵ*cq*(sq^2 + cq^2 - 1)
    ẍ2 =  cq*qdot - ϵ*sq*(sq^2 + cq^2 - 1)
    ẍ3 = 1/I*(-m*g*l*sq - b*qdot + u)
    return [ẍ1, ẍ2, ẍ3]
end

function dynamics!(dx, x, u)
    cq, sq, qdot = x
    m  = sys.m
    l  = sys.l
    b  = sys.b
    I  = sys.I
    g  = 9.81f0
    ϵ  = 0.01f0
    dx[1] = -sq*qdot - ϵ*cq*(sq^2 + cq^2 - 1)
    dx[2] =  cq*qdot - ϵ*sq*(sq^2 + cq^2 - 1)
    dx[3] = 1/I*(-m*g*l*sq - b*qdot + u)
end

function loss(x)
    dist = eltype(x)(10)*abs.(2*(1 .+ x[1,:])) + x[3,:].^2 
    # dist = sqrt.(dist)
    return minimum(dist)
end

function assemble_data(;n=80, x0=Float32[cosd(170), sind(170), 0])
    traj = predict(Hd, x0, Hd.θ, 10f0)
    trajvec = collect(eachcol(traj))
    return sample(trajvec, n, replace=false)
end

Hd = EnergyFunction(Float32, 3, dynamics, loss, dim_S1=vcat(1), num_hidden_nodes=32)

# First start with short horizon with points near desired equilibrium
data = collect.(assemble_data())
Hd.hyper.time_horizon = 3f0
update!(Hd, data, batchsize=4, max_iters=10)

# Slowly expanding time horizon
data = collect.(assemble_data())
Hd.hyper.time_horizon = 6f0
update!(Hd, data, batchsize=4, max_iters=5)

# Now use starting points near downward equilibrium
data = collect.(assemble_data(x0=Float32[cosd(10),sind(10),0]))
Hd.hyper.time_horizon = 15f0
update!(Hd, data, batchsize=4, max_iters=5)

using Plots
traj = predict(Hd, Float32[cosd(10),sind(10),0], Hd.θ, 30f0)
traj = mapslices(x->vcat(atan(x[2],x[1]), x[3]), traj, dims=1)
plot(range(0, step=Hd.hyper.step_size, length=size(x,2)), traj')
