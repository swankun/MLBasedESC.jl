using MLBasedESC
using LinearAlgebra
using Plots; pyplot()
using ReverseDiff

function dynamics(x, u)
    q1, q2, q1dot, q2dot = x
    g = eltype(x)(9.81)
    L = eltype(x)(10)
    ẋ1 = q1dot
    ẋ2 = q2dot
    ẋ3 = -g*sin(q2) + q1*q2dot^2
    ẋ4 = ( u - 2f0*q1*q1dot*q2dot - g*q1*cos(q2) ) / ( L^2 + q1^2 )
    return [ẋ1, ẋ2, ẋ3, ẋ4]
end
function dynamics!(dx, x, u)
    q1, q2, q1dot, q2dot = x
    g = eltype(x)(9.81)
    L = eltype(x)(10)
    dx[1] = -g*sin(q2) + q1*q2dot^2
    dx[2] = ( u - eltype(x)(2)*q1*q1dot*q2dot - g*q1*cos(q2) ) / ( L^2 + q1^2 )
end
function loss(x)
    dist = eltype(x)(1)*x[1,:].^2 +
        eltype(x)(1)*x[2,:].^2 +
        eltype(x)(1)*x[3,:].^2 +
        eltype(x)(2)*x[4,:].^2 
    return eltype(x)(10)*minimum(map(sqrt, dist)) + sum(sqrt, dist)/length(x)
    # return sum(abs2, dist)
end
function ∂KE∂q(q,j)
    if j == 2
        return eltype(q)[0 0; -(10f0^2 + q[1]^2)^(-2)*2*q[1] 0]
    else
        return zeros(eltype(q),(2,2))
    end
end
function ∂PE∂q(q)
    g = eltype(q)(9.81)
    return g*[ sin(q[2]); q[1]*cos(q[2]) ]
end
∂H∂q(q,p) = eltype(q)(1/2)*sum([∂KE∂q(q,k)*p[k] for k=1:2])'*p .+ ∂PE∂q(q)
function mass_matrix(q)
    return diagm(eltype(q)[1; 10f0^2 + q[1]^2])
end
const input_matrix = Float32[0; 1]
const input_matrix_perp = Float32[1 0]
function random_state(T::DataType)
    q1 = T(π/6)*T(2)*(rand(T) - T(0.5))
    q2 = T(π/6)*T(2)*(rand(T) - T(0.5))
    vcat(
        q1,
        q2,
        T(0.25)*T(2)*(rand(T) - T(0.5)),
        T(0.25)*T(2)*(rand(T) - T(0.5))
    )
end

NX = 4
dim_S1 = [2]
Hd_quad = QuadraticEnergyFunction(Float32,
    4, dynamics, loss, ∂KE∂q, ∂PE∂q, mass_matrix, input_matrix, input_matrix_perp, 
    num_hidden_nodes=16, symmetric=!true
)
x0 = Float32[4,0,1,1]

surf_Vd(type=:surface) = begin
    Q1 = range(-10, 10, length=20)
    Q2 = range(-pi/4, pi/4, length=20)
    plot(
        Q1, Q2,
        (x,y) -> Hd_quad.Vd( Float32[x; y] )[1],
        st=type
    )
end

surf_Md(type=:surface) = begin
    Q1 = range(-10, 10, length=20)
    Q2 = range(-pi/4, pi/4, length=20)
    plot(
        Q1, Q2,
        (x,y) -> cond( inv(Hd_quad.Md_inv( Float32[x; y] )) ),
        st=type
    )
end

function plot_traj(Hd; x0=Float32[8,0,1,1], tf=Hd_quad.hyper.time_horizon)
    num_plot_samples = 500
    old_tf = Hd_quad.hyper.time_horizon
    old_dt = Hd_quad.hyper.step_size
    Hd_quad.hyper.time_horizon = typeof(old_tf)(tf)
    Hd_quad.hyper.step_size = typeof(old_tf)(tf/num_plot_samples)
    t = range(0, Hd_quad.hyper.time_horizon, step=Hd_quad.hyper.step_size)
    x = predict(Hd_quad, x0)
    Hd_quad.hyper.time_horizon = old_tf 
    Hd_quad.hyper.step_size = old_dt
    plot(
        plot(t,x[1,:], label="q1"),
        plot(t,x[2,:], label="q2"),
        plot(t,x[3,:], label="q1dot"),
        plot(t,x[4,:], label="q2dot"),
        plot(t,x[5,:], label="u"),
        dpi=100, layout=(5,1)
    )
end

true_Vd(q, kp=1f0) = begin
    g = 9.81f0
    L = 10f0
    a = g*(1-cos(q[2]))
    b = kp/2 * (q[2] - 1/sqrt(2)*asinh(q[1]/L))^2
    return a + b
end