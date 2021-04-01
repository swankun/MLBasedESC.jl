using MLBasedESC
using LinearAlgebra
using Plots; pyplot()
using ReverseDiff

function dynamics(x, u)
    cq1, sq1, cq2, sq2, q1dot, q2dot = x
    I1 = 0.1f0
    I2 = 0.2f0
    m3 = 10f0
    ϵ  = 0.01f0
    b1 = b2 = 0.01f0
    ẍ1 = -sq1*q1dot - ϵ*cq1*(sq1^2 + cq1^2 - 1)
    ẍ2 =  cq1*q1dot - ϵ*sq1*(sq1^2 + cq1^2 - 1)
    ẍ3 = -sq2*q2dot - ϵ*cq2*(sq2^2 + cq2^2 - 1)
    ẍ4 =  cq2*q2dot - ϵ*sq2*(sq2^2 + cq2^2 - 1)
    ẍ5 = m3*sq1/I1 - u/I1 - b1/I1*q1dot
    ẍ6 = 0f0 + u/I2 - b2/I2*q2dot
    return [ẍ1, ẍ2, ẍ3, ẍ4, ẍ5, ẍ6]
end
function dynamics!(dx, x, u)
    cq1, sq1, cq2, sq2, q1dot, q2dot = x
    I1 = 0.1f0
    I2 = 0.2f0
    m3 = 10f0
    ϵ  = 0.01f0
    b1 = b2 = 0.01f0
    dx[1] = -sq1*q1dot - ϵ*cq1*(sq1^2 + cq1^2 - 1)
    dx[2] =  cq1*q1dot - ϵ*sq1*(sq1^2 + cq1^2 - 1)
    dx[3] = -sq2*q2dot - ϵ*cq2*(sq2^2 + cq2^2 - 1)
    dx[4] =  cq2*q2dot - ϵ*sq2*(sq2^2 + cq2^2 - 1)
    dx[5] = m3*sq1/I1 - u/I1 - b1/I1*q1dot
    dx[6] = 0f0 + u/I2 - b2/I2*q2dot
end
function loss(x)
    dist = abs.(eltype(x)(10*2)*(one(eltype(x)) .- x[1,:])) +
        abs.(eltype(x)(1*2)*(one(eltype(x)) .- x[3,:])) +
        x[5,:].^2 +
        eltype(x)(2)*x[6,:].^2 
    return eltype(x)(10)*minimum(map(sqrt, dist)) + sum(sqrt, dist)/length(x)
    # return sum(abs2, dist)
end
∂KE∂q(q,j) = zeros(Float32, 2, 2)
∂PE∂q(q) =  [ -10f0*q[2]; 0f0 ]
∂H∂q(q,p) = eltype(q)(1/2)*sum([∂KE∂q(q,k)*p[k] for k=1:2])'*p + ∂PE∂q(q)
function mass_matrix(q)
    return diagm(Float32[0.1; 0.2])
    # return Float32[0.1 0; 0 0.2]
end
const input_matrix = Float32[-1; 1]
const input_matrix_perp = Float32[1 1]
function random_state(T::DataType)
    q1 = T(π/6)*T(2)*(rand(T) - T(0.5))
    q2 = T(π/6)*T(2)*(rand(T) - T(0.5))
    vcat(
        cos(q1),
        sin(q1),
        cos(q2),
        sin(q2),
        T(0.25)*T(2)*(rand(T) - T(0.5)),
        T(0.25)*T(2)*(rand(T) - T(0.5))
    )
end


NX = 6
Hd = EnergyFunction(Float32, NX, dynamics!, loss, dim_S1=[1,2], num_hidden_nodes=32)
Hd_quad = QuadraticEnergyFunction(Float32,
    NX, dynamics, loss, ∂KE∂q, ∂PE∂q, mass_matrix, input_matrix, input_matrix_perp, 
    dim_q=2, num_hidden_nodes=16, symmetric=!true
)
q = random_state(Float32)[1:4]
p = rand(Float32, 2)
x0 = Float32[cos(3), sin(3), cosd(0), sind(0), 0, 0]

circle_constraints(x) = begin
    map(1:2) do i
        isapprox(x[2*i-1]^2 + x[2*i]^2, 1f0, atol=1e-4, rtol=1e-4)
    end |> all
end

to_states(x::Vector) = begin
    vcat(
        atan(x[2], x[1]), 
        atan(x[4], x[3]),
        x[5],
        x[6]
    )
end
to_states(x::Matrix) = begin
    _pi = eltype(x)(π)
    q1     = atan.(x[2,:], x[1,:])
    q2     = atan.(x[4,:], x[3,:])
    q1dot  = view(x, 5, :)
    q2dot  = view(x, 6, :)
    q1revs = cumsum( -round.(Int, [0; diff(q1)]./_pi) )
    q2revs = cumsum( -round.(Int, [0; diff(q2)]./_pi) )
    q1 = q1 .+ q1revs*_pi
    q2 = q2 .+ q2revs*_pi
    return [q1'; q2'; q1dot'; q2dot']
end
get_q_samples(n::Int) = begin
    [random_state(Float32)[1:4] for _=1:n]
end

test_Hd_gradient() = begin
    x = random_state(Float32)
    hcat(
        ReverseDiff.gradient(
            x->Hd([cos(x[1]); sin(x[1]); cos(x[2]); sin(x[2]); x[3]; x[4]]), 
            [atan(x[2], x[1]); atan(x[4], x[3]); x[5]; x[6]]
        ),
        gradient(Hd)(x)[:]
    )
end

test_Hd_quad_gradient() = begin
    q = random_state(Float32)[1:4];
    p = rand(Float32, 2)
    hcat(
        ReverseDiff.gradient(
            x->Hd_quad([cos(x[1]); sin(x[1]); cos(x[2]); sin(x[2])], p), 
            [atan(q[2], q[1]); atan(q[4], q[3])]
        ),
        gradient(Hd_quad)(q, p)
    )
end

test_ke_gradient() = begin
    q = random_state(Float32)[1:4];
    p = rand(Float32, 2)
    hcat(
        ReverseDiff.gradient(
            x->p'*Hd_quad.Md_inv([cos(x[1]); sin(x[1]); cos(x[2]); sin(x[2])])*p, 
            [atan(q[2], q[1]); atan(q[4], q[3])]
        ),
        MLBasedESC._ke_gradient(Hd_quad, q, p)
    )
end

test_pe_gradient() = begin
    q = random_state(Float32)[1:4];
    hcat(
        ReverseDiff.gradient(
            x->Hd_quad.Vd([cos(x[1]); sin(x[1]); cos(x[2]); sin(x[2])]), 
            [atan(q[2], q[1]); atan(q[4], q[3])]
        ),
        MLBasedESC._pe_gradient(Hd_quad, q) |> vec
    )
end

surf_Vd() = begin
    x = y = range(-pi, pi, length=20)
    surface(
        x, y,
        (x,y) -> Hd_quad.Vd( [cos(x[1]); sin(x[1]); cos(y[1]); sin(y[1])] )[1]
    )
end

surf_Md() = begin
    x = y = range(-pi, pi, length=20)
    surface(
        x, y,
        (x,y) -> cond( inv(Hd_quad.Md_inv( [cos(x[1]); sin(x[1]); cos(y[1]); sin(y[1])] )) )
    )
end

plot_traj(Hd_quad, x0::Vector, tf=Hd_quad.hyper.time_horizon) = begin
    num_plot_samples = 500
    old_tf = Hd_quad.hyper.time_horizon
    old_dt = Hd_quad.hyper.step_size
    Hd_quad.hyper.time_horizon = typeof(old_tf)(tf)
    Hd_quad.hyper.step_size = typeof(old_tf)(tf/num_plot_samples)
    t = range(0, Hd_quad.hyper.time_horizon, step=Hd_quad.hyper.step_size)
    x = predict(Hd_quad, x0)
    xbar = to_states(x[1:6,:])
    Hd_quad.hyper.time_horizon = old_tf 
    Hd_quad.hyper.step_size = old_dt
    plot(
        plot(t,xbar[1,:], label="q1"),
        plot(t,xbar[2,:], label="q2"),
        plot(t,xbar[3,:], label="q1dot"),
        plot(t,xbar[4,:], label="q2dot"),
        plot(t,x[7,:], label="u"),
        dpi=100, layout=(5,1)
    )
end
plot_traj(Hd_quad, tf=Hd_quad.hyper.time_horizon) = plot_traj(Hd_quad, random_state(Float32), tf)

train_ode(Hd::EnergyFunction; data=[random_state(Float32) for _=1:16], iters=100, verbose=true) = begin
    for i = 1:iters
        update!(Hd, data, verbose=verbose)
    end
    x0 = Float32[cos(3), sin(3), cosd(0), sind(0), 0, 0]
    plot_traj(Hd, x0, 30f0)
end

generate_training_samples(Hd::EnergyFunction) = begin
    # xf = Float32[cosd(5), sind(5), cosd(-5), sind(-5), -0.1, 0.1]
    horizon = 46f0
    xf = Float32[cosd(190), sind(190), cosd(-5), sind(-5), -0.1, 0.1]
    f = ODEFunction( (dx,x,p,t)->dynamics!(dx,x,zero(eltype(x))) )
    prob = ODEProblem(f, xf, (horizon, 0f0))
    sol = solve(prob, Tsit5(), saveat=horizon/500f0, rtol=1e-8, atol=1e-8)
    # plot(sol)
    x = to_states(Array(sol))
    tbar = reverse(sol.t)
    plot(
        plot(tbar,x[1,:], label="q1"),
        plot(tbar,x[2,:], label="q2"),
        plot(tbar,x[3,:], label="q1dot"),
        plot(tbar,x[4,:], label="q2dot"),
        dpi=100, layout=(4,1)
    )
end
