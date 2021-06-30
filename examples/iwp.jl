using MLBasedESC
using LinearAlgebra
using Plots
pyplot(
    guidefontsize=28,
    labelfontsize=24,
    tickfontsize=26,
    titlefontsize=30,
    fontfamily="cmuserif",
)
Plots.pyrcparams["text.usetex"]="true"
Plots.pyrcparams["mathtext.fontset"]="cm"
using LaTeXStrings
using ReverseDiff

const Kt = 1/(60.3e-3)
const r = 45/34*13/3        # Gear and belt ratio
const mp = 0.706
# const mr = 0.084 + 4 * 0.12946    # m_resin + 2*m_steel = 0.084 + 2 * 0.12946
const mr = 0.223
const m = mp + mr
const lr = 0.18217
const lp = lr/2
const l = (mp*lp + mr*lr)/m
const g = 9.81
const Ip_com = 0.00249
const I_epoxy = 1100 * 7.919e4 / 1e9 * (0.18/2)^2
const I_steel = 9.484e-4
const I_resin = 7.559e-4
const I_resin_new = 4.64e-4
const I1 = 0.1# Ip_com + mp*lp^2 + mr*lr^2
# const I2 = I_resin_new + 4*I_steel
const I2 = 0.2#I_resin + I_epoxy
const τ_max = 0.1
const γ1 = 2. # 1.5*ceil(m*g*l)
const γ2 = 25. # 3*ceil(I1/I2) * γ1 / (γ1 - m*g*l)
const k = (1.0, 10.0)

function dynamics(x, u)
    cq1, sq1, cq2, sq2, q1dot, q2dot = x
    m3 = 10f0#m*g*l
    ϵ  = 0.1f0
    b1 = b2 = 0.0f0
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
    m3 = 10f0#m*g*l
    ϵ  = 0.1f0
    b1 = b2 = 0.0f0
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
    dist = sqrt.(dist)
    return eltype(x)(10)*minimum(dist) + sum(dist)/length(x)
    # return sum(abs2, dist)
end
∂KE∂q(q,j) = zeros(Float32, 2, 2)
∂PE∂q(q) =  [ -10f0*q[2]; 0f0 ]
∂H∂q(q,p) = eltype(q)(1/2)*sum([∂KE∂q(q,k)*p[k] for k=1:2])'*p + ∂PE∂q(q)
function mass_matrix(q)
    return diagm(Float32[I1; I2])
    # return Float32[0.1 0; 0 0.2]
end
const input_matrix = Float32[-1; 1]
const input_matrix_perp = Float32[1 1]
function random_state(T::DataType)
    q1 = T(π)*T(2)*(rand(T) - T(0.5))
    q2 = T(π)*T(2)*(rand(T) - T(0.5))
    vcat(
        cos(q1),
        sin(q1),
        cos(q2),
        sin(q2),
        T(0.25)*T(2)*(rand(T) - T(0.5)),
        T(0.25)*T(2)*(rand(T) - T(0.5))
    )
end


NX=6
dim_S1=[1,2]
Hd = EnergyFunction(Float32, NX, dynamics!, loss, dim_S1=dim_S1, num_hidden_nodes=32)
Hd_quad = QuadraticEnergyFunction(Float32,
    NX, dynamics, loss, ∂KE∂q, ∂PE∂q, mass_matrix, input_matrix, input_matrix_perp, 
    dim_S1=dim_S1, num_hidden_nodes=32, symmetric=false
)
q = random_state(Float32)[1:4]
p = rand(Float32, 2)
x0 = Float32[cos(3), sin(3), cos(0), sin(0), 0, 0]

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
    x = y = range(-pi, pi, length=50)
    plot(
        x, y,
        (x,y) -> Hd_quad.Vd( [cos(x[1]); sin(x[1]); cos(y[1]); sin(y[1])] )[1],
        st=:contourf,
        c=:coolwarm,
        xlabel=L"q_1",
        ylabel=L"q_2"
    )
end

surf_Hd() = begin
    x = y = range(-pi, pi, length=50)
    plot(
        x, y,
        (x,y) -> Hd_quad( [cos(x[1]); sin(x[1]); cos(y[1]); sin(y[1])], [0f0, 0f0] )[1],
        st=:contourf,
        c=:coolwarm,
        xlabel=L"q_1",
        ylabel=L"q_2",
        # title=L"H_d^{\theta}(q,0)"
    )
end

surf_control() = begin
    u = controller(Hd_quad)
    x = y = range(-pi, pi, length=50)
    plot(
        x, y,
        (x,y) -> 0.1f0*u( [cos(x[1]); sin(x[1]); cos(y[1]); sin(y[1]); 0f0; 0f0] )[1],
        st=:contourf,
        c=:coolwarm,
        xlabel=L"q_1",
        ylabel=L"q_2",
        # title=L"u_\theta(q,0)"
    )
end

surf_Md() = begin
    x = y = range(-pi, pi, length=20)
    surface(
        x, y,
        (x,y) -> cond( inv(Hd_quad.Md_inv( [cos(x[1]); sin(x[1]); cos(y[1]); sin(y[1])] )) )
    )
end

surf_loss() = begin
    x = y = range(-pi, pi, length=50)
    plot(
        x, y,
        (x,y) -> pde_loss(Hd_quad, [cos(x[1]); sin(x[1]); cos(y[1]); sin(y[1])]),
        # levels=[0.1,1,10],
        st=:contourf,
        c=:coolwarm,
        xlabel=L"q_1",
        ylabel=L"q_2"
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
    u = controller(Hd_quad)
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
    x0 = Float32[cos(3), sin(3), cos(0), sin(0), 0, 0]
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

function publish_plot()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    merge!(
        rcParams,
        Dict(
            "patch.edgecolor" => "white",
            "text.color" => "black",
            "axes.facecolor" => "white",
            "axes.edgecolor" => "black",
            "axes.labelcolor" => "black",
            "xtick.color" => "black",
            "ytick.color" => "black",
            "grid.color" => "black",
            "figure.facecolor" => "white",
            "figure.edgecolor" => "black",
            "savefig.facecolor" => "white",
            "savefig.edgecolor" => "black",
            "toolbar" => "toolbar2",
            "image.cmap" => "plasma",
            "text.usetex" => true,
            "font.family" => "serif",
            "font.serif" => "DejaVu Serif",
            "font.weight" => "bold",
            "font.size" => 24.0,
            "axes.titleweight" => "bold"
        )
    )
    pi_ticks = -5pi:pi:5pi
    pi_ticklabels = ["$(i)" * L"\pi" for i = -5:5]
    halfpi_ticks = -2.5pi:0.5pi:2.5pi
    halfpi_ticklabels = ["$(i)" * L"\pi" for i = -2.5:0.5:2.5]
    pi_axlim = [first(pi_ticks), last(pi_ticks)]

    num_plot_samples = 500
    tf = 28f0
    x0 = Float32[cos(3), sin(3), cosd(0), sind(0), 0, 0]
    old_tf = Hd_quad.hyper.time_horizon
    old_dt = Hd_quad.hyper.step_size
    Hd_quad.hyper.time_horizon = typeof(old_tf)(tf)
    Hd_quad.hyper.step_size = typeof(old_tf)(tf/num_plot_samples)
    t = range(0, Hd_quad.hyper.time_horizon, step=Hd_quad.hyper.step_size)
    x = predict(Hd_quad, x0)
    xbar = to_states(x[1:6,:])
    Hd_quad.hyper.time_horizon = old_tf
    Hd_quad.hyper.step_size = old_dt

    fig = PyPlot.figure(1)
    fig.clf()
    for i = 1:4
        fig.add_subplot(1,4,i)
    end
    fig.axes[1].plot(t, xbar[1,:], "k")
    fig.axes[2].plot(t, xbar[3,:], "k")
    fig.axes[3].plot(t, xbar[2,:], "k")
    fig.axes[4].plot(t, xbar[4,:], "k")

    fig.axes[1].set_yticks(halfpi_ticks)
    fig.axes[1].set_yticklabels(halfpi_ticklabels)
    fig.axes[1].set_ylim([-0.2pi, 2.2pi])
    fig.axes[3].set_yticks(-0.2pi:0.1pi:0.2pi)
    fig.axes[3].set_yticklabels(["$(i)" * L"\pi" for i = -0.2:0.1:0.2])
    fig.axes[2].set_ylim([-25., 25.])
    fig.axes[3].set_ylim([-0.12pi, 0.12pi])
    fig.axes[4].set_ylim([-0.52, 0.52])

    for i = 1:4
        fig.axes[i].set_xlim([-2.0, 25])
        fig.axes[i].set_xticks(0:5:25)
    end

    fig.axes[1].set_title(L"$q_1(t)$ (rad)")
    fig.axes[2].set_title(L"$\dot{q}_1(t)$ (rad/s)")
    fig.axes[3].set_title(L"$q_2(t)$ (rad)")
    fig.axes[4].set_title(L"$\dot{q}_2(t)$ (rad/s)")

    fig.show()
    fig.tight_layout()
end
