using MLBasedESC
using LinearAlgebra
using Plots; pyplot(size=(1268,734))
using ReverseDiff
using Symbolics

using MeshCat
using CoordinateTransformations, Rotations
using GeometryBasics: Sphere, Cylinder, HyperRectangle, Vec, Point3f0
using Colors: RGBA, RGB
using Blink

function mass_matrix(q)
    return diagm(eltype(q)[1; 10f0^2 + q[1]^2])
end

function ham(q,p)
    eltype(q)(1/2)*dot(p, inv(mass_matrix(q))*p) + eltype(q)(9.81)*sin(q[2])*q[1]
end

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


const input_matrix = Float32[0; 1]
const input_matrix_perp = Float32[1 0]

function true_control(x,θ)
    M  = mass_matrix
    G  = input_matrix
    L  = 10f0
    g  = 9.81f0
    kp = 1f0
    kv = 50f0

    Mdi(q)  = (L^2 + q[1]^2) * [sqrt(2f0/(L^2+q[1]^2)) 1f0; 1f0 sqrt(2f0*(L^2+q[1]^2))] |> inv
    # Mdi(q) = Hd_quad.Md_inv(q)
    Vd(q)   = g*(1f0-cos(q[2])) + 0.5f0*kp*( q[2] - 1f0/sqrt(2)*asinh(q[1]/L) )^2
    Hd(q,p) = 0.5f0 * dot(p, Mdi(q)*p) + Vd(q)
    ∇q_Hd(q,p) = ReverseDiff.gradient(x->Hd(x,p), q)

    q = x[1:2]
    p = x[3:end] .* diag(M(q))
    j   = q[1]*(p[1] - sqrt(2f0/(L^2+q[1]^2))*p[2])
    J2  = [0f0 j; -j 0f0]
    Gu_es = ∂H∂q(q, p) .- (M(q) * Mdi(q)) \ ∇q_Hd(q, p) .+ J2*Mdi(q)*p
    u_di = -kv*dot(G, 2f0*Mdi(q)*p)
    return dot( (G'*G)\G', Gu_es ) + u_di
end

function random_state(T::DataType)
    q1 = T(5)*T(2)*(rand(T) - T(0.5))
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
    num_hidden_nodes=16, symmetric=true
)
x0 = Float32[4,0,1,1]

surf_Vd(type=:surface; n=20, kwargs...) = begin
    Q1 = range(-10, 10, length=n)
    Q2 = range(-0.6, 0.6, length=n)
    plot(
        Q1, Q2,
        (x,y) -> Hd_quad.Vd( Float32[x; y] )[1],
        st=type;
        kwargs...
    )
end

surf_Md(type=:surface; n=20, kwargs...) = begin
    Q1 = range(-10, 10, length=n)
    Q2 = range(-0.6, 0.6, length=n)
    plot(
        Q1, Q2,
        (x,y) -> cond( inv(Hd_quad.Md_inv( Float32[x; y] )) ),
        st=type;
        kwargs...
    )
end

function plot_traj(Hd; x0=Float32[8,0,1,1], tf=Hd_quad.hyper.time_horizon)
    num_plot_samples = 500
    old_tf = Hd_quad.hyper.time_horizon
    old_dt = Hd_quad.hyper.step_size
    Hd_quad.hyper.time_horizon = typeof(old_tf)(tf)
    Hd_quad.hyper.step_size = typeof(old_tf)(tf/num_plot_samples)
    t = range(0, Hd_quad.hyper.time_horizon, step=Hd_quad.hyper.step_size)
    # x = predict(Hd_quad, x0, u=true_control)
    x = predict(Hd_quad, x0)
    Hd_quad.hyper.time_horizon = old_tf 
    Hd_quad.hyper.step_size = old_dt
    plot(
        plot(t,x[1,:], label="q1"),
        plot(t,x[2,:], label="q2"),
        plot(t,x[3,:], label="q1dot"),
        plot(t,x[4,:], label="q2dot"),
        plot(t,x[5,:], label="u"),
        dpi=100, layout=(5,1), show=true
    )
end

true_Vd(q, kp=1f0) = begin
    g = 9.81f0
    L = 10f0
    a = g*(1-cos(q[2]))
    b = kp/2 * (q[2] - 1/sqrt(2)*asinh(q[1]/L))^2
    return a + b
end


function animate(x, vis)
    q1, q2, p1, p2, u = eachrow(x)
    L = eltype(x)(10)
    red_material = MeshPhongMaterial(color=RGBA(252/255, 206/255, 0.0, 0.7))
    green_material = MeshPhongMaterial(color=RGBA(255/255, 124/255, 59/255, 1.0))
    beam_radius = 0.025f0
    ball_radius = 0.125f0

    trans = Translation(6, 0, 0)
    rot = LinearMap(RotY(deg2rad(5)))
    composed = trans ∘ rot
    settransform!(vis["/Cameras/default"], composed)
    
    beam = HyperRectangle(Vec(-L/10,-L/2,-beam_radius/2), Vec(L/5,L,beam_radius))#Cylinder(Point3f0(0,-L/2,0), Point3f0(0,L/2,0), beam_radius)
    ball = Sphere(Point3f0(0,0,beam_radius/2+ball_radius), ball_radius)

    delete!(vis)
    setobject!(vis["links"]["beam"], beam, red_material)
    setobject!(vis["links"]["ball"], ball, green_material)
    
    ball_pos(q1, q2) = Float32[0, q1*cos(q2), q1*sin(q2)]
    settransform!(vis["links"]["beam"], 
        LinearMap(AngleAxis(q2[1], 1, 0, 0))
    )
    settransform!(vis["links"]["ball"], 
        Translation(ball_pos(q1[1], q2[1]))
    )
    
    anim = MeshCat.Animation()
    step = max(1, round(Int, 1/20/Hd_quad.hyper.step_size))
    for (frame, i) in enumerate(1:step:length(q1))
        atframe(anim, frame-1) do
            settransform!(vis["links"]["beam"], 
                LinearMap(AngleAxis(q2[i], 1, 0, 0))
            )
            settransform!(vis["links"]["ball"], 
                Translation(ball_pos(q1[i], q2[i]))
            )
        end
    end
    setanimation!(vis, anim) 

end

function lsq_Vd()
    Vd = Hd_quad.Vd
    # Vd = SymmetricSOSPoly(2, 4)
    # Vd = SOSPoly(2,2)
    mon = Vd.mono
    vars = Vd.vars
    dmon = MLBasedESC.differentiate(mon, vars)
    
    @variables θ[1:length(Vd.θ)]
    @variables q[1:2]
    L = MLBasedESC.coeff_matrix(Vd, θ)
    R = L + L'
    R = R - Diagonal(diag(R) / 2)
    X = reduce(vcat, m(q[1], q[2]) for m in mon)
    dX = reshape(reduce(vcat, m(q[1], q[2]) for m in dmon), (size(L,1), 2))
    sos = transpose(dX)*R*X
    J = Symbolics.jacobian(sos, θ)
    Jf = build_function(J, q, expression=Val{false})[1]
    
    # Generate data
    T = Float32
    data = Vector{T}[]
    qmax = 1f0;
    qmin = -qmax
    q1range = range(-10f0, 10f0, length=201)
    q2range = range(qmin, qmax, length=21)
    for q1 in q1range
        for q2 in q2range
            push!(data, [q1; q2])
        end
    end
    A = Matrix{T}(undef, length(data), length(θ))
    B = Vector{T}(undef, length(data))
    for (i,point) in enumerate(data)
        A[i, :] = [1 0]*( 2 * inv( mass_matrix(point) * Hd_quad.Md_inv(point) ) * Jf(point) )
        B[i] = ([1 0]*∂PE∂q(point))[1]
    end
    A, B
    # res = A \ B
    # V = substitute.(R, (Dict(x => r for (x,r) in zip(θ,res)),))
    # res, T.(Symbolics.value.(V))
end

using JuMP, MosekTools
function convex_solve_Vd(A, B)
    Vd = Hd_quad.Vd
    m = length(Vd.mono)
    model = Model(Mosek.Optimizer)
    JuMP.@variable(model, R[1:m, 1:m], PSD)
    # denseidx = zip(Vd.j, Vd.i)
    # for i=1:m
    #     for j=1:m
    #         if !((i,j) in denseidx) && !((j,i) in denseidx)
    #             @constraint(model, R[i,j] == 0.0)
    #             # M[i,j] = 1
    #         end
    #     end
    # end
    
    denseidx = filter(collect(Iterators.product(1:m, 1:m))) do x
        first(x) <= last(x)
    end
    Rvec = [R[CartesianIndex(i,j)] for (i,j) in denseidx]

    x = A*Rvec - B
    JuMP.@variable(model, t)
    @constraint(model, [t; x] in SecondOrderCone())
    @objective(model, Min, t)
    optimize!(model)
    res = value.(R)
    L = cholesky(res).U
    θ = [L[CartesianIndex(i,j)] for (i,j) in denseidx]
    θ, res
end