using MLBasedESC
using LinearAlgebra
using OrdinaryDiffEq
using Plots; pyplot(display=false)

const USE_J2 = !true

function create_true_hamiltonian()
    I1 = 0.0455f0
    I2 = 0.00425f0
    m3 = 0.183f0*9.81f0
    mass_inv = inv(diagm(vcat(I1, I2)))
    # pe(q) = m3*(cos(q[1])-one(eltype(q)))
    pe(q) = begin
        # qbar = input_mapping(q)
        # return -m3*qbar[1]
        return -m3*q[1]
    end
    Hamiltonian(mass_inv, pe, input_jacobian)
end

# input_mapping(x) = [
#     one(eltype(x))-cos(x[1]); 
#     sin(x[1]); 
#     one(eltype(x))-cos(x[2]); 
#     sin(x[2])
# ]
input_mapping(x) = [
    one(eltype(x))-cos(x[1]); 
    x[1]
    x[2]
    sin(x[1])
]

function input_jacobian(x)
    """
    Input mapping f(x) = [1-cos(x1), sin(x1), 1-cos(x2), sin(x2)]
    This is Jₓf
    """
    T = eltype(x)
    [
        x[4] zero(T); 
        one(T) zero(T); 
        zero(T) one(T); 
        one(T)-x[1] zero(T)
    ]
end

function create_learning_hamiltonian()
    massd_inv = PSDNeuralNetwork(Float32, 2, nin=4)
    vd = NeuralNetwork(Float32, [4,16,16,1], symmetric=!true, fout=x->x.^2, dfout=x->eltype(x)(2x))
    # vd = SOSPoly(4, 1:2)
    Hamiltonian(massd_inv, vd, input_jacobian)
end

function create_partial_learning_hamiltonian()
    a1,a2,a3 = (0.001f0, -0.002f0, 0.005f0)
    massd = [a1 a2; a2 a3]
    massd_inv = inv(massd)
    vd = SOSPoly(4, 1:1) # Float32[-6.3120513, 2.4831183f-5, -0.00027178923, -4.817491f-7, 0.0051580914, -0.0009012511, -0.0027414176, -5.4961457, 3.1027546, 4.6f-43]
    # vd = IWPSOSPoly() # Float32[6.389714, 0.0, 0.2223909, 0.0, 0.006149394, 0.0049287872, 0.0, 0.0, 0.0, -0.30627432]
    Hamiltonian(massd_inv, vd, input_jacobian)
end

function create_ida_pbc_problem()
    input = vcat(-1.0f0,1.0f0)
    input_annihilator = hcat(1.0f0,1.0f0)
    ham = create_true_hamiltonian()
    hamd = create_partial_learning_hamiltonian()
    if USE_J2
        J2 = InterconnectionMatrix(
            SkewSymNeuralNetwork(Float32, 2, nin=4),
            SkewSymNeuralNetwork(Float32, 2, nin=4)
        )
        return IDAPBCProblem(ham, hamd, input, input_annihilator, J2)
    else
        return IDAPBCProblem(ham, hamd, input, input_annihilator)
    end
end

function create_known_ida_pbc()
    I1 = 0.0455f0
    I2 = 0.00425f0
    m3 = 0.183f0*9.81f0
    a1,a2,a3 = (0.001f0, -0.002f0, 0.005f0)
    k1 = 0.1f0
    γ2 = -I1*(a2+a3)/(I2*(a1+a2))
    input = vcat(-1.0f0,1.0f0)
    input_annihilator = hcat(1.0f0,1.0f0)
    
    mass_inv = inv(diagm(vcat(I1, I2)))
    pe(q) = m3*(cos(q[1]) - one(q[1]))
    ham = Hamiltonian(mass_inv, pe)

    massd = [a1 a2; a2 a3]
    massd_inv = inv(massd)
    ϕ(z) = 0.5f0*k1*z^2
    z(q) = q[2] + γ2*q[1]
    ped(q) = I1*m3/(a1+a2)*cos(q[1]) + ϕ(z(q))
    hamd = Hamiltonian(massd_inv, ped)

    prob = IDAPBCProblem(ham, hamd, input, input_annihilator)
end


function assemble_data()
    data = Vector{Float32}[]
    dq = Float32(pi/20)
    qmax = Float32(pi)
    q1range = range(-qmax, qmax, step=dq)
    q2range = range(-qmax, qmax, step=dq)
    for q1 in q1range
        for q2 in q2range
            push!(data, input_mapping([q1,q2]))
        end
    end
    return data
end

function generate_trajectory(prob, x0, tf, θ=prob.init_params; umax=eltype(x0)(Inf), Kv=eltype(x0)(10))
    I1 = 0.0455f0
    I2 = 0.00425f0
    m3 = 0.183f0*9.81f0
    b1 = b2 = 0.02f0
    M = diagm(vcat(I1,I2))
    policy = controller(prob, θ, damping_gain=Kv)
    u(q,p) = clamp(policy(q,p), -umax, umax)
    f!(dx,x,p,t) = begin
        q1, q2, q1dot, q2dot = x
        qbar = input_mapping(x[1:2])
        momentum = M * [q1dot, q2dot]
        effort = u(qbar,momentum)
        dx[1] = q1dot
        dx[2] = q2dot
        dx[3] = m3*sin(q1)/I1 - effort/I1 - b1/I1*q1dot
        dx[4] = effort/I2 - b2/I2*q2dot
    end
    ode = ODEProblem{true}(f!, x0, (zero(tf), tf))
    sol = OrdinaryDiffEq.solve(ode, Tsit5(), saveat=tf/200)
    ctrl = mapslices(x->u(input_mapping(x[1:2]), M*x[3:4]), Array(sol), dims=1) |> vec
    (sol, ctrl)
end
function generate_true_trajectory(prob, x0, tf, umax=eltype(x0)(Inf); ps=(3f0, 30f0, 0.001f0, 0.1f0))
    I1 = 0.1f0#0.0455f0
    I2 = 0.2f0#0.00425f0
    m3 = 10f0#0.183f0*9.81f0
    b1 = b2 = 0.0f0
    M = diagm(vcat(I1,I2))
    # policy = controller(prob, θ, damping_gain=Kv)
    # u(q,p) = clamp(policy(q,p), -umax, umax)
    f!(dx,x,p,t) = begin
        q1, q2, q1dot, q2dot = x
        qbar = input_mapping(x[1:2])
        momentum = M * [q1dot, q2dot]

        γ1, γ2, kp, kv = ps
        qhat = q2 + γ2*q1
        qhatdot = q2dot + γ2*q1dot
        effort = γ1*sin(q1) + kp*sin(qhat) - kv*qhatdot
        effort = clamp(effort, -umax, umax)

        dx[1] = q1dot
        dx[2] = q2dot
        dx[3] = m3*sin(q1)/I1 - effort/I1 - b1/I1*q1dot
        dx[4] = effort/I2 - b2/I2*q2dot
    end
    ode = ODEProblem{true}(f!, x0, (zero(tf), tf))
    sol = OrdinaryDiffEq.solve(ode, Tsit5(), saveat=tf/200)
    # ctrl = mapslices(x->u(input_mapping(x[1:2]), M*x[3:4]), Array(sol), dims=1) |> vec
    # (sol, ctrl)
end

function test_hamd_gradient()
    θ = Float32[pi*rand(), pi*rand()]
    p = rand(Float32, 2)
    q = input_mapping(θ)
    isapprox(
        ReverseDiff.gradient(x->prob.hamd(input_mapping(x),p), θ),
        gradient(prob.hamd, q, p),
        atol=1e-4
    )
end

get_Vd(prob,ps) = (θ)->prob.hamd.potential(input_mapping(θ), ps)
get_Md(prob,ps) = (θ)->inv(prob.hamd.mass_inv(input_mapping(θ), ps))
