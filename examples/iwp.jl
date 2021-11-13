using MLBasedESC
using LinearAlgebra
using OrdinaryDiffEq
using Plots; pyplot(display=false)

const USE_J2 = !true
const PRECISION = Float64
const I1 = PRECISION(0.0455)
const I2 = PRECISION(0.00425)
const m3 = PRECISION(0.183*9.81)
const b1 = PRECISION(0.005)
const b2 = PRECISION(0.010)

function create_true_hamiltonian()
    mass_inv = inv(diagm(vcat(I1, I2)))
    pe(q) = m3*(cos(q[1])-one(eltype(q)))
    # pe(q) = begin
    #     # qbar = input_mapping(q)
    #     # return -m3*qbar[1]
    #     return -m3*q[1]
    # end
    Hamiltonian(mass_inv, pe)
end

function input_mapping(x) 
    # [
    #     one(eltype(x))-cos(x[1]); 
    #     sin(x[1]); 
    #     one(eltype(x))-cos(x[2]); 
    #     sin(x[2])
    # ]
    # [
    #     one(eltype(x))-cos(x[1]); 
    #     sin(x[1])
    #     x[2]
    # ]
    # [x[1], one(eltype(x))-cos(x[2]), sin(x[2])]
    identity(x)
end
function input_jacobian(x)
    """
    Input mapping f(x) = [1-cos(x1), sin(x1), 1-cos(x2), sin(x2)]
    This is Jₓf
    """
    T = eltype(x)
    # [
    #     x[2] zero(T); 
    #     one(T)-x[1] zero(T); 
    #     zero(T) x[4]; 
    #     zero(T) one(T)-x[3]
    # ]
    # [
    #     x[2] zero(T); 
    #     one(T)-x[1] zero(T)
    #     zero(T) one(T); 
    # ]
    # [
    #     one(T) zero(T)
    #     zero(T) x[3]
    #     zero(T) one(T)-x[2]
    # ]
    one(T)
end

function create_learning_hamiltonian()
    # massd_inv = PSDNeuralNetwork(PRECISION, 2, 3, nin=2, num_hidden_nodes=8)
    massd_inv = PSDMatrix(PRECISION,2,2)
    # a1,a2,a3 = PRECISION.((0.001, -0.002, 0.005))
    # massd = [a1 a2; a2 a3]
    # massd_inv = inv(massd)
    vd = NeuralNetwork(PRECISION, [2,16,48,1], symmetric=true, fout=x->x.^2, dfout=x->2x)
    # vd = SOSPoly(4, 1:1)
    Hamiltonian(massd_inv, vd)
end

function create_partial_learning_hamiltonian()
    # a1,a2,a3 = PRECISION.(0.001, -0.002, 0.005)
    # massd = [a1 a2; a2 a3]
    # massd_inv = inv(massd)
    massd_inv = PSDMatrix(PRECISION,2,3)
    vd = SOSPoly(3, 1:1) # PRECISION[-6.3120513, 2.4831183f-5, -0.00027178923, -4.817491f-7, 0.0051580914, -0.0009012511, -0.0027414176, -5.4961457, 3.1027546, 4.6f-43]
    # vd = IWPSOSPoly() # PRECISION[6.389714, 0.0, 0.2223909, 0.0, 0.006149394, 0.0049287872, 0.0, 0.0, 0.0, -0.30627432]
    Hamiltonian(massd_inv, vd, input_jacobian)
end

function create_ida_pbc_problem()
    input = PRECISION[-1.0,1.0]
    input_annihilator = PRECISION[1.0 1.0]
    ham = create_true_hamiltonian()
    hamd = create_learning_hamiltonian()
    if USE_J2
        J2 = InterconnectionMatrix(
            SkewSymNeuralNetwork(PRECISION, 2, nin=4),
            SkewSymNeuralNetwork(PRECISION, 2, nin=4)
        )
        p = IDAPBCProblem(ham, hamd, input, input_annihilator, J2)
    else
        p = IDAPBCProblem(ham, hamd, input, input_annihilator)
    end
    init_Md = diagm([I2,I1])
    # init_Md = [0.001 0.0; 0.0 0.002]
    set_constant_Md!(p, p.init_params, init_Md)
    return p
end

function create_known_ida_pbc()
    a1,a2,a3 = PRECISION.((0.001, -0.002, 0.005))
    k1 = PRECISION(0.1)
    γ2 = -I1*(a2+a3)/(I2*(a1+a2))
    input = PRECISION[-1.0,1.0]
    input_annihilator = PRECISION[1.0 1.0]
    
    mass_inv = inv(diagm(vcat(I1, I2)))
    pe(q) = m3*(cos(q[1]) - one(q[1]))
    ham = Hamiltonian(mass_inv, pe)

    massd = [a1 a2; a2 a3]
    massd_inv = inv(massd)
    ϕ(z) = 0.5*k1*z^2
    z(q) = q[2] + γ2*q[1]
    ped(q) = I1*m3/(a1+a2)*cos(q[1]) + ϕ(z(q))
    hamd = Hamiltonian(massd_inv, ped)

    prob = IDAPBCProblem(ham, hamd, input, input_annihilator)
end


function assemble_data(;input_mapping::Function=input_mapping, dq=pi/20, qmax=(pi, pi))
    T = PRECISION
    data = Vector{T}[]
    q1max = first(qmax)
    q2max = last(qmax)
    q1range = range(0, q2max, step=dq)
    q2range = range(0, q2max, step=dq)
    for q1 in q1range
        for q2 in q2range
            push!(data, input_mapping(T[q1,q2]))
        end
    end
    return data
end

function generate_trajectory(prob, x0, tf, θ=prob.init_params; umax=Inf, Kv=1.0, uscale=1.0)
    T = eltype(x0)
    M = diagm(vcat(I1,I2))
    policy = controller(prob, θ, damping_gain=T(Kv))
    u(q,p) = clamp(policy(q,p), -umax, umax)
    f!(dx,x,p,t) = begin
        q1, q2, q1dot, q2dot = x
        qbar = input_mapping(x[1:2])
        momentum = M * [q1dot, q2dot]
        effort = T(uscale)*u(qbar,momentum)
        dx[1] = q1dot
        dx[2] = q2dot
        dx[3] = m3*sin(q1)/I1 - effort/I1 - b1/I1*q1dot
        dx[4] = effort/I2 - b2/I2*q2dot
    end
    ode = ODEProblem{true}(f!, x0, (zero(tf), tf))
    sol = OrdinaryDiffEq.solve(ode, BS5(), saveat=tf/1000)
    ctrl = mapslices(x->u(input_mapping(x[1:2]), M*x[3:4]), Array(sol), dims=1) |> vec
    (sol, ctrl)
end
function generate_true_trajectory(prob, x0, tf, umax=Inf; ps=(3.0, 30.0, 0.001, 0.1))
    T = eltype(x0)
    M = diagm(vcat(I1,I2))
    # policy = controller(prob, θ, damping_gain=Kv)
    # u(q,p) = clamp(policy(q,p), -umax, umax)
    ps = T.(ps)
    f!(dx,x,p,t) = begin
        q1, q2, q1dot, q2dot = x
        qbar = input_mapping(x[1:2])
        momentum = M * [q1dot, q2dot]

        γ1, γ2, kp, kv = ps
        qhat = q2 + γ2*q1
        qhatdot = q2dot + γ2*q1dot
        effort = γ1*sin(q1) + kp*(qhat) - kv*qhatdot
        effort = clamp(effort, -umax, umax)

        dx[1] = q1dot
        dx[2] = q2dot
        dx[3] = m3*sin(q1)/I1 - effort/I1 - b1/I1*q1dot
        dx[4] = effort/I2 - b2/I2*q2dot
    end
    ode = ODEProblem{true}(f!, x0, (zero(tf), tf))
    sol = OrdinaryDiffEq.solve(ode, BS5(), reltol=1e-6, abstol=1e-8, saveat=tf/200)
    # ctrl = mapslices(x->u(input_mapping(x[1:2]), M*x[3:4]), Array(sol), dims=1) |> vec
    # (sol, ctrl)
end

function test_hamd_gradient()
    θ = PRECISION[pi*rand(), pi*rand()]
    p = rand(PRECISION, 2)
    q = input_mapping(θ)
    isapprox(
        ReverseDiff.gradient(x->prob.hamd(input_mapping(x),p), θ),
        gradient(prob.hamd, q, p),
        atol=1e-4
    )
end

get_Vd(prob,ps) = (θ)->prob.hamd.potential(input_mapping(θ), getindex(ps, prob.ps_index[:potential]))
get_Md(prob,ps) = (θ)->inv(prob.hamd.mass_inv(input_mapping(θ), getindex(ps, prob.ps_index[:mass_inv])))

function bilinear_alternation()
    T = Float64

    input_mapping(x) = begin 
        [
            one(eltype(x))-cos(x[1])
            sin(x[1])
            # one(eltype(x))-cos(x[2])
            # sin(x[2])
            # x[1]
            x[2]
        ] 
    end
    input_jacobian(x) = begin
        T = eltype(x)
        [
            x[2] zero(T); 
            one(T)-x[1] zero(T); 
            # zero(T) x[4]; 
            # zero(T) one(T)-x[3]; 
            # one(T) zero(T)
            zero(T) one(T); 
        ] 
    end
    data = assemble_data(input_mapping=input_mapping, dq=T(pi/20));
        
    input = T.(vcat(-1.0,1.0))
    input_annihilator = T.(hcat(1.0,1.0))

    mass_inv = T.(inv(diagm(vcat(I1, I2))))
    pe(q) = begin
        return -m3*q[1]
    end
    ham = Hamiltonian(mass_inv, pe, input_jacobian)

    n = length(rand(data))
    massd_inv = PSDMatrix(T,2,n)
    Vd = SOSPoly(n, 1:2)
    hamd = Hamiltonian(massd_inv, Vd, input_jacobian)

    prob =  IDAPBCProblem(ham, hamd, input, input_annihilator);
    θ = T.(MLBasedESC.params(prob));
    l1 = ConvexVdLoss(prob, input_jacobian);
    l2 = ConvexMdLoss(prob);

    set_constant_Md!(prob, θ, [0.001 -0.002; -0.002 0.005])
    # set_constant_Md!(prob, θ, [1.0 -2; -2 5])
    # set_constant_Md!(prob, θ, 1.0*T.(collect(I(2))))
    # set_constant_Md!(prob, θ, [0.01 -0.002; -0.002 0.001])
# 
    # Vd_trained_params = T[-6.3120513, 2.4831183f-5, -0.00027178923, -4.817491f-7, 0.0051580914, -0.0009012511, -0.0027414176, -5.4961457, 3.1027546, 4.6f-43];
    # θ[prob.ps_index[:potential]] = Vd_trained_params;
    # optimize!(l2, θ, data);
    # optimize!(l1, θ, data);
    # optimize!(l2, θ, data);
    # optimize!(l1, θ, data);

    optimize!(l1, θ, data);
    optimize!(l2, θ, data);
    optimize!(l1, θ, data);
    optimize!(l2, θ, data);
    optimize!(l1, θ, data);
    optimize!(l2, θ, data);
    optimize!(l1, θ, data);
    optimize!(l2, θ, data);
    

    get_Vd(prob,ps) = (θ)->prob.hamd.potential(input_mapping(θ), getindex(ps, prob.ps_index[:potential]))
    get_Md(prob,ps) = (θ)->inv(prob.hamd.mass_inv(input_mapping(θ), getindex(ps, prob.ps_index[:mass_inv])))

    return prob, θ, get_Vd(prob, θ)
end
