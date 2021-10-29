using MLBasedESC
using Test
using LinearAlgebra
using OrdinaryDiffEq

const USE_J2 = !true

function create_true_hamiltonian()
    I1 = 0.0455f0
    I2 = 0.00425f0
    m3 = 0.183f0*9.81f0
    mass_inv = inv(diagm(vcat(I1, I2)))
    pe(q) = m3*(cos(q[1]) - one(q[1]))
    # pe(q) = m3*(q[1] - one(q[1]))
    Hamiltonian(mass_inv, pe)
end

function input_jacobian(x)
    T = eltype(x)
    [-x[2] zero(T); x[1] zero(T); zero(T) -x[4]; zero(T) x[3]]
end

function create_learning_hamiltonian()
    massd_inv = PSDNeuralNetwork(Float32, 2, nin=2)
    vd = NeuralNetwork(Float32, [2,128,128,1], symmetric=!true, fout=x->x.^2, dfout=x->eltype(x)(2)*x)
    # vd = SOSPoly(2, 1:3)
    Hamiltonian(massd_inv, vd)
end

function create_ida_pbc_problem()
    input = vcat(-1.0f0,1.0f0)
    input_annihilator = hcat(1.0f0,1.0f0)
    ham = create_true_hamiltonian()
    hamd = create_learning_hamiltonian()
    if USE_J2
        J2 = InterconnectionMatrix(
            SkewSymNeuralNetwork(Float32, 2, nin=2),
            SkewSymNeuralNetwork(Float32, 2, nin=2)
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
    a1 = 1.0f0
    a2 = -1.1f0
    a3 = 2.0f0
    k1 = 0.0001f0
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
    # z(q) = atan(q[4],q[3]) + γ2*atan(q[2],q[1])
    # ped(q) = I1*m3/(a1+a2)*q[1] + ϕ(z(q))

    hamd = Hamiltonian(massd_inv, ped)
    prob = IDAPBCProblem(create_true_hamiltonian(), hamd, input, input_annihilator)
end


function assemble_data()
    data = Vector{Float32}[]
    dq = Float32(pi/30)
    qmax = Float32(pi)
    q1range = range(-qmax, qmax, step=dq)
    q2range = range(-qmax, qmax, step=dq)
    for q1 in q1range
        for q2 in q2range
            push!(data, [q1,q2])
        end
    end
    return data
end


function train()
    prob = create_ida_pbc_problem()
    data = assemble_data()
    θ = MLBasedESC.params(prob)
    qdesired = Float32[cos(0); sin(0); cos(0); sin(0)]
    solve_sequential!(prob, θ, data, qdesired, maxiters=20, η=0.001)
    return prob, θ
end

function to_augmented(q)
    return [cos(q[1]); sin(q[1]); cos(q[2]); sin(q[2])]
end
function to_angles(q)
    return [atan(q[2],q[1]), atan(q[4],q[3])]
end

unwrap(x::Matrix) = begin
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

function generate_trajectory(prob, x0, tf, θ=prob.init_params)
    policy = controller(prob, θ, damping_gain=10f0)
    M = prob.ham.mass_inv(0) |> inv
    # f(dx,x,p,t) = begin
    #    cq1, sq1, cq2, sq2, q1dot, q2dot = x
    #    I1 = 0.0455f0
    #    I2 = 0.00425f0
    #    m3 = 0.183f0*9.81f0
    #    ϵ  = 0.1f0
    #    b1 = b2 = 0.1f0
    #    effort = policy(x[1:4],M*x[5:6])
    #    dx[1] = -sq1*q1dot - ϵ*cq1*(sq1^2 + cq1^2 - 1)
    #    dx[2] =  cq1*q1dot - ϵ*sq1*(sq1^2 + cq1^2 - 1)
    #    dx[3] = -sq2*q2dot - ϵ*cq2*(sq2^2 + cq2^2 - 1)
    #    dx[4] =  cq2*q2dot - ϵ*sq2*(sq2^2 + cq2^2 - 1)
    #    dx[5] = m3*sq1/I1 - effort/I1 - b1/I1*q1dot
    #    dx[6] = effort/I2 - b2/I2*q2dot
    #end
    f(dx,x,p,t) = begin
        q1, q2, q1dot, q2dot = x
        momentum = M * [q1dot, q2dot]
        I1 = 0.0455f0
        I2 = 0.00425f0
        m3 = 0.183f0*9.81f0
        effort = policy(x[1:2],momentum)
        # effort = clamp(effort, -1f0, 1f0)
         b1 = b2 = 0.00f0
        # dx[1] = -sq1*q1dot - ϵ*cq1*(sq1^2 + cq1^2 - 1)
        # dx[2] =  cq1*q1dot - ϵ*sq1*(sq1^2 + cq1^2 - 1)
        # dx[3] = -sq2*q2dot - ϵ*cq2*(sq2^2 + cq2^2 - 1)
        # dx[4] =  cq2*q2dot - ϵ*sq2*(sq2^2 + cq2^2 - 1)
        dx[1] = x[3]
        dx[2] = x[4]
        dx[3] = m3*sin(q1)/I1 - effort/I1 - b1/I1*q1dot
        dx[4] = effort/I2 - b2/I2*q2dot
    end
    ode = ODEProblem(f, x0, (zero(tf), tf))
    sol = OrdinaryDiffEq.solve(ode, Tsit5(), saveat=tf/200)
    solu = transpose(Array(sol))
    ctrl = mapslices(x->1*u(x[1:2], inv(prob.ham.mass_inv(0))*x[3:4]), solu, dims=2)
    (sol.t, solu, ctrl)
    # (sol.t, transpose(unwrap(Array(sol))))
end
