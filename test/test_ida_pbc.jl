using MLBasedESC
using Test
using LinearAlgebra

function create_true_hamiltonian()
    I1 = 0.1
    I2 = 0.2
    m3 = 10.0
    mass_inv = inv(diagm(vcat(I1, I2)))
    # pe(q) = m3*(cos(q[1]) - one(q[1]))
    pe(q) = m3*(q[1] - one(q[1]))
    Hamiltonian(mass_inv, pe)
end

function input_jacobian(x)
    T = eltype(x)
    [-x[2] zero(T); x[1] zero(T); zero(T) -x[4]; zero(T) x[3]]
end

function create_learning_hamiltonian()
    massd_inv = PSDNeuralNetwork(Float64, 2, nin=4)
    vd = NeuralNetwork(Float64, [4,8,8,1])
    Hamiltonian(massd_inv, vd, input_jacobian)
end

function create_ida_pbc_problem()
    input = vcat(-1.0,1.0)
    input_annihilator = hcat(1.0,1.0)
    ham = create_true_hamiltonian()
    hamd = create_learning_hamiltonian()
    J2 = InterconnectionMatrix(
        SkewSymNeuralNetwork(Float32, 2, nin=4),
        SkewSymNeuralNetwork(Float32, 2, nin=4)
    )
    prob = IDAPBCProblem(ham, hamd, input, input_annihilator, J2)
end

function create_known_ida_pbc()
    I1 = 0.1
    I2 = 0.2
    m3 = 10.0
    a1 = 1.0
    a2 = 1/2
    a3 = 2.0
    k1 = 1.0
    γ2 = -I1*(a2+a3)/(I2*(a1+a2))
    input = vcat(-1.0,1.0)
    input_annihilator = hcat(1.0,1.0)
    
    massd = [a1 a2; a2 a3]
    massd_inv = inv(massd)
    ϕ(z) = 0.5*k1*z^2
    z(q) = q[2] + γ2*q[1]
    pe(q) = I1*m3/(a1+a2)*cos(q[1]) + ϕ(z(q))

    ham = create_true_hamiltonian()
    hamd = Hamiltonian(massd_inv, pe)
    prob = IDAPBCProblem(ham, hamd, input, input_annihilator)
end

function assemble_data()
    data = Vector{Float32}[]
    dq = pi/10
    qmax = pi
    q1range = range(-qmax, qmax, step=dq)
    q2range = range(-qmax, qmax, step=dq)
    for q1 in q1range
        for q2 in q2range
            push!(data, [cos(q[1]), sin(q[1]), cos(q[2]), sin(q[2])])
        end
    end
    return data
end
