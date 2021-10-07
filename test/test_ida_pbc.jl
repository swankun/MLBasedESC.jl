using MLBasedESC
using Test
using LinearAlgebra

const USE_J2 = true

function create_true_hamiltonian()
    I1 = 0.1f0
    I2 = 0.2f0
    m3 = 10.0f0
    mass_inv = inv(diagm(vcat(I1, I2)))
    # pe(q) = m3*(cos(q[1]) - one(q[1]))
    pe(q) = m3*(q[1] - one(q[1]))
    Hamiltonian(mass_inv, pe, input_jacobian)
end

function input_jacobian(x)
    T = eltype(x)
    [-x[2] zero(T); x[1] zero(T); zero(T) -x[4]; zero(T) x[3]]
end

function create_learning_hamiltonian()
    massd_inv = PSDNeuralNetwork(Float32, 2, nin=4)
    # vd = NeuralNetwork(Float32, [4,128,128,1])
    vd = SOSPoly(4, 2)
    Hamiltonian(massd_inv, vd, input_jacobian)
end

function create_ida_pbc_problem()
    input = vcat(-1.0f0,1.0f0)
    input_annihilator = hcat(1.0f0,1.0f0)
    ham = create_true_hamiltonian()
    hamd = create_learning_hamiltonian()
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
    I1 = 0.1f0
    I2 = 0.2f0
    m3 = 10.0f0
    a1 = 1.0f0
    a2 = 0.5f0
    a3 = 2.0f0
    k1 = 1.0f0
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
    dq = Float32(pi/100)
    qmax = Float32(pi)
    q1range = range(-qmax, qmax, step=dq)
    q2range = range(-qmax, qmax, step=dq)
    for q1 in q1range
        for q2 in q2range
            push!(data, [cos(q1), sin(q1), cos(q2), sin(q2)])
        end
    end
    return data
end
