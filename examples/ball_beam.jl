using MLBasedESC
using LinearAlgebra
using Plots; pyplot()
using ReverseDiff

function dynamics(x, u)
    q1, q2, q1dot, q2dot = x
    g = eltype(x)(9.81)
    L = eltype(x)(10)
    ẋ1 = -g*sin(q2) + q1*q2dot^2
    ẋ2 = ( u - eltype(x)(2)*q1*q1dot*q2dot - g*q1*cos(q2) ) / ( L^2 + q1^2 )
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

Hd_quad = QuadraticEnergyFunction(Float32,
    4, dynamics, loss, ∂KE∂q, ∂PE∂q, mass_matrix, input_matrix, input_matrix_perp, 
    dim_q=2, num_hidden_nodes=16, symmetric=!true
)