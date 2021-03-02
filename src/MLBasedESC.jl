module MLBasedESC

using Flux: Chain, Dense, elu, glorot_uniform, ADAM, throttle
using Zygote: bufferfrom, @nograd

using OrdinaryDiffEq
using DiffEqFlux
using DiffEqSensitivity

using LinearAlgebra
using Random: randperm
using Revise

# using Printf      will replace Formatting when Julia 1.6 is out
using Formatting

include("utils.jl")
include("neural_net.jl")
include("energy.jl")
include("energy_quadratic.jl")

export NeuralNetwork, ∇NN, get_weights, get_biases
export EnergyFunction, QuadraticEnergyFunction, HyperParameters
export controller, update!, predict, params_to_npy
export mass_matrix

end
