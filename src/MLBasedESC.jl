module MLBasedESC

using Flux: Chain, Dense, elu, glorot_uniform, ADAM
using Zygote: bufferfrom, @nograd

using OrdinaryDiffEq
using DiffEqFlux
using DiffEqSensitivity

using Random: randperm
using Revise

include("utils.jl")
include("neural_net.jl")
include("energy.jl")

export NeuralNetwork, ∇NN, get_weights, get_biases
export EnergyFunction, HyperParameters, update!, forward, params_to_npy

end
