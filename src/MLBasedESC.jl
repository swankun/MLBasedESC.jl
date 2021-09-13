module MLBasedESC

using ReverseDiff
using Flux: Chain, Dense, relu, glorot_uniform, ADAM, throttle, Data, Optimise
using Zygote

using OrdinaryDiffEq
using DiffEqFlux
import DiffEqFlux: FastChain, FastDense, initial_params
using DiffEqSensitivity

using GalacticOptim

using LinearAlgebra
using Random: randperm
using Revise
using SparseArrays

using Printf      
using Formatting
using Random

using DynamicPolynomials

include("utils.jl")
include("neural_net.jl")
include("sos.jl")
include("energy.jl")
include("energy_quadratic.jl")
include("hamiltonian.jl")

end
