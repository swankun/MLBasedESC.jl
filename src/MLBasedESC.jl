module MLBasedESC

# using ReverseDiff
# using Flux
# import Flux: Chain, Dense, relu, glorot_uniform, ADAM, throttle, Data, Optimise
# using Zygote

# using LinearAlgebra
# using Random: randperm
using Revise
# using SparseArrays

# using Convex, Mosek

# using Printf      
# using Formatting
# using Random

using DynamicPolynomials

# include("utils.jl")
# include("neural_net.jl")
# include("sos.jl")
# include("hamiltonian.jl")
# include("ida_pbc.jl")
# include("loss.jl")
# include("constants.jl")
# include("systems/abstractsys.jl")
# include("systems/iwp.jl")

include("fluxhelpers/fluxhelpers.jl")
include("idapbc.jl")
include("soslayers.jl")

end
