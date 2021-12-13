module MLBasedESC

using Base: front, tail
using LinearAlgebra
using Revise

import Flux
using Flux: Chain, Dense, elu

import DiffEqFlux
using DiffEqFlux: FastChain, FastDense

using DynamicPolynomials

using OrdinaryDiffEq, SciMLBase

include("fluxhelpers.jl")
include("idapbc.jl")
include("soslayers.jl")
include("constants.jl")
include("systems.jl")

end
