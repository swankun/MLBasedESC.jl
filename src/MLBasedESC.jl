module MLBasedESC

using Base: front, tail
using LinearAlgebra
using Revise
using Printf

import Flux
import Flux.Zygote
using Flux: Chain, Dense, elu

import DiffEqFlux
using DiffEqFlux: FastChain, FastDense

using DynamicPolynomials

import ReverseDiff

using OrdinaryDiffEq, SciMLBase

include("fluxhelpers.jl")
include("idapbc.jl")
include("soslayers.jl")
include("constants.jl")
include("systems.jl")
include("neuralpbc.jl")
include("optimize.jl")

end
