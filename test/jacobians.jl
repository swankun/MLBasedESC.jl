using MLBasedESC
using MLBasedESC: posdef, skewsym
using Flux
using LinearAlgebra
import DiffEqFlux
using DiffEqFlux: FastChain, FastDense

const NIN = 2

inmap(q,::Any=nothing) = [cos(q[1]); sin(q[1]); cos(q[2]); sin(q[2])]
function MLBasedESC.jacobian(::typeof(inmap), q,::Any=nothing)
    qbar = inmap(q)
    [
        -qbar[2] 0
         qbar[1] 0
         0 -qbar[4] 
         0  qbar[3] 
    ]
end

function runalltests()
    c_single_output()
    c_single_output_odd()
    c_single_output_inmap()
    c_single_output_nonneg()
    c_single_output_nonneg_odd()
    c_single_output_nonneg_inmap()
    fc_single_output()
    fc_single_output_odd()
    fc_single_output_inmap()
    fc_single_output_nonneg()
    fc_single_output_nonneg_odd()
    fc_single_output_nonneg_inmap()
    c_multi_output()
    c_multi_output_odd()
    c_multi_output_inmap()
    c_multi_output_nonneg()
    c_multi_output_nonneg_odd()
    c_multi_output_nonneg_inmap()
    fc_multi_output()
    fc_multi_output_odd()
    fc_multi_output_inmap()
    fc_multi_output_nonneg()
    fc_multi_output_nonneg_odd()
    fc_multi_output_nonneg_inmap()
    c_posdef_output()
    c_posdef_output_odd()
    c_posdef_output_inmap()
    c_posdef_output_nonneg()
    c_posdef_output_nonneg_odd()
    c_posdef_output_nonneg_inmap()
    fc_posdef_output()
    fc_posdef_output_odd()
    fc_posdef_output_inmap()
    fc_posdef_output_nonneg()
    fc_posdef_output_nonneg_odd()
    fc_posdef_output_nonneg_inmap()
    c_skewsym_output()
    c_skewsym_output_odd()
    c_skewsym_output_inmap()
    c_skewsym_output_nonneg()
    c_skewsym_output_nonneg_odd()
    c_skewsym_output_nonneg_inmap()
    fc_skewsym_output()
    fc_skewsym_output_odd()
    fc_skewsym_output_inmap()
    fc_skewsym_output_nonneg()
    fc_skewsym_output_nonneg_odd()
    fc_skewsym_output_nonneg_inmap()
end

#==============================================================================
Test 1 ===> Chain w/ single output
==============================================================================#
function c_single_output()
    x = rand(NIN)
    C = Chain(
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function c_single_output_odd()
    x = rand(NIN)
    C = Chain(
        square,
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
    @assert isapprox(C(x), C(-x), atol=1e-6, rtol=1e-6)
end
function c_single_output_inmap()
    x = rand(NIN)
    C = Chain(
        inmap,
        Dense(4, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function c_single_output_nonneg()
    x = rand(NIN)
    C = Chain(
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1, square),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function c_single_output_nonneg_odd()
    x = rand(NIN)
    C = Chain(
        square,
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1, square),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
    @assert isapprox(C(x), C(-x), atol=1e-6, rtol=1e-6)
end
function c_single_output_nonneg_inmap()
    x = rand(NIN)
    C = Chain(
        inmap,
        Dense(4, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 1, square),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end

#==============================================================================
Test 2 ===> FastChain w/ single output
==============================================================================#
function fc_single_output()
    x = rand(NIN)
    C = FastChain(
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function fc_single_output_odd()
    x = rand(NIN)
    C = FastChain(
        square,
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
    @assert isapprox(C(x,p), C(-x,p), atol=1e-6, rtol=1e-6)
end
function fc_single_output_inmap()
    x = rand(NIN)
    C = FastChain(
        inmap,
        FastDense(4, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function fc_single_output_nonneg()
    x = rand(NIN)
    C = FastChain(
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1, square),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function fc_single_output_nonneg_odd()
    x = rand(NIN)
    C = FastChain(
        square,
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1, square),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
    @assert isapprox(C(x,p), C(-x,p), atol=1e-6, rtol=1e-6)
end
function fc_single_output_nonneg_inmap()
    x = rand(NIN)
    C = FastChain(
        inmap,
        FastDense(4, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 1, square),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end



#==============================================================================
Test 3 ===> Chain w/ multiple output
==============================================================================#
function c_multi_output()
    x = rand(NIN)
    C = Chain(
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 10),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function c_multi_output_odd()
    x = rand(NIN)
    C = Chain(
        square,
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 10),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
    @assert isapprox(C(x), C(-x), atol=1e-6, rtol=1e-6)
end
function c_multi_output_inmap()
    x = rand(NIN)
    C = Chain(
        inmap,
        Dense(4, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 10),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function c_multi_output_nonneg()
    x = rand(NIN)
    C = Chain(
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 10, square),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function c_multi_output_nonneg_odd()
    x = rand(NIN)
    C = Chain(
        square,
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 10, square),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
    @assert isapprox(C(x), C(-x), atol=1e-6, rtol=1e-6)
end
function c_multi_output_nonneg_inmap()
    x = rand(NIN)
    C = Chain(
        inmap,
        Dense(4, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 10, square),
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end

#==============================================================================
Test 4 ===> FastChain w/ multiple output
==============================================================================#
function fc_multi_output()
    x = rand(NIN)
    C = FastChain(
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 10),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function fc_multi_output_odd()
    x = rand(NIN)
    C = FastChain(
        square,
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 10),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
    @assert isapprox(C(x,p), C(-x,p), atol=1e-6, rtol=1e-6)
end
function fc_multi_output_inmap()
    x = rand(NIN)
    C = FastChain(
        inmap,
        FastDense(4, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 10),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function fc_multi_output_nonneg()
    x = rand(NIN)
    C = FastChain(
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 10, square),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end
function fc_multi_output_nonneg_odd()
    x = rand(NIN)
    C = FastChain(
        square,
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 10, square),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
    @assert isapprox(C(x,p), C(-x,p), atol=1e-6, rtol=1e-6)
end
function fc_multi_output_nonneg_inmap()
    x = rand(NIN)
    C = FastChain(
        inmap,
        FastDense(4, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 10, square),
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    @assert isapprox(Ja, Jn, atol=1e-6, rtol=1e-6)
end


#==============================================================================
Test 5 ===> Chain w/ posdef output
==============================================================================#
function c_posdef_output()
    x = rand(NIN)
    C = Chain(
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9), posdef
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function c_posdef_output_odd()
    x = rand(NIN)
    C = Chain(
        square,
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9), posdef
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
    @assert isapprox(C(x), C(-x), atol=1e-6, rtol=1e-6)
end
function c_posdef_output_inmap()
    x = rand(NIN)
    C = Chain(
        inmap,
        Dense(4, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9), posdef
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function c_posdef_output_nonneg()
    x = rand(NIN)
    C = Chain(
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9, square), posdef,
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function c_posdef_output_nonneg_odd()
    x = rand(NIN)
    C = Chain(
        square,
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9, square), posdef,
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
    @assert isapprox(C(x), C(-x), atol=1e-6, rtol=1e-6)
end
function c_posdef_output_nonneg_inmap()
    x = rand(NIN)
    C = Chain(
        inmap,
        Dense(4, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9, square), posdef,
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end

#==============================================================================
Test 6 ===> FastChain w/ posdef output
==============================================================================#
function fc_posdef_output()
    x = rand(NIN)
    C = FastChain(
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9), posdef
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function fc_posdef_output_odd()
    x = rand(NIN)
    C = FastChain(
        square,
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9), posdef
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
    @assert isapprox(C(x,p), C(-x,p), atol=1e-6, rtol=1e-6)
end
function fc_posdef_output_inmap()
    x = rand(NIN)
    C = FastChain(
        inmap,
        FastDense(4, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9), posdef
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function fc_posdef_output_nonneg()
    x = rand(NIN)
    C = FastChain(
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9, square), posdef,
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function fc_posdef_output_nonneg_odd()
    x = rand(NIN)
    C = FastChain(
        square,
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9, square), posdef,
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
    @assert isapprox(C(x,p), C(-x,p), atol=1e-6, rtol=1e-6)
end
function fc_posdef_output_nonneg_inmap()
    x = rand(NIN)
    C = FastChain(
        inmap,
        FastDense(4, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9, square), posdef,
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end


#==============================================================================
Test 7 ===> Chain w/ skewsym output
==============================================================================#
function c_skewsym_output()
    x = rand(NIN)
    C = Chain(
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9), skewsym
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function c_skewsym_output_odd()
    x = rand(NIN)
    C = Chain(
        square,
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9), skewsym
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
    @assert isapprox(C(x), C(-x), atol=1e-6, rtol=1e-6)
end
function c_skewsym_output_inmap()
    x = rand(NIN)
    C = Chain(
        inmap,
        Dense(4, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9), skewsym
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function c_skewsym_output_nonneg()
    x = rand(NIN)
    C = Chain(
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9, square), skewsym,
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function c_skewsym_output_nonneg_odd()
    x = rand(NIN)
    C = Chain(
        square,
        Dense(NIN, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9, square), skewsym,
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
    @assert isapprox(C(x), C(-x), atol=1e-6, rtol=1e-6)
end
function c_skewsym_output_nonneg_inmap()
    x = rand(NIN)
    C = Chain(
        inmap,
        Dense(4, 10, elu),
        Dense(10, 5, elu),
        Dense(5, 9, square), skewsym,
    )
    Ja = jacobian(C, x)
    Jn = Flux.jacobian(C, x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end

#==============================================================================
Test 8 ===> FastChain w/ skewsym output
==============================================================================#
function fc_skewsym_output()
    x = rand(NIN)
    C = FastChain(
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9), skewsym
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function fc_skewsym_output_odd()
    x = rand(NIN)
    C = FastChain(
        square,
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9), skewsym
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
    @assert isapprox(C(x,p), C(-x,p), atol=1e-6, rtol=1e-6)
end
function fc_skewsym_output_inmap()
    x = rand(NIN)
    C = FastChain(
        inmap,
        FastDense(4, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9), skewsym
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function fc_skewsym_output_nonneg()
    x = rand(NIN)
    C = FastChain(
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9, square), skewsym,
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
function fc_skewsym_output_nonneg_odd()
    x = rand(NIN)
    C = FastChain(
        square,
        FastDense(NIN, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9, square), skewsym,
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
    @assert isapprox(C(x,p), C(-x,p), atol=1e-6, rtol=1e-6)
end
function fc_skewsym_output_nonneg_inmap()
    x = rand(NIN)
    C = FastChain(
        inmap,
        FastDense(4, 10, elu),
        FastDense(10, 5, elu),
        FastDense(5, 9, square), skewsym,
    )
    p = DiffEqFlux.initial_params(C)
    Ja = jacobian(C, x, p)
    Jn = Flux.jacobian(_1->C(_1,p), x)[1]
    for (jn, ja) in zip(eachcol(Jn), Ja)
        @assert isapprox(jn, ja[:], atol=1e-6, rtol=1e-6)
    end
end
