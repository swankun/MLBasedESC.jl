export QuadraticEnergyFunction, gradient, controller, predict,
    pde_loss, train_Md!, reset_Vd!, set_params

mutable struct QuadraticEnergyFunction{T<:Real, HY, N1, N2, N3, F1, F2, F3, F4, F5}
    hyper::HY
    Md_inv::N1
    Vd::N2
    J2::N3
    θ::Vector{T}
    _θind::Dict{Symbol,UnitRange{Int}}
    dim_input::Int
    dim_S1::Vector{Int} # indiciates which dimensions of q is on 𝕊¹
    dynamics::F1      # ẋ::Vector{T} = f(x::Vector{T},u::T), u is control input
    loss::F2          # J::T = r(x::Array{T,2})
    ∂KE∂q::F3
    ∂PE∂q::F4
    mass_matrix::F5
    input_matrix::VecOrMat{T}
    input_matrix_perp::VecOrMat{T}
end

function QuadraticEnergyFunction( 
    T::DataType, 
    dim_input::Int,
    dynamics::Function,
    loss::Function,
    ∂KE∂q::Function,
    ∂PE∂q::Function,
    mass_matrix::Function,
    input_matrix::AbstractVecOrMat,
    input_matrix_perp::AbstractVecOrMat
    ;
    dim_S1::Vector{Int}=Vector{Int}(),
    num_hidden_nodes::Int=64,
    initθ_path::String="",
    symmetric::Bool=false 
)
    
    # Parameters setup    
    hyper = HyperParameters(T)

    # Neural networks
    dim_q = (dim_input - length(dim_S1)) ÷ 2
    nin = length(dim_S1) + dim_q 
    Md_inv = PSDNeuralNetwork(T, dim_q, nin=nin, symmetric=symmetric, num_hidden_nodes=num_hidden_nodes)
    # Vd = NeuralNetwork(T, 
    #     [nin, num_hidden_nodes, num_hidden_nodes, 1],
    #     symmetric=symmetric, 
    #     fout=x->x.^2, dfout=x->2x
    #     # fout=relu, dfout=drelu
    # )
    Vd = symmetric ? SymmetricSOSPoly(nin, 8) : SOSPoly(nin, 4)
    J2 = [SkewSymNeuralNetwork(T, dim_q, nin=nin, num_hidden_nodes=num_hidden_nodes, symmetric=symmetric) for _=1:dim_q]
    θ = [ Md_inv.net.θ; Vd.θ; (Uk.net.θ for Uk in J2)...]
    _θind = Dict(
        :Md => 1 : length(Md_inv.net.θ), 
        :Vd => length(Md_inv.net.θ)+1 : length(Md_inv.net.θ)+length(Vd.θ)
    )
    prevlen = length(Md_inv.net.θ) + length(Vd.θ)
    for (k, Uk) in enumerate(J2)
        thislen = length(Uk.net.θ)
        push!(_θind, 
            Symbol("U",k) => 
            prevlen+1+(k-1)*thislen : prevlen+k*thislen
        )
    end

    # Create instance
    Hd = QuadraticEnergyFunction{T, typeof(hyper), typeof(Md_inv), typeof(Vd), typeof(J2), typeof(dynamics), typeof(loss), typeof(∂KE∂q), typeof(∂PE∂q), typeof(mass_matrix)}(
        hyper,
        Md_inv,
        Vd,
        J2,
        θ,
        _θind,
        dim_input,
        dim_S1,
        dynamics,
        loss,
        ∂KE∂q,
        ∂PE∂q,
        mass_matrix,
        T.(input_matrix),
        T.(input_matrix_perp),
    )
end


function (Hd::QuadraticEnergyFunction)(q, p, θ=Hd.θ) 
    ke = eltype(q)(1/2)*dot(p, Hd.Md_inv(q, θ)*p)
    pe = Hd.Vd( q, @view θ[Hd._θind[:Vd]] )[1]
    return ke + pe
end


function Base.show(io::IO, Hd::QuadraticEnergyFunction)
    print(io, "QuadraticEnergyFunction{$(typeof(Hd).parameters[1])} with $(Int(Hd.dim_input))-dimensional input")
    print(io, "\n\nHyperParameters: \n"); show(io, Hd.hyper);
    print(io, "\n")
    print(io, "Mass matrix "); show(io, Hd.Md_inv); print(io, "\n")
    print(io, "Potential energy "); show(io, Hd.Vd)
end


###############################################################################
###############################################################################


function set_params(Hd::QuadraticEnergyFunction, θ)
    Hd.θ[Hd._θind[:Md]] = θ[Hd._θind[:Md]]
    set_params(Hd.Md_inv, getindex(θ, Hd._θind[:Md]))
    Hd.θ[Hd._θind[:Vd]] = θ[Hd._θind[:Vd]]
    set_params(Hd.Vd, getindex(θ, Hd._θind[:Vd]))
    for (k, Uk) in enumerate(Hd.J2)
        Hd.θ[Hd._θind[Symbol("U",k)]] = θ[Hd._θind[Symbol("U",k)]]
        set_params(Uk, getindex(θ, Hd._θind[Symbol("U",k)]))
    end
end


function reset_Vd!(Hd::QuadraticEnergyFunction) 
    T = first(typeof(Hd).parameters)
    dim = length(Hd._θind[:Vd])
    new_θ = T.(glorot_uniform(dim))
    Hd.θ[Hd._θind[:Vd]] = new_θ
    set_params(Hd.Vd, new_θ)
    nothing
end


###############################################################################
###############################################################################


function _get_input_jacobian(Hd::QuadraticEnergyFunction{T}, x) where {T<:Real}
    if isempty(Hd.dim_S1) 
        return LinearAlgebra.I
    end
    
    nq = Hd.Md_inv.n
    nin = first(Hd.Md_inv.net.widths)
    J = zeros(T, nin, nq)
    row = 1
    for i = 1:nq
        if i in Hd.dim_S1
            J[row,i] = -x[row+1]
            J[row+1,i] = x[row]
            row += 2
        else
            J[row,i] = one(eltype(x))
            row += 1
        end
    end
    J
end


function _pe_gradient(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    θVd = @view θ[ Hd._θind[:Vd] ]
    return gradient(Hd.Vd, q, θVd) * _get_input_jacobian(Hd, q)
end


function _ke_gradient(Hd::QuadraticEnergyFunction{T}, q, p, θ=Hd.θ) where {T<:Real}
    dim_q = Hd.Md_inv.n
    θMd = @view θ[ Hd._θind[:Md] ]
    ∇q_Mdinv = reduce(vcat, gradient(Hd.Md_inv, q, θMd))
    jac = _get_input_jacobian(Hd, q)
    gs = map(1:dim_q) do j
        ∇q_Mdinv_j = reshape(∇q_Mdinv[:,j], dim_q, :) * jac
        ∇q_Mdinv_j * p[j]
    end 
    sum(gs)' * p
end


function gradient(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    """
    Returns ∇ₓHd(x, ẋ, θ), the gradient of Hd with respect to x
    """
    ∇q_Hd(q, p, θ=Hd.θ) = begin
        T(1/2) * _ke_gradient(Hd, q, p, θ) .+ _pe_gradient(Hd, q, θ)[:]
    end 
end


###############################################################################
###############################################################################


function pde_loss_Md(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    dim_q = Hd.Md_inv.n
    Minv = inv(Hd.mass_matrix(q))
    θMd = @view θ[ Hd._θind[:Md] ]
    Mdinv = Hd.Md_inv(q, θMd)
    Md = inv(Mdinv)
    ∇q_Mdinv = reduce(vcat, gradient(Hd.Md_inv, q, θMd))
    jac = _get_input_jacobian(Hd, q)
    Gperp = Hd.input_matrix_perp
    map(1:dim_q) do j
        ∇q_Minv_j = Hd.∂KE∂q(q,j)
        ∇q_Mdinv_j = reshape(∇q_Mdinv[:,j], dim_q, :) * jac
        θUk = @view θ[ Hd._θind[Symbol("U", j)] ]
        Uk = Hd.J2[j](q, θUk)
        sum(abs2, Gperp * (∇q_Minv_j' - Md*Minv*∇q_Mdinv_j' + Uk*Mdinv))
    end |> sum
end


function pde_loss_Vd(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    dim_q = Hd.Md_inv.n
    # θMd = @view Hd.θ[ Hd._θind[:Md] ]
    θMd = @view θ[ Hd._θind[:Md] ]
    
    ∇q_Vd = _pe_gradient(Hd, q, θ)
    M = Hd.mass_matrix(q)
    Md_inv = Hd.Md_inv(q, θMd)
    return dot( Hd.input_matrix_perp, ( Hd.∂PE∂q(q) - inv(M*Md_inv)*vec(∇q_Vd) ) ) |> abs2
end

function pde_loss(Hd::QuadraticEnergyFunction, q, θ=Hd.θ)
    pde_loss_Md(Hd,q,θ) + pde_loss_Vd(Hd,q,θ)
end


function symmetry_loss(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    dim_q = Hd.Md_inv.n
    θMd = @view θ[ Hd._θind[:Md] ]
    # +(
        # norm(inv(Hd.Md_inv(q, θMd)) - inv(Hd.Md_inv(-q, θMd))),
        map(1:dim_q) do j
            θUk = @view θ[ Hd._θind[Symbol("U", j)] ]
            norm(Hd.J2[j](q, θUk) + Hd.J2[j](-q, θUk))
        end |> sum
    # )
end


function mimic_quadratic_Vd(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    n = Hd.Md_inv.n
    qbar = reduce(vcat, [atan(q[2*i], q[2*i-1]) for i=1:n])
    if norm(qbar) < 0.2
        return zero(T)
    end

    if isequal(length(θ), length(Hd.θ))
        p = ones(T, Int(n*(n+1) ÷ 2))
    else
        p = @view θ[ length(Hd.θ)+1:length(Hd.θ)+n+1 ]
    end
    L = vec2tril(p)
    P = L*L' + diagm(fill(T(0.1), n))   
    (dot(qbar, P*qbar) - Hd.Vd(q, θ[Hd._θind[:Vd]])[1]) |> abs
end


function train_Md!(Hd::QuadraticEnergyFunction{T}; max_iters=100, η=0.01, batchsize=80, step=0.1) where {T<:Real}
    
    # Generate data
    data = Vector{T}[]
    qmax = Float32(pi);
    # q1range = range(-qmax, qmax, step=step)
    # q2range = range(-qmax, qmax, step=step)
    q1range = range(-5f0, 5f0, step=0.05f0)
    q2range = range(-0.6f0, 0.6f0, step=0.05f0)
    # q0 = T[cos(0); sin(0); cos(0); sin(0)]
    q0 = T[0, 0]
    for q1 in q1range
        for q2 in q2range
            push!(data, [q1; q2])
            # push!(data, [cos(q1); sin(q1); cos(q2); sin(q2)])
        end
    end
    dataloader = Data.DataLoader(data; batchsize=batchsize, shuffle=true)
    batches = round(Int, dataloader.imax / dataloader.batchsize, RoundUp)

    # Print format
    fexpr = *(
        "Epoch {1:4d}  |  PDE loss = {2:.4E}  \n"
    )

    # Optimize
    opt = ADAM(η)
    _loss(data, θ=Hd.θ) = begin
        θVd = θ[Hd._θind[:Vd]]
        N = size(data, 1)
        +(
            map(x -> pde_loss_Md(Hd,x,θ), data) |> sum |> x -> /(x,N),
            map(x -> pde_loss_Vd(Hd,x,θ), data) |> sum |> x -> /(x,N),
            # 100f0*abs2(Hd.Vd(q0,θVd)[1]),
            # sum(abs2, gradient(Hd.Vd, q0, θVd)),
            # map(x -> -Hd.Vd(x,θVd)[1], data) |> sum, #x -> *(x,-one(x)),
            # map(x -> mimic_quadratic_Vd(Hd,x,θ), data) |> sum |> x -> *(x,T(1)),
            # map(x -> symmetry_loss(Hd,x,θ), data) |> sum |> x -> /(x,N),
        )
    end
    # params_to_train = Hd.θ
    n = Hd.Md_inv.n
    quadratic_mimic_coeff = glorot_uniform(Int(n*(n+1) ÷ 2))
    epoch = 1
    while epoch < max_iters
        params_to_train = [Hd.θ; quadratic_mimic_coeff]
        current_loss = _loss(data)
        printfmt(fexpr, epoch, current_loss)
        sum(current_loss) <= 1e-5 && break
        batch = 1
        for x in dataloader
            gs = ReverseDiff.gradient(θ -> sum(_loss(x, θ)), params_to_train)
            if !any(isnan.(gs))
                Optimise.update!(opt, params_to_train, gs)
                set_params(Hd, params_to_train[1:length(Hd.θ)])
                quadratic_mimic_coeff = params_to_train[length(Hd.θ)+1:end]
                # Hd.θ = params_to_train[1:length(Hd.θ)]
                # set_params(Hd.Md_inv, Hd.θ[Hd._θind[:Md]])
                # set_params(Hd.Vd,     Hd.θ[Hd._θind[:Vd]])
                batch += 1
            else
                # @warn "Training produced NaN."
                # return nothing
            end
        end
        epoch += 1
    end

end


###############################################################################
###############################################################################


function controller(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    M = Hd.mass_matrix
    G = Hd.input_matrix
    ∇q_Hd = gradient(Hd)
    ∂H∂q(q, p) = T(1/2)*sum([Hd.∂KE∂q(q,k)*p[k] for k=1:2])'*p .+ Hd.∂PE∂q(q)
    k = first(Hd.Md_inv.net.widths)
    u(x, θ=Hd.θ) = begin
        q = x[1:k]
        p = x[k+1:end] .* diag(M(q))

        θMd = @view θ[ Hd._θind[:Md] ]
        Mdi = Hd.Md_inv(q,θMd)
        J2 = zeros(T, length(p), length(p))
        for j = 1:length(p)
            θUj = @view θ[ Hd._θind[Symbol("U", j)] ]
            J2 .+= T(1/2)*Hd.J2[j](q, θUj)*p[j]
        end

        Gu_es = ∂H∂q(q, p) .- T(1)*( inv(M(q) * Mdi) * ∇q_Hd(q, p, θ) .+ J2*Mdi*p )
        u_di = -T(1)*dot(G, 2*Mdi*p)
        return T(1)*dot( inv(G'*G)*G', Gu_es ) + u_di
    end
end


function predict(Hd::QuadraticEnergyFunction{T}, x0::Vector, θ::Vector=Hd.θ, tf=Hd.hyper.time_horizon; u=controller(Hd)) where {T<:Real}
    x = Array( 
        OrdinaryDiffEq.solve(
            ODEProblem(
                (x,p,t) -> Hd.dynamics( x, u(x,p) ), 
                x0, 
                (0f0, tf), 
                θ
            ), 
            Vern9(), abstol=1f-3, reltol=1f-3,  
            u0=x0, 
            p=θ, 
            saveat=Hd.hyper.step_size
        )
    )
    [x; mapslices(x->u(x,θ), x, dims=1)]
end

###############################################################################
###############################################################################

test_Hd_gradient(Hd::QuadraticEnergyFunction) = begin
    nx = Hd.dim_input
    nq = first(Hd.Md_inv.net.widths)
    np = nx - nq
    q = randn(Float32, nq)
    p = randn(Float32, np)
    hcat(
        ReverseDiff.gradient(x->Hd(x, p), q),
        gradient(Hd)(q, p)
    )
end

test_ke_gradient(Hd::QuadraticEnergyFunction) = begin
    nx = Hd.dim_input
    nq = first(Hd.Md_inv.net.widths)
    np = nx - nq
    q = randn(Float32, nq)
    p = randn(Float32, np)
    hcat(
        ReverseDiff.gradient(x->p'*Hd.Md_inv(x)*p, q),
        MLBasedESC._ke_gradient(Hd, q, p)
    )
end

test_pe_gradient(Hd::QuadraticEnergyFunction) = begin
    nq = first(Hd.Md_inv.net.widths)
    q = randn(Float32, nq)
    hcat(
        ReverseDiff.gradient(Hd.Vd, q),
        MLBasedESC._pe_gradient(Hd, q) |> vec
    )
end
