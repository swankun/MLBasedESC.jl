export QuadraticEnergyFunction, gradient, controller, predict,
    pde_loss, train_Md!, reset_Vd!

mutable struct QuadraticEnergyFunction{T<:Real, HY, N1, N2, F1, F2, F3, F4}
    hyper::HY
    Md_inv::N1
    Vd::N2
    θ::Vector{T}
    _θind::Dict{Symbol,UnitRange{Int}}
    num_states::Int
    dim_q::Int
    dynamics::F1      # ẋ::Vector{T} = f(x::Vector{T},u::T), u is control input
    loss::F2          # J::T = r(x::Array{T,2})
    ∂H∂q::F3
    mass_matrix::F4
    input_matrix::VecOrMat{T}
end
function QuadraticEnergyFunction( 
    T::DataType, 
    num_states::Int,
    dynamics::Function,
    loss::Function,
    ∂H∂q::Function,
    mass_matrix::Function,
    input_matrix::AbstractVecOrMat
    ;
    dim_q::Int=Int(num_states/2),
    num_hidden_nodes::Int=64,
    initθ_path::String="",
    symmetric::Bool=false 
)
    
    # Verify dim_q
    @assert dim_q < num_states "Dimension of q must be less than dim([q; p])"

    # Parameters setup    
    hyper = HyperParameters(T)

    # Neural network
    nin = isequal(dim_q, num_states/2) ? Int(dim_q) : Int(dim_q*2)
    Md_inv = PSDNeuralNetwork(T, dim_q, nin=nin, symmetric=symmetric)
    Vd = NeuralNetwork(T, 
        [nin, num_hidden_nodes, num_hidden_nodes, 1],
        symmetric=symmetric
    )
    θ = [ Md_inv.net.θ; Vd.θ ]
    _θind = Dict(
        :Md => 1 : length(Md_inv.net.θ), 
        :Vd => length(Md_inv.net.θ)+1 : length(Md_inv.net.θ)+length(Vd.θ)
    )

    # Load parameters if applicable
    if !isempty(initθ_path)
        initθ = pop!( load(initθ_path) ).second.θ
        if isequal(size(initθ), size(θ))
            θ = deepcopy(initθ)
            set_params(Md_inv, getindex(θ, _θind[:Md]))
            set_params(Vd, getindex(θ, _θind[:Vd]))
        else
            @error "Incompatible initθ dimension. Needed $(size(θ)), got $(size(initθ))."
        end
    end

    # Create instance
    Hd = QuadraticEnergyFunction{T, typeof(hyper), typeof(Md_inv), typeof(Vd), typeof(dynamics), typeof(loss), typeof(∂H∂q), typeof(mass_matrix)}(
        hyper,
        Md_inv,
        Vd,
        θ,
        _θind,
        num_states,
        dim_q,
        dynamics,
        loss,
        ∂H∂q,
        mass_matrix,
        T.(input_matrix),
    )
end
function (Hd::QuadraticEnergyFunction)(q, p, θ=Hd.θ) 
    ke = eltype(q)(1/2)*dot(p, Hd.Md_inv(q, θ)*p)
    pe = Hd.Vd( q, @view θ[Hd._θind[:Vd]] )[1]
    return ke + pe
end
function Base.show(io::IO, Hd::QuadraticEnergyFunction)
    print(io, "QuadraticEnergyFunction{$(typeof(Hd).parameters[1])} with $(Int(Hd.num_states))-dimensional input")
    print(io, "\n\nHyperParameters: \n"); show(io, Hd.hyper);
    print(io, "\n")
    print(io, "Mass matrix "); show(io, Hd.Md_inv); print(io, "\n")
    print(io, "Potential energy "); show(io, Hd.Vd)
end

###############################################################################
###############################################################################

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
    Nq = Hd.dim_q
    J = zeros(T, Nq*2, Nq)
    @inbounds for i = 1:Nq
        J[2*i-1, i] = -x[2*i]
        J[2*i, i] = x[2*i-1]
    end
    J
end
function _pe_gradient(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    θVd = @view θ[ Hd._θind[:Vd] ]
    if Hd.dim_q == Hd.num_states/2
        return gradient(Hd.Vd, q, θVd)
    else
        return gradient(Hd.Vd, q, θVd) * _get_input_jacobian(Hd, q)
    end
end
function _ke_gradient(Hd::QuadraticEnergyFunction{T}, q, p, θ=Hd.θ) where {T<:Real}
    dim_q = Hd.dim_q
    θMd = @view θ[ Hd._θind[:Md] ]
    ∇q_Mdinv = reduce(vcat, gradient(Hd.Md_inv, q, θMd))
    if Hd.dim_q == Hd.num_states/2
        gs = map(1:dim_q) do j
            ∇q_Mdinv_j = reshape(∇q_Mdinv[:,j], dim_q, :)
            ∇q_Mdinv_j * p[j]
        end 
        sum(gs)' * p
    else
        jac = _get_input_jacobian(Hd, q)
        gs = map(1:dim_q) do j
            ∇q_Mdinv_j = reshape(∇q_Mdinv[:,j], dim_q, :) * jac
            ∇q_Mdinv_j * p[j]
        end 
        sum(gs)' * p
    end
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
    dim_q = Hd.dim_q
    Minv = inv(Hd.mass_matrix(q))
    θMd = @view θ[ Hd._θind[:Md] ]
    Md = Hd.Md_inv(q, θMd) |> inv
    ∇q_Mdinv = reduce(vcat, gradient(Hd.Md_inv, q, θMd))
    jac = _get_input_jacobian(Hd, q)
    map(1:dim_q) do j
        ∇q_Mdinv_j = reshape(∇q_Mdinv[:,j], dim_q, :) * jac
        sum(abs2, [1f0 1f0] * -Md*Minv*∇q_Mdinv_j')
    end |> sum
end
function pde_loss_Vd(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    dim_q = Hd.dim_q
    θMd = @view Hd.θ[ Hd._θind[:Md] ]
    
    ∇q_Vd = _pe_gradient(Hd, q, θ)
    M = Hd.mass_matrix(q)
    Md_inv = Hd.Md_inv(q, θMd)
    return dot([1f0, 1f0], ( Hd.∂H∂q(q, 0f0) - (M*Md_inv)\vec(∇q_Vd) )) |> abs2
end
function mimic_quadratic_Vd(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    n = Hd.dim_q
    qbar = reduce(vcat, [atan(q[2*i], q[2*i-1]) for i=1:n])
    if norm(qbar) < 0.2
        return zero(T)
    end

    if isequal(length(θ), length(Hd.θ))
        p = ones(T, Int(n*(n+1)/2))
    else
        p = @view θ[ length(Hd.θ)+1:length(Hd.θ)+n+1 ]
    end
    L = vec2tril(p)
    P = L*L' + diagm(fill(T(0.1), n))   
    (dot(qbar, P*qbar) - Hd.Vd(q, θ[Hd._θind[:Vd]])[1]) |> abs
end
function train_Md!(Hd::QuadraticEnergyFunction{T}; max_iters=100, η=0.01, batchsize=96) where {T<:Real}
    
    # Generate data
    data = Vector{T}[]
    qmax = pif0; 
    qmin = -qmax
    q1range = range(qmin, qmax, length=25)
    q2range = range(qmin, qmax, length=25)
    q0 = T[cos(0); sin(0); cos(0); sin(0)]
    for q1 in q1range
        for q2 in q2range
            # push!(data, [q1; q2])
            push!(data, [cos(q1); sin(q1); cos(q2); sin(q2)])
        end
    end
    dataloader = Data.DataLoader(data; batchsize=batchsize, shuffle=true)
    batches = round(Int, dataloader.imax / dataloader.batchsize, RoundUp)

    # Print format
    fexpr = *(
        "Epoch {1:$(ndigits(max_iters))d}/{2:d}  |  PDE loss = {3:.4E}  \n"
    )

    # Optimize
    opt = ADAM(η)
    _loss(data, θ=Hd.θ) = begin
        N = size(data, 1)
        +(
            # map(x -> pde_loss_Md(Hd,x,θ), data) |> sum |> x -> /(x,N),
            map(x -> pde_loss_Vd(Hd,x,θ), data) |> sum |> x -> /(x,N),
            # map(x -> Hd.Vd(x,θ)[1], data) |> minimum |> x -> *(x,-one(x)),
            # map(x -> mimic_quadratic_Vd(Hd,x,θ), data) |> sum |> x -> *(x,T(0.001))
        ) #+ abs2(Hd.Vd(q0,θ)[1])
    end
    params_to_train = [Hd.θ; rand(T, Int(Hd.dim_q*(Hd.dim_q+1)/2))]
    for epoch in 1:max_iters
        current_loss = _loss(data)
        printfmt(fexpr, epoch, max_iters, current_loss)
        # sum(current_loss) <= 1e-4 && break
        batch = 1
        for x in dataloader
            gs = ReverseDiff.gradient(θ -> sum(_loss(x, θ)), params_to_train)
            Optimise.update!(opt, params_to_train, gs)
            Hd.θ = params_to_train[1:length(Hd.θ)]
            set_params(Hd.Md_inv, Hd.θ[Hd._θind[:Md]])
            set_params(Hd.Vd,     Hd.θ[Hd._θind[:Vd]])
            batch += 1
        end
        # contour(q1range, q2range, (x,y)->cond(Hd.Md_inv([x; y])), show=true)
    end

end

###############################################################################
###############################################################################

function controller(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    M = Hd.mass_matrix
    G = Hd.input_matrix
    ∇q_Hd = gradient(Hd)
    k = first(Hd.Md_inv.net.widths)
    u(x, θ=Hd.θ) = begin
        q = x[1:k]
        p = x[k+1:end]
        θMd = @view θ[ Hd._θind[:Md] ]
        Gu_es = Hd.∂H∂q(q, p) .- (M(q) * Hd.Md_inv(q,θMd)) \ ∇q_Hd(q, p, θ)
        u_di = -T(0.1)*dot(G, T(2)*Hd.Md_inv(q,θMd)*p)
        return dot( (G'*G)\G', Gu_es ) + u_di
    end
end
function predict(Hd::QuadraticEnergyFunction{T}, x0::Vector, θ::Vector=Hd.θ, tf=Hd.hyper.time_horizon) where {T<:Real}
    u = controller(Hd)
    Array( 
        OrdinaryDiffEq.solve(
            ODEProblem(
                (x,p,t) -> Hd.dynamics( x, u(x,p) ), 
                x0, 
                (0f0, tf), 
                θ
            ), 
            Tsit5(), abstol=1e-4, reltol=1e-4,  
            u0=x0, 
            p=θ, 
            saveat=Hd.hyper.step_size, 
            sensealg=TrackerAdjoint()
        )
    )
end
