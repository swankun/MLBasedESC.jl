mutable struct QuadraticEnergyFunction{T<:Real}
    hyper::HyperParameters{T}
    Md_inv::PSDNeuralNetwork{T}
    Vd::NeuralNetwork{T}
    θ::Vector{T}
    _θind::Dict{Symbol,UnitRange{Int64}}
    num_states::Integer
    dim_q::Integer
    dynamics::Function      # ẋ::Vector{T} = f(x::Vector{T},u::T), u is control input
    loss::Function          # J::T = r(x::Array{T,2})
    ∂H∂q::Function
    mass_matrix::Function
    input_matrix::VecOrMat{T}
end
function QuadraticEnergyFunction( 
    T::DataType, 
    num_states::Integer,
    dynamics::Function,
    loss::Function,
    ∂H∂q::Function,
    mass_matrix::Function,
    input_matrix::AbstractVecOrMat
    ;
    dim_q::Integer=Int(num_states/2),
    num_hidden_nodes::Integer=64,
    initθ_path::String="",
    symmetric::Bool=false 
)
    
    # Verify dynamics(x,u)
    dx = dynamics(rand(T, num_states), rand(T))
    @assert isequal(valtype(dx), T) "Expected type-stable function ẋ::Vector{T} = dynamics(x::Vector{T}, u::T) where {T<:Real}."

    # Verify loss()
    J = loss(rand(T,2))
    @assert isa(J, T) "Expected type-stable function J::T = r(x::Array{T,2}) where {T<:Real}."
    
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
    Hd = QuadraticEnergyFunction{T}(
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
function (Hd::QuadraticEnergyFunction{T})(q, p, θ=Hd.θ) where {T<:Real}
    ke = 0.5f0 * (p' * Hd.Md_inv(q, θ) * p)[1]
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

function _get_input_jacobian(Hd::QuadraticEnergyFunction{T}, x) where {T<:Real}
    if Hd.dim_q == Hd.num_states/2
        return I(Hd.dim_q)
    else
        N = Hd.dim_q
        # return hcat(([zeros(Int,2*(i-1))..., -x[2*i], x[2*i-1], zeros(Int,2*(N-i))...] for i=1:N)...)
        return reduce(hcat, [[zeros(Int,2*(i-1)); -x[2*i]; x[2*i-1]; zeros(Int,2*(N-i))] for i=1:N])
    end
end
function _pe_gradient(Hd::QuadraticEnergyFunction{T}, q, θ=Hd.θ) where {T<:Real}
    θVd = @view θ[ Hd._θind[:Vd] ]
    ∂V∂q = gradient(Hd.Vd, q, θVd) * _get_input_jacobian(Hd, q)
end
function _ke_gradient(Hd::QuadraticEnergyFunction{T}, q, p, θ=Hd.θ) where {T<:Real}
    dim_q = Hd.dim_q
    θMd = @view θ[ Hd._θind[:Md] ]
    # ∇q_Mdinv = reduce(vcat, gradient(Hd.Md_inv, q, θMd))
    ∇q_Mdinv = vcat(gradient(Hd.Md_inv, q, θMd)...)
    jac = _get_input_jacobian(Hd, q)
    gs = map(1:dim_q) do j
        ∇q_Mdinv_j = reshape(∇q_Mdinv[:,j], dim_q, :) * jac
        ∇q_Mdinv_j .* p[j]
    end 
    sum(gs)' * p
end
function gradient(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    """
    Returns ∇ₓHd(x, ẋ, θ), the gradient of Hd with respect to x
    """
    ∇q_Hd(q, p, θ=Hd.θ) = begin
        θMd = @view θ[ Hd._θind[:Md] ]
        ∂Md_inv∂q = gradient(Hd.Md_inv, q, θMd)
        T(0.5) * _ke_gradient(Hd, q, p, θ) .+ _pe_gradient(Hd, q, θ)[:]
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
    θMd = @view θ[ Hd._θind[:Md] ]
    θVd = @view θ[ Hd._θind[:Vd] ]

    Minv = inv(Hd.mass_matrix(q))
    Md = Hd.Md_inv(q, θMd) |> inv
    ∇q_Vd = vec(gradient(Hd.Vd, q, θVd) * _get_input_jacobian(Hd, q))
    dot([1f0, 1f0], ( -Md*Minv*∇q_Vd ))
end
function train_Md!(Hd::QuadraticEnergyFunction{T}; max_iters=100, η=0.01, batchsize=96) where {T<:Real}
    
    # Generate data
    data = Vector{T}[]
    qmax = pif0; 
    qmin = -qmax
    q1range = range(qmin, qmax, length=25)
    q2range = range(qmin, qmax, length=25)
    for q1 in q1range
        for q2 in q2range
            # push!(data, [q1; q2])
            push!(data, [cos(q1); sin(q1); cos(q2); sin(q2)])
        end
    end
    dataloader = Data.DataLoader(data; batchsize=batchsize, shuffle=true)
    batches = round(Integer, dataloader.imax / dataloader.batchsize, RoundUp)

    # Print format
    fexpr = *(
        "Epoch {1:$(ndigits(max_iters))d}/{2:d}  |  PDE loss = {3:.4f}  \n"
    )

    # Optimize
    opt = ADAM(η)
    _loss(data, θ=Hd.θ) = sum( map(x->pde_loss_Md(Hd,x,θ), data) ) / size(data, 1)
    for epoch in 1:max_iters
        current_loss = _loss(data)
        printfmt(fexpr, epoch, max_iters, current_loss)
        sum(current_loss) <= 1e-4 && break
        batch = 1
        for x in dataloader
            gs = ReverseDiff.gradient(θ -> sum(_loss(x, θ)), Hd.θ)
            Optimise.update!(opt, Hd.θ, gs)
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
        return sum( ((G'*G)\G')[:] .* Gu_es )
    end
end
function predict(Hd::QuadraticEnergyFunction{T}, x0::Vector, θ::Vector=Hd.θ, tf=Hd.hyper.time_horizon) where {T<:Real}
    u = controller(Hd)
    Array( 
        OrdinaryDiffEq.solve(
            ODEProblem(
                (x,p,t) -> Hd.dynamics( x, T(1e-3)*u(x,p) ), 
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
function update!(Hd::QuadraticEnergyFunction{T}, x0s::Vector{Array{T,1}}; verbose=false) where {T<:Real}
    num_traj = length(x0s)


    function _loss(θ)
        losses = bufferfrom( zeros(T, num_traj) )
        for (j, x0) in enumerate(x0s)
            ϕ = predict(Hd,x0,θ)
            losses[j] = Hd.loss(ϕ)
        end
        mean_loss = sum(losses)/num_traj
        reg_loss = Hd.hyper.regularization_coeff*sum(abs, θ)/length(θ)
        return mean_loss + reg_loss, copy(losses), x0s
    end
    
    res = DiffEqFlux.sciml_train(
        _loss, 
        Hd.θ, 
        ADAM(), 
        cb=throttle( (args...)->_update_cb(args...; do_print=verbose), 0.5 ), 
        maxiters=Hd.hyper.epochs_per_minibatch, 
        progress=false
    )
    if !any(isnan.(res.minimizer))
        Hd.θ = res.minimizer
        set_params(Hd.Md_inv, res.minimizer[Hd._θind[:Md]])
        set_params(Hd.Vd, res.minimizer[Hd._θind[:Vd]])
    end
    nothing
end