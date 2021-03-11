mutable struct QuadraticEnergyFunction{T<:Real}
    hyper::HyperParameters{T}
    Md_inv::PSDNeuralNetwork{T}
    Vd::NeuralNetwork{T}
    θ::Vector{T}
    _θind::Dict{Symbol,UnitRange{Int64}}
    num_states::Integer
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
    
    # Parameters setup    
    hyper = HyperParameters(T)

    # Neural network
    dim_q = Int(num_states / 2)
    num_triangular = Int(dim_q*(dim_q+1)/2)
    # Md_inv = NeuralNetwork(T, 
    #     [dim_q, num_hidden_nodes, num_hidden_nodes, num_triangular],
    #     symmetric=symmetric
    # )
    Md_inv = PSDNeuralNetwork(T, dim_q, symmetric=symmetric)
    Vd = NeuralNetwork(T, 
        [dim_q, num_hidden_nodes, num_hidden_nodes, 1],
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
        dynamics,
        loss,
        ∂H∂q,
        mass_matrix,
        T.(input_matrix)
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


function gradient(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    """
    Returns ∇ₓHd(x, ẋ, θ), the gradient of Hd with respect to x
    """
    ∇q_Hd(q, p, θ=Hd.θ) = begin
        θMd = @view θ[ Hd._θind[:Md] ]
        θVd = @view θ[ Hd._θind[:Vd] ]
        ∂Md_inv∂q = gradient(Hd.Md_inv, q, θMd)
        ∂V∂q = gradient(Hd.Vd, q, θVd) 
        T(0.5) * vcat((dot(p, Q*p) for Q in ∂Md_inv∂q)...) .+ ∂V∂q[:]
    end 
end


function controller(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    M = Hd.mass_matrix
    G = Hd.input_matrix
    ∇q_Hd = gradient(Hd)
    u(q, p, θ=Hd.θ) = begin
        θMd = @view θ[ Hd._θind[:Md] ]
        Gu_es = Hd.∂H∂q(q, p) .- (M(q) * Hd.Md_inv(q,θMd)) \ ∇q_Hd(q, p, θ)
        return sum( ((G'*G)\G')[:] .* Gu_es )
    end
end


function predict(Hd::QuadraticEnergyFunction{T}, x0::Vector, θ::Vector=Hd.θ, tf=Hd.hyper.time_horizon) where {T<:Real}
    u = controller(Hd)
    n = Int(Hd.num_states/2)
    Array( 
        OrdinaryDiffEq.solve(
            ODEProblem(
                (x,p,t) -> Hd.dynamics( x, T(1e-3)*u(x[1:n],x[n+1:end],p) ), 
                # (x,p,t) -> Hd.dynamics(x, 0f0), 
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
    n = Int(Hd.num_states/2)
    M = Hd.mass_matrix
    G = Hd.input_matrix
    ∇q_Hd = gradient(Hd)

    function _pde_loss(θ)
        θMd = @view θ[ Hd._θind[:Md] ]
        pde_loss = zero(T)
        map(1:trajectories) do j
            for x in eachcol(ϕ)
                q = x[1:n]
                p = x[n+1:end]
                pde_loss += sum( Hd.∂H∂q(q, p) .- (M(q) * Hd.Md_inv(q,θMd)) \ ∇q_Hd(q, p, θ) )
            end
        end
    end

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

