mutable struct QuadraticEnergyFunction{T<:Real}
    hyper::HyperParameters{T}
    Lθ_net::NeuralNetwork{T}
    Vθ_net::NeuralNetwork{T}
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
    Lθ_net = NeuralNetwork(T, 
        [dim_q, num_hidden_nodes, num_hidden_nodes, num_triangular],
        symmetric=symmetric
    )
    Vθ_net = NeuralNetwork(T, 
        [dim_q, num_hidden_nodes, num_hidden_nodes, 1],
        symmetric=symmetric
    )
    θ = [ Lθ_net.θ; Vθ_net.θ ]
    _θind = Dict(
        :L => 1 : length(Lθ_net.θ), 
        :V => length(Lθ_net.θ)+1 : length(Lθ_net.θ)+length(Vθ_net.θ)
    )

    # Load parameters if applicable
    if !isempty(initθ_path)
        initθ = pop!( load(initθ_path) ).second.θ
        if isequal(size(initθ), size(θ))
            θ = deepcopy(initθ)
            set_params(Lθ_net, getindex(θ, _θind[:L]))
            set_params(Vθ_net, getindex(θ, _θind[:V]))
        else
            @error "Incompatible initθ dimension. Needed $(size(θ)), got $(size(initθ))."
        end
    end

    # Create instance
    Hd = QuadraticEnergyFunction{T}(
        hyper,
        Lθ_net,
        Vθ_net,
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
function mass_matrix_inv(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    dim_q = Int(Hd.num_states / 2)
    Md_inv(q, θ=Hd.θ) = begin
        θL = @view θ[ Hd._θind[:L] ]
        L = Hd.Lθ_net( q, θL ) |> vec2tril
        # return L*L' + eps(T)*I(dim_q)
        return L*L' + T(1e-4)*I(dim_q)
    end
end
function (Hd::QuadraticEnergyFunction{T})(q, p, θ=Hd.θ) where {T<:Real}
    Md_inv = mass_matrix_inv(Hd)
    ke = 0.5f0 * (p' * Md_inv(q, θ) * p)[1]
    pe = Hd.Vθ_net( q, @view θ[Hd._θind[:V]] )[1]
    return ke + pe
end
function Base.show(io::IO, Hd::QuadraticEnergyFunction)
    print(io, "QuadraticEnergyFunction{$(typeof(Hd).parameters[1])} with $(Int(Hd.num_states))-dimensional input")
    print(io, "\n\nHyperParameters: \n"); show(io, Hd.hyper);
    print(io, "\n")
    print(io, "Mass matrix "); show(io, Hd.Lθ_net); print(io, "\n")
    print(io, "Potential energy "); show(io, Hd.Vθ_net)
end


function gradient(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    """
    Returns ∇ₓHd(x, ẋ, θ), the gradient of Hd with respect to x
    """
    N = Int(Hd.num_states/2)
    ∇q_Hd(q, p, θ=Hd.θ) = begin
        θL = θ[ Hd._θind[:L] ]
        θV = θ[ Hd._θind[:V] ]

        L = Hd.Lθ_net( q, θL ) |> vec2tril
        ∂L∂q = [ vec2tril(col) for col in eachcol(∇NN(Hd.Lθ_net, q, θL)) ]
        ∂Mdinv∂q = [ (L * dL') + (dL * L') for dL in ∂L∂q ]
        ∂V∂q = ∇NN(Hd.Vθ_net, q, θV) 

        T(0.5) * vcat((dot(p, Q*p) for Q in ∂Mdinv∂q)...) .+ ∂V∂q[:]
        # T(0.5) * [dot(p, Q*p) for Q in ∂Mdinv∂q] .+ ∂V∂q[:]
    end 
end


function controller(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    Md_inv = mass_matrix_inv(Hd)
    M = Hd.mass_matrix
    G = Hd.input_matrix
    ∇q_Hd = gradient(Hd)
    u(q, p, θ=Hd.θ) = begin
        Gu_es = Hd.∂H∂q(q, p) .- (M(q) * Md_inv(q,θ)) \ ∇q_Hd(q, p, θ)
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
    Md_inv = mass_matrix_inv(Hd)
    M = Hd.mass_matrix
    G = Hd.input_matrix
    ∇q_Hd = gradient(Hd)

    function _loss(θ)
        losses = bufferfrom( zeros(T, num_traj) )
        for (j, x0) in enumerate(x0s)
            ϕ = predict(Hd,x0,θ)
            pde_loss = zero(T)
            for x in eachcol(ϕ)
                q = x[1:n]
                p = x[n+1:end]
                # pde_loss += sum( Hd.∂H∂q(q, p) .- (M(q) * Md_inv(q,θ)) \ ∇q_Hd(q, p, θ) )
            end
            losses[j] = Hd.loss(ϕ) + pde_loss
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
        set_params(Hd.Lθ_net, res.minimizer[Hd._θind[:L]])
        set_params(Hd.Vθ_net, res.minimizer[Hd._θind[:V]])
    end
    nothing
end

