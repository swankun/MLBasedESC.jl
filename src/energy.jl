mutable struct EnergyFunction{T<:Real}
    hyper::HyperParameters{T}
    net::NeuralNetwork{T}
    θ::Vector{T}
    num_states::T
    dynamics::Function      # ẋ::Vector{T} = f(x::Vector{T},u::T), u is control input
    loss::Function          # J::T = r(x::Array{T,2})
end
function EnergyFunction( T::DataType, 
                    num_states::Integer,
                    dynamics::Function,
                    loss::Function ;
                    num_hidden_nodes::Integer=64,
                    initθ_path::String="",
                    symmetric::Bool=false )

    
    # Verify dynamics(x,u)
    dx = dynamics(rand(T, num_states), rand(T))
    @assert isequal(valtype(dx), T) "Expected type-stable function ẋ::Vector{T} = dynamics(x::Vector{T}, u::T) where {T<:Real}."

    # Verify loss()
    J = loss(rand(T,2))
    @assert isa(J, T) "Expected type-stable function J::T = r(x::Array{T,2}) where {T<:Real}."
    
    # Parameters setup    
    hyper = HyperParameters(T)

    # Neural network
    net = NeuralNetwork(T, 
        [num_states, num_hidden_nodes, num_hidden_nodes, 1],
        symmetric=symmetric
    )
    θ = [ net.θ; T.(glorot_uniform(num_states)) ]

    # Load parameters if applicable
    if !isempty(initθ_path)
        initθ = pop!( load(initθ_path) ).second.θ
        if isequal(size(initθ), size(θ))
            θ = deepcopy(initθ)
            set_params(net, θ[1:length(net.θ)])
        else
            @error "Incompatible initθ dimension. Needed $(size(θ)), got $(size(initθ))."
        end
    end

    # Create instance
    Hd = EnergyFunction{T}(
        hyper,
        net,
        θ,
        num_states,
        dynamics,
        loss
    )
end
function (Hd::EnergyFunction)(x, θ=Hd.θ)
    nn_dim = last(last(Hd.net.inds).flat)
    Hd.net(x, θ[1:nn_dim])
end
function Base.show(io::IO, Hd::EnergyFunction)
    print(io, "EnergyFunction{$(typeof(Hd).parameters[1])} with $(Int(Hd.num_states))-dimensional input")
    print(io, "\n\nHyperParameters: \n"); show(io, Hd.hyper);
    print(io, "\n")
    show(io, Hd.net); print(io, " ")
end


function gradient(Hd::EnergyFunction{T}) where {T<:Real}
    """
    Returns ∇ₓHd(x, θ), the gradient of Hd with respect to x
    """
    nn_dim = last(last(Hd.net.inds).flat)
    (x, θ=Hd.θ) -> gradient(Hd.net, x, θ[1:nn_dim])
end


function controller(Hd::EnergyFunction)
    nn_dim = last(last(Hd.net.inds).flat)
    ∇x_Hd = gradient(Hd)
    u(x, θ=Hd.θ) = begin
        (∇x_Hd(x, θ) * θ[nn_dim+1 : end])[1]
    end
end


function predict(Hd::EnergyFunction, x0::Vector, θ::Vector=Hd.θ, tf=Hd.hyper.time_horizon)
    u = controller(Hd)
    Array( 
        OrdinaryDiffEq.solve(
            ODEProblem(
                (x,p,t) -> Hd.dynamics(x, u(x,p)), 
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


function update!(Hd::EnergyFunction{T}, x0s::Vector{Array{T,1}}; verbose=false) where {T<:Real}
    num_traj = length(x0s)
    nn_dim   = last(last(Hd.net.inds).flat)

    function _loss(θ)
        losses = bufferfrom( zeros(T, num_traj) )
        for (j, x0) in enumerate(x0s)
            losses[j] = Hd.loss( predict(Hd,x0,θ) )
        end
        mean_loss = sum(losses)/num_traj
        reg_loss = Hd.hyper.regularization_coeff*sum(abs, θ[1:nn_dim])/nn_dim
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
        set_params(Hd.net, res.minimizer[1:nn_dim])
    end
    nothing
end


function _update_cb(ϕ, batch_loss, losses, x0s; do_print=false)
    if any(isnan.(ϕ))
        @warn("params are NaN, skipping training")
        return true;
    end
    
    if do_print
        n = length(x0s[1])
        fexpr = *(
            "x0 = (",
            join(["{$i:8.4f}, " for i=1:n])[1:end-2],
            " )  |  loss = {$(n+1):.4f}  \n"
        )
        for (loss, x0) in zip(losses, x0s)
            printfmt(fexpr, x0..., loss)
        end
    end

    return false;
end


function params_to_npy(Hd::EnergyFunction, prefix::String)
    net = Hd.net
    nn_dim = last(last(net.inds).flat)
    depth = length(net.widths)-1
    for i = 1:depth
        npzwrite(prefix * "W$(i).npy", get_weights(net, net.θ, i))
        npzwrite(prefix * "b$(i).npy", get_biases(net, net.θ, i))
    end
    npzwrite(prefix * "gradient_coeffs.npy", Hd.θ[ nn_dim+1:end ])
end
