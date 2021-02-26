mutable struct HyperParameters{T<:Real}
    epochs_per_minibatch::T
    traj_sampling_size::T
    replay_buffer_size::T
    time_horizon::T
    step_size::T
    regularization_coeff::T
end
function HyperParameters(
    T::DataType ;
    epochs_per_minibatch::Integer=1,
    traj_sampling_size::Integer=0,
    replay_buffer_size::Integer=256,
    time_horizon::Real=3.0,
    step_size::Real=1/20,
    regularization_coeff::Real=0.1
)
    HyperParameters(
        T(epochs_per_minibatch),
        T(traj_sampling_size),
        T(replay_buffer_size),
        T(time_horizon),
        T(step_size),
        T(regularization_coeff)
    )
end
function Base.show(io::IO, hyper::HyperParameters)
    for f in fieldnames(HyperParameters)
        print(io, f)
        print(io, " => ")
        print(io, getfield(hyper, f))
        println()
    end
end



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
    u = EnergyFunction{T}(
        hyper,
        net,
        θ,
        num_states,
        dynamics,
        loss
    )
end
function (policy::EnergyFunction)(x, θ=policy.θ)
    nn_dim = last(last(policy.net.inds).flat)
    ∇NNx = ∇NN(policy.net, x, θ[1:nn_dim])
    return sum(∇NNx .* θ[nn_dim+1 : end])
end
function Base.show(io::IO, policy::EnergyFunction)
    print(io, "EnergyFunction{$(typeof(policy).parameters[1])} with $(Int(policy.num_states))-dimensional input")
    print(io, "\n\nHyperParameters: \n"); show(io, policy.hyper);
    print(io, "\n")
    show(io, policy.net); print(io, " ")
end

function forward(policy::EnergyFunction, x0::Vector, θ::Vector=policy.θ, tf=policy.hyper.time_horizon)
    Array( 
        OrdinaryDiffEq.solve(
            ODEProblem(
                (x,p,t) -> policy.dynamics(x, policy(x,p)), 
                x0, 
                (0f0, tf), 
                θ
            ), 
            Tsit5(), abstol=1e-4, reltol=1e-4,  
            u0=x0, 
            p=θ, 
            saveat=policy.hyper.step_size, 
            sensealg=TrackerAdjoint()
        )
    )
end


function update!(policy::EnergyFunction{T}, x0s::Vector{Array{T,1}}) where {T<:Real}
    num_traj = length(x0s)
    nn_dim   = last(last(policy.net.inds).flat)
    maxiters = policy.hyper.epochs_per_minibatch

    function _loss(θ)
        losses = bufferfrom( zeros(T, num_traj) )
        for (j, x0) in enumerate(x0s)
            losses[j] = policy.loss( forward(policy,x0,θ) )
        end
        mean_loss = sum(losses)/num_traj
        reg_loss = policy.hyper.regularization_coeff*sum(abs, θ[1:nn_dim])/nn_dim
        return mean_loss + reg_loss
    end
    
    res = DiffEqFlux.sciml_train(
        _loss, 
        policy.θ, 
        ADAM(), 
        cb=_update_cb, 
        maxiters=maxiters, 
        progress=false
    )
    if !any(isnan.(res.minimizer))
        policy.θ = res.minimizer
        set_params(policy.net, res.minimizer[1:nn_dim])
    end
    nothing
end


function _update_cb(ϕ, loss)
    ret = false
    if any(isnan.(ϕ))
        @warn("params are NaN, skipping training")
        ret = true
    end
    ret
end


function params_to_npy(policy::EnergyFunction, prefix::String)
    net = policy.net
    nn_dim = last(last(net.inds).flat)
    depth = length(net.widths)-1
    for i = 1:depth
        npzwrite(prefix * "W$(i).npy", get_weights(net, net.θ, i))
        npzwrite(prefix * "b$(i).npy", get_biases(net, net.θ, i))
    end
    npzwrite(prefix * "gradient_coeffs.npy", policy.θ[ nn_dim+1:end ])
end
