export EnergyFunction, gradient, controller, 
    update!, predict, params_to_npy


mutable struct EnergyFunction{T<:Real, HY, NE, F1, F2} 
    hyper::HY
    net::NE
    θ::Vector{T}
    dim_input::Int
    dim_S1::Vector{Int} # indiciates which dimensions of q is on 𝕊¹
    dynamics::F1      # ẋ::Vector{T} = f(x::Vector{T},u::T), u is control input
    loss::F2          # J::T = r(x::Array{T,2})
end
function EnergyFunction( T::DataType, 
                    dim_input::Int,
                    dynamics::Function,
                    loss::Function ;
                    dim_S1::Vector{Int}=Vector{Int}(),
                    num_hidden_nodes::Int=64,
                    initθ_path::String="",
                    symmetric::Bool=false )


    # Parameters setup    
    hyper = HyperParameters(T)

    # Neural network
    net = NeuralNetwork(T, 
        [dim_input, num_hidden_nodes, num_hidden_nodes, 1],
        symmetric=symmetric
    )
    n_gains = dim_input - length(dim_S1)
    θ = [ net.θ; T.(glorot_uniform(n_gains)) ]

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
    Hd = EnergyFunction{T, typeof(hyper), typeof(net), typeof(dynamics), typeof(loss)}(
        hyper,
        net,
        θ,
        dim_input,
        dim_S1,
        dynamics,
        loss
    )
end
function (Hd::EnergyFunction)(x, θ=Hd.θ)
    nn_dim = last(last(Hd.net.inds).flat)
    Hd.net(x, θ[1:nn_dim])
end
function Base.show(io::IO, Hd::EnergyFunction)
    print(io, "EnergyFunction{$(typeof(Hd).parameters[1])} with $(Int(Hd.dim_input))-dimensional input")
    print(io, "\n\nHyperParameters: \n"); show(io, Hd.hyper);
    print(io, "\n")
    show(io, Hd.net); print(io, " ")
end


function _get_input_jacobian(Hd::EnergyFunction{T}, x) where {T<:Real}
    if isempty(Hd.dim_S1) 
        return LinearAlgebra.I
    end

    nx = Hd.dim_input
    nqp = nx - length(Hd.dim_S1)
    J = zeros(eltype(x), nx, nqp)
    row = 1
    for i = 1:nqp
        if i in Hd.dim_S1
            J[row,i] = -x[row+1]
            J[row+1,i] = x[row]
            row += 2
        else
            J[row,i] = one(eltype(x))
            row += 1
        end
    end
    return J
end
function gradient(Hd::EnergyFunction{T}) where {T<:Real}
    """
    Returns ∇ₓHd(x, θ), the gradient of Hd with respect to x
    """
    nn_dim = last(last(Hd.net.inds).flat)
    (x, θ=Hd.θ) -> gradient(Hd.net, x, θ[1:nn_dim]) * _get_input_jacobian(Hd, x)
end


function controller(Hd::EnergyFunction)
    nn_dim = last(last(Hd.net.inds).flat)
    ∇x_Hd = gradient(Hd)
    u(x, θ=Hd.θ) = begin
        dot(∇x_Hd(x, θ), θ[nn_dim+1 : end])
    end
end


function predict(Hd::EnergyFunction, x0::Vector, θ=Hd.θ, tf=Hd.hyper.time_horizon)
    u = controller(Hd)
    Array( 
        OrdinaryDiffEq.solve(
            ODEProblem(
                # (x,p,t) -> Hd.dynamics(x, u(x,p)),
                (dx,x,p,t) -> Hd.dynamics(dx, x, u(x,p)),
                # (x,p,t) -> Hd.dynamics(x,controller(Hd,x,p)),
                # (dx,x,p,t) -> Hd.dynamics(dx, x, controller(Hd,x,p)),
                x0, 
                (zero(eltype(x0)), tf), 
                θ
            ), 
            Tsit5(), abstol=1e-4, reltol=1e-4,  
            u0=x0, 
            p=θ, 
            saveat=Hd.hyper.step_size, 
            # sensealg=TrackerAdjoint()
            # sensealg=ReverseDiffAdjoint()
            sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))
        )
    )
end
function update!(Hd::EnergyFunction{T}, x0s::Vector{Array{T,1}}; verbose=false, η=0.001) where {T<:Real}
    num_traj = length(x0s)
    nn_dim   = last(last(Hd.net.inds).flat)

    function _loss(θ)
        losses = Zygote.bufferfrom( zeros(T, num_traj) )
        for (j, x0) in enumerate(x0s)
            losses[j] = Hd.loss( predict(Hd,x0,θ) )
        end
        mean_loss = sum(losses)/num_traj
        reg_loss = Hd.hyper.regularization_coeff*sum(abs.(θ[1:nn_dim]))/nn_dim
        return mean_loss + reg_loss, copy(losses), x0s
    end
    
    adtype = GalacticOptim.AutoZygote()
    optf = GalacticOptim.OptimizationFunction((x,p) -> _loss(x), adtype)
    optfunc = GalacticOptim.instantiate_function(optf, Hd.θ, adtype, nothing)
    optprob = GalacticOptim.OptimizationProblem(optfunc, Hd.θ)
    res = GalacticOptim.solve(optprob,
        ADAM(η),
        cb=throttle( (args...)->_update_cb(args...; do_print=verbose), 0.5 ), 
        maxiters=Hd.hyper.epochs_per_minibatch
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
