mutable struct QuadraticEnergyFunction{T<:Real}
    hyper::HyperParameters{T}
    Lθ_net::NeuralNetwork{T}
    Vθ_net::NeuralNetwork{T}
    θ::Vector{T}
    _θind::Dict{Symbol,UnitRange{Int64}}
    num_states::Integer
    dynamics::Function      # ẋ::Vector{T} = f(x::Vector{T},u::T), u is control input
    loss::Function          # J::T = r(x::Array{T,2})
end
function QuadraticEnergyFunction( T::DataType, 
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
        loss
    )
end
function mass_matrix(Hd::QuadraticEnergyFunction{T}) where {T<:Real}
    dim_q = Int(Hd.num_states / 2)
    Md(q, θ=Hd.θ) = begin
        L = Hd.Lθ_net( q, getindex(θ, Hd._θind[:L]) ) |> tril
        return L*L' + eps(T)*I(dim_q)
    end
end
function (Hd::QuadraticEnergyFunction)(q, qdot, θ=Hd.θ)
    Md = mass_matrix(Hd)
    ke = 0.5 * (qdot' * mass_matrix(Hd)(q, θ) * qdot)[1]
    pe = Hd.Vθ_net( q, getindex(θ, Hd._θind[:V]) )[1]
    return ke + pe
end
function Base.show(io::IO, Hd::QuadraticEnergyFunction)
    print(io, "QuadraticEnergyFunction{$(typeof(Hd).parameters[1])} with $(Int(Hd.num_states))-dimensional input")
    print(io, "\n\nHyperParameters: \n"); show(io, Hd.hyper);
    print(io, "\n")
    print(io, "Mass Matrix "); show(io, Hd.Lθ_net); print(io, "\n")
    print(io, "Potential energy "); show(io, Hd.Vθ_net)
end
