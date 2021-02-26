using MLBasedESC

function dynamics(x, u)
    return [x[2], -sin(x[1]) + u]
end
function loss(x)
    return sum(abs2, x)
end

controller = EnergyFunction(Float32, 2, dynamics, loss)