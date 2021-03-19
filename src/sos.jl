mutable struct SOSPoly{DI<:Integer, DE<:Integer, CO<:AbstractVector{<:Real}}
    dim::DI
    degree::DE
    coeffs::CO
end
function (P::SOSPoly)(x, θ) 
    return P.coeffs * x
end

function decompose_coeff(P::SOSPoly)
    P.coeffs
end