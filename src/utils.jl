dtanh(x) = one(x) - tanh(x)*tanh(x)
delu(x::Real, α=one(x)) = ifelse(x > 0, one(x), α*exp(x) )
const pif0 = Float32(pi)