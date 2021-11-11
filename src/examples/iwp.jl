const _I1 = 0.0455
const _I2 = 0.00425
const _m3 = 0.183*9.81

struct IWPSystemConvex{T,TPROB}
    I1::T
    I2::T
    m3::T
    M::Matrix{T}
    p::TPROB
end

function IWPSystemConvex(;I1=_I1, I2=_I2, m3=_m3)
    nothing
end