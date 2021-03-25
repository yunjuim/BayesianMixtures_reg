module MVN

include("Lower.jl")
using .Lower

export MVN_params
# Multivariate normal distribution
mutable struct MVN_params
    m::Array{Float64,1} # mean
    L::Array{Float64,2} # lower triangular matrix such that L*L' = R
    _R::Array{Float64,2} # precision matrix (inverse covariance)
    logdetR::Float64 # log of the determinant of the precision matrix
    _Rm::Array{Float64, 1} # _R * m
    quad::Float64 # the quadratic form m' * _R * m
    d::Int64 # dimension
    function MVN_params(m,R)
        p = new(); p.m = copy(m); p._R = copy(R); p.d = d = length(m)
        p.L = zeros(d,d); Lower.Cholesky!(p.L,p._R,d)
        p.logdetR = Lower.logdetsq(p.L,d)
        p._Rm = p._R*p.m
        p.quad = Lower.quadratic(p.m, zeros(d), p.L, d)# p.m'*p._R*p.m
        return p
    end
end

end
