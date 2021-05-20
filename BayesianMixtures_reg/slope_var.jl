module slope_var

include("MVN.jl")
include("Lower.jl")
using .MVN
using .Lower
using LinearAlgebra
using SpecialFunctions
using Distributions

# Hyperparemters
struct Hyperparameters
    p::Int64 # dimension of X
    d::Int64 # dimension of W
    a::Float64 # prior shape of phi's
    b::Float64 # prior rate of phi's
    mb::MVN_params # prior of betas
    me::MVN_params # prior of eta
end

function construct_hyperparameters(X, W)
    p = length(X[1])
    d = length(W[1])
    a = 0.1
    b = 0.1
    mb = MVN_params(zeros(p), 0.1 * Matrix(1.0I, p, p)) 
    me = MVN_params(zeros(d), 0.1 * Matrix(1.0I, d, d))
    return Hyperparameters(p, d, a, b, mb, me)
end

function construct_hyperparameters(X)
    p = length(X[1])
    a = 0.1
    b = 0.1
    mb = MVN_params(zeros(p), 0.1 * Matrix(1.0I, p, p))
    me = MVN_params(zeros(1), 0 * Matrix(1.0I, 1, 1))
    return Hyperparameters(p, 0, a, b, mb, me)
end
    
mutable struct Group
    n::Int64
    sum_xx::Array{Float64,2}
    sum_xres::Array{Float64,1}
    sum_res2::Float64
    Group(p) = (g = new(); g.n = 0; g.sum_xx = zeros(p, p); g.sum_xres = zeros(p); g.sum_res2 = 0.0; g)
end
    
Group_clear!(g) = (g.n = 0; fill!(g.sum_xx, 0); fill!(g.sum_xres, 0); g.sum_res2 = 0.)

function Group_adjoin!(g, y, X, W, eta)
    g.sum_xx += X * X'
    g.sum_xres += X * (y - dot(W, eta))
    g.sum_res2 += (y - dot(W, eta)) * (y - dot(W, eta))
    g.n += 1
end

function Group_remove!(g, y, X, W, eta) 
    g.sum_xx -= X * X'
    g.sum_xres -= X * (y - dot(W, eta))
    g.sum_res2 -= (y - dot(W, eta)) * (y - dot(W, eta))
    g.n -= 1
end

function Group_adjoin!(g, y, X)
    g.sum_xx += X * X'
    g.sum_xres += X * y
    g.sum_res2 += y * y
    g.n += 1
end

function Group_remove!(g, y, X) 
    g.sum_xx -= X * X'
    g.sum_xres -= X * y
    g.sum_res2 -= y * y
    g.n -= 1
end

function log_m(g, H)
    n = g.n; p = H.p 
    x, L = zeros(p), zeros(p, p)
    _E = g.sum_xx + H.mb._R
    b = g.sum_xres + H.mb._Rm
    Lower.Cholesky!(L, _E, p)
    Lower.solve_Lx_eq_y!(L, x, b, p)
    Lower.solve_Ltx_eq_y!(L, b, x, p)
    
    a1 = 0.5 * n + H.a
    b1 = 0.5 * (2 * H.b + H.mb.quad + g.sum_res2 - Lower.quadratic(b, zeros(p), L, p))
     
    return -0.5 * (n * log(2 * pi) - H.mb.logdetR + Lower.logdetsq(L, p)) + 
    (H.a * log(H.b) - a1 * log(b1)) + (logabsgamma(a1)[1] - logabsgamma(H.a)[1])
end  

function log_m(y, X, W, eta, g, H)
    Group_adjoin!(g, y, X, W, eta)
    result = log_m(g, H)
    Group_remove!(g, y, X, W, eta)
    return result
end

function log_m(y, X, g, H)
    Group_adjoin!(g, y, X)
    result = log_m(g, H)
    Group_remove!(g, y, X)
    return result
end

mutable struct Theta
    beta::Array{Float64,1}
    phi::Float64
    Theta(p) = (g = new(); g.beta = zeros(p); g.phi = 0.; g)
end

Theta_clear!(g) = (fill!(g.beta, 0); g.phi = 0.)

function update_theta!(g, theta, H)
    n, p = g.n, H.p
    x, L = zeros(p), zeros(p, p)
    _E = g.sum_xx + H.mb._R
    b = g.sum_xres + H.mb._Rm
    Lower.Cholesky!(L, _E, p)
    Lower.solve_Lx_eq_y!(L, x, b, p)
    Lower.solve_Ltx_eq_y!(L, b, x, p)
    
    a1 = 0.5 * n + H.a
    b1 = 0.5 * (2 * H.b + H.mb.quad + g.sum_res2 - Lower.quadratic(b, zeros(p), L, p))
    
    theta.phi = rand(Gamma(a1, 1 / b1))
    theta.beta = Lower.sample_Normal!(x, b, (L * sqrt(theta.phi)), p) 
end

update_theta!(theta, t, list, g, H) = (for j = 1 : t; c = list[j]; update_theta!(g[c], theta[c], H); end)

function update_eta!(y, X, W, eta, theta, z, t, list, n, H)
    d = H.d
    ww, wres, x, L = zeros(d, d), zeros(d), zeros(d), zeros(d, d)
    for j = 1 : t; c = list[j];
        for k = 1:n; if z[k] == c; ww += theta[c].phi * W[k] * W[k]';
                wres += theta[c].phi * W[k] * (y[k] - dot(X[k], theta[c].beta)); end; end #theta[c].beta)); end; end
    end    
    _E = ww + H.me._R
    b = (wres + H.me._Rm)
    Lower.Cholesky!(L, _E, d)
    Lower.solve_Lx_eq_y!(L, x, b, d)
    Lower.solve_Ltx_eq_y!(L, b, x, d)
    Lower.sample_Normal!(eta, b, L, d)
end

include("slope_var_splitmerge_update.jl")
include("generic.jl")

end
