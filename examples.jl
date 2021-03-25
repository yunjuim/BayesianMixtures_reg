# I use PyPlot to plot the results.
# This Julia programs depend on Julia packages `LinearAlgebra', `SpecialFunctions', and `Distributions'. 

include("BayesianMixtures_reg/BayesianMixtures_reg.jl")
B = BayesianMixtures_reg

using PyPlot 
using LinearAlgebra
using SpecialFunctions
using Distributions
using StatsBase
using Clustering # To calculate the adjusted rand index

# Data Generation
n = 100
true_K = 2
d = 2
p = 5
p0 = [0.3, 0.7]

true_z = wsample(1 : true_K, p0, n)
true_theta = [[zeros(p)]; [zeros(p)]]

true_theta[1] = [-.5, -.5, 0.5, 0, 0]
true_theta[2] = [ .5, 2.5, 1, .5, .5]
    
mu_w = zeros(d)
S_w = [1 0.3; 0.3 1] 
W = [rand(MvNormal(mu_w, S_w)) for i = 1 : n]

true_eta = [0.25, 1.0]
Weta = hcat(W...)' * true_eta
    
X = [[1; rand(Normal(1, sqrt(3)), p-1)] for i = 1 : n]
y = zeros(Float64, n)
Xbeta = copy(y)

for i = 1 : n
    for j = 1 : true_K
        if true_z[i] == j
            Xbeta[i] = dot(X[i], true_theta[j][1 : p]) 
            y[i] = Xbeta[i] + Weta[i] + rand(Normal(0, 0.5))
        end
    end
end

# Place all covariates into one vector
XW = [zeros(Float64, p + d) for i = 1 : n]
for i = 1 : n; XW[i] = vcat(X[i], W[i]); end;

n_burn, n_total = 2000, 10000

## Run sampler
## When which covariates have heterogeneous effects are known
o1 = B.options("slope_var", "MFM", y, X, n_total, n_burn = n_burn, W = W);
cMFM1 = B.run_sampler(o1);

## When which covariates have heterogeneous effects are unknown
o2 = B.options("slope_var", "MFM", y, XW, n_total, n_burn = n_burn, t_max = n);
cMFM2 = B.run_sampler(o2);

## Point estimates for the split-merge version
# the posterior mode of T
mode(cMFM1.t[(n_burn + 1): n_total]) 
# the posterior mean of eta
mean(cMFM1.eta[: , (n_burn + 1) : n_total], dims = 2) 
# medain ARIs between truth and the clustering formed in each MCMC iteration
RI1 = zeros(n_use)
for i = 1 : n_use; RI1[i] = randindex(cMFM1.z[:, n_burn + i], true_z)[1]; end 
median(RI1) 

## Point estimates for the split-merge version
# the posterior mode of T
mode(cMFM2.t[(n_burn + 1): n_total]) 
# medain ARIs between truth and the clustering formed in each MCMC iteration
RI2 = zeros(n_use)
for i = 1 : n_use; RI2[i] = randindex(cMFM2.z[:, n_burn + i], true_z)[1]; end 
median(RI2) 

## Draw individual profiles 
n_use = n_total - n_burn
indtheta_m1 = zeros(n_use, length(X[1]), n) 
indtheta_m2 = zeros(n_use, length(XW[1]), n);
for i = 1 : n; indtheta_m1[:, :, i] = B.individual_theta(i, cMFM1, use_burnin = true); end
for i = 1 : n; indtheta_m2[:, :, i] = B.individual_theta(i, cMFM2, use_burnin = true); end

thinning = 1 : 100 : n_use # Thinned at every 100th iteration
fig = figure(figsize = (5, 5))
PyPlot.scatter(indtheta_m1[thinning, 2, 1], indtheta_m1[thinning, 3, 1], s = 20, alpha = 0.5, c = "C0", label = "correct cMFM") 
PyPlot.scatter(indtheta_m2[thinning, 2, 1], indtheta_m2[thinning, 3, 1], s = 20, alpha = 0.5, marker = "^", facecolors=:none, edgecolors = "C2", label = "overparameterized cMFM") 
ylim(-1.7, 3); xlim(-1.7, 4); title("Subject 1")
legend(fontsize = 10, markerscale = 2)
ylabel(L"β_2"); xlabel(L"β_1")

fig = figure(figsize = (5, 5))
PyPlot.scatter(indtheta_m1[thinning, 2, 2], indtheta_m1[thinning, 3, 2], s = 20, alpha = 0.5, c = "C0", label = "correct cMFM") 
PyPlot.scatter(indtheta_m2[thinning, 2, 2], indtheta_m2[thinning, 3, 2], s = 20, alpha = 0.5, marker = "H", facecolors=:none, edgecolors = "C1", label = "overparameterized cMFM") 
ylim(-1.7, 3); xlim(-1.7, 4)
legend(fontsize = 10, markerscale = 2)
ylabel(L"β_2"); xlabel(L"β_1"); title("Subject 2")

fig = figure(figsize = (5, 5))
PyPlot.scatter(indtheta_m1[thinning, 2, 24], indtheta_m1[thinning, 3, 24], s = 20, alpha = 0.5, c = "C0", label = "correct cMFM") 
PyPlot.scatter(indtheta_m2[thinning, 2, 24], indtheta_m2[thinning, 3, 24], s = 20, alpha = 0.5, marker = "+", c = "black", label = "overparameterized cMFM") 
ylim(-1.7, 3); xlim(-1.7, 4)
legend(fontsize = 10, markerscale = 1.5)
ylabel(L"β_2"); xlabel(L"β_1"); title("Subject 24")

fig = figure(figsize = (5, 5))
PyPlot.scatter(cMFM1.eta[1, thinning .+ n_burn], cMFM1.eta[2, thinning .+ n_burn], s = 39, alpha = 0.7, c = "C0", label = "correct cMFM")
PyPlot.scatter(indtheta_m2[thinning, 6, 1], indtheta_m2[thinning, 7, 1], s = 50, alpha = 0.5, marker = "^", facecolors=:none, edgecolors = "C2", label = "Subject 1") 
PyPlot.scatter(indtheta_m2[thinning, 6, 2], indtheta_m2[thinning, 7, 2], s = 40, alpha = 0.5, marker = "H", facecolors=:none, edgecolors =  "C1", label = "Subject 2") 
PyPlot.scatter(indtheta_m2[thinning, 6, 24], indtheta_m2[thinning, 7, 24], s = 50, alpha = 0.5, marker = "+", c = "black", label = "Subject 24") 
ylim(-1.4, 3.6); xlim(-2.2, 2.4)
legend(fontsize = 9, markerscale = 1)
ylabel(L"β_6"); xlabel(L"β_5")