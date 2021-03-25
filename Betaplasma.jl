# I use PyPlot to plot the results.
# This Julia programs depend on Julia packages `LinearAlgebra', `SpecialFunctions', and `Distributions'. 

include("BayesianMixtures_reg/BayesianMixtures_reg.jl")
B = BayesianMixtures_reg

# using DelimitedFiles
# using FreqTables
# using PyPlot

## Read data
mydat = readdlm("betaplasma.txt",'\t',skipstart = 1)

y = mydat[:, 14] .- 0.0;
n = length(y);
smoking3 = [[0.0] for i = 1 : n];
for i = 1 : n
    if mydat[i, 4] == "Never"
        smoking3[i] = [1.0, 0.0] 
        elseif mydat[i, 4] == "Former"
        smoking3[i] = [1.0, 1.0]
        else smoking3[i] = [1.0, 2.0]
    end
end
diet = [[mydat[i, 12] - 0.0] for i = 1 : n];


## Run sampler 
# Split-merge version
n_burn, n_total = 2000, 20000
o1 = B.options("slope_var", "MFM", y, smoking3, n_total, n_burn = n_burn, W = diet, t_max = 40);
cMFM1 = B.run_sampler(o1);

# Non split merge version
n_burn, n_total = 2000, 20000
o2 = B.options("slope_var", "MFM", y, smoking3, n_total, n_burn = n_burn, W = diet, t_max = 40, use_splitmerge = false)
cMFM2 = B.run_sampler(o2); 


## Frequency table of T
freqtable(cMFM1.t[n_burn + 1 : n_total]) 


## Individual profiles
n_use = n_total - n_burn
indtheta_m1 = zeros(n_use, 2, n);
for i = 1 : n; indtheta_m1[:, :, i] = B.individual_theta(i, cMFM1, use_burnin = true); end

# Index for T = 3
ind_t3 = findall(x -> x ==3, cMFM1.t[(n_burn + 1) : n_total])

# Subject 3
fig = figure(figsize = (5, 5))
PyPlot.scatter(indtheta_m1[ind_t3, 1, 3], indtheta_m1[ind_t3, 2, 3], s = 2, alpha = 0.2, cmap = :rainbow)
ylim(-600, 500); xlim(-100, 1400); title("Subject 3"); ylabel(L"β_1"); xlabel(L"β_0")

# Subject 6
fig = figure(figsize = (5, 5))
PyPlot.scatter(indtheta_m1[ind_t3, 1, 6], indtheta_m1[ind_t3, 2, 6], s = 2, alpha = 0.2, cmap = :rainbow)
ylim(-600, 500); xlim(-100, 1400); title("Subject 6"); ylabel(L"β_1"); xlabel(L"β_0")


## Trace Plots
## Split-merge version
# The relative group sizes
list_N1 = copy(cMFM1.N) 
for i = 1 : n_total
    list_N1[:, i] = sort(list_N1[:, i], rev = true) # Sort the group sizes in descending order
end
list_N1[2, :] = list_N1[2, :] + list_N1[1, :] # Sum of the two largest groups
list_N1[3, :] = list_N1[3, :] + list_N1[2, :] # Sum of the three largest groups
list_N1 = list_N1 / n; # Into percentage

# Group-specific effects corresponding the three largest groups
beta2_1 = copy(list_N1)
for i = 1 : n_total; beta2_1[:, i] = sortslices([cMFM1.N[:, i] cMFM1.beta[:, 2, i]], dims = 1, rev = true)[:, 2]; end


## Non split-merge
# The relative group sizes
list_N2 = copy(cMFM2.N) 
for i = 1 : n_total
    list_N2[:, i] = sort(list_N2[:, i], rev = true) # Sort the group sizes in descending order
end

list_N2[2, :] = list_N2[2, :] + list_N2[1, :] # Sum of the two most common groups
list_N2[3, :] = list_N2[3, :] + list_N2[2, :] # Sum of the three most common groups
list_N2 = list_N2 / n; # Into percentage

# Group-specific effects corresponding the three largest groups
beta2_2 = copy(list_N2)
for i = 1 : n_total; beta2_2[:, i] = sortslices([cMFM2.N[:, i] cMFM2.beta[:, 2, i]], dims = 1, rev = true)[:, 2]; end

##
thinning = 1 : 20 : n_total # thinned at every 20th iteration
fig = figure(figsize = (15, 18))
subplot(3, 2, 1)
PyPlot.plot(list_N2[1, :][thinning], linewidth = 0.5)
PyPlot.plot(list_N2[2, :][thinning], linewidth = 0.5)
PyPlot.plot(list_N2[3, :][thinning], linewidth = 0.5)
PyPlot.xlabel("Iteration"); PyPlot.title("Gibbs sampler");
PyPlot.ylim([0.25, 1.05]); PyPlot.yticks(0.3:0.1:1.0); PyPlot.xticks(0:200:1000, 0:4000:n_total);

subplot(3, 2, 2)
PyPlot.plot(list_N1[1, :][thinning], linewidth = 0.5)
PyPlot.plot(list_N1[2, :][thinning], linewidth = 0.5)
PyPlot.plot(list_N1[3, :][thinning], linewidth = 0.5)
PyPlot.xlabel("Iteration"); PyPlot.title("Split-merge");
PyPlot.ylim([0.25, 1.05]); PyPlot.yticks(0.3:0.1:1.0); PyPlot.xticks(0:200:1000, 0:4000:n_total);

subplot(3, 2, 3)
PyPlot.plot(beta2_2[1, thinning], linewidth = 0.5)
PyPlot.plot(beta2_2[2, thinning], linewidth = 0.5)
#PyPlot.plot(tmp[2:end], beta2_2[3, thinning][tmp[2:end]], linewidth = 0.5)
PyPlot.plot(beta2_2[3, thinning], linewidth = 0.5)
PyPlot.xlabel("Iteration"); PyPlot.title("Gibbs sampler"); PyPlot.ylabel(L"\beta_1"); 
PyPlot.ylim([-580, 550]); PyPlot.yticks(-500:250:500); PyPlot.xticks(0:200:1000, 0:4000:n_total);

subplot(3, 2, 4)
PyPlot.plot(beta2_1[1, thinning], linewidth = 0.5)
PyPlot.plot(beta2_1[2, thinning], linewidth = 0.5)
PyPlot.plot(beta2_1[3, thinning], linewidth = 0.5)
PyPlot.xlabel("Iteration"); PyPlot.title("Split-merge"); PyPlot.ylabel(L"\beta_1");
PyPlot.ylim([-580, 550]); PyPlot.yticks(-500:250:500); PyPlot.xticks(0:200:1000, 0:4000:n_total);
tmp = findall(x->x !=0, beta2_2[3, thinning]);

subplot(3, 2, 5)
PyPlot.plot(cMFM2.eta'[thinning], linewidth = 0.5)
PyPlot.xlabel("Iteration"); PyPlot.title("Gibbs sampler"); PyPlot.ylabel(L"\eta");
PyPlot.ylim([-0.01, 0.2]); PyPlot.yticks(0:0.05:0.2); PyPlot.xticks(0:200:1000, 0:4000:n_total);

subplot(3, 2, 6)
PyPlot.plot(cMFM1.eta'[thinning], linewidth = 0.5)
PyPlot.xlabel("Iteration"); PyPlot.title("Split-merge"); PyPlot.ylabel(L"\eta");
PyPlot.ylim([-0.01, 0.2]); PyPlot.yticks(0:0.05:0.2); PyPlot.xticks(0:200:1000, 0:4000:n_total);
