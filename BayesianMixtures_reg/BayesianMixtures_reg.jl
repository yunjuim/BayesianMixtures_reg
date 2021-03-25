module BayesianMixtures_reg

include("MFM.jl")
include("itcp.jl")
include("slope.jl")
include("itcp_var.jl")
include("slope_var.jl")

using SpecialFunctions
function options(
        mode, # "itcp", "itcp_var", "slope" or "slope_var"
        model_type, # "MFM" or "DPM"
        y, X, # data
        n_total;
        
        n_burn = round(Int, n_total/10),
        verbose = true, # display information or not
        t_max = 40, 
        Initial_grouping = "onecluster",
        W = [[]], # If common covariates exist
        
        # MFM
        gamma = 1.0, # Dirichlet_k(gamma, ..., gamma)
        log_pk = k -> log(0.1)+(k-1)*log(0.9), 

        # DPM 
        alpha = 1.0, # value of alpha 
        
        # Jain-Neal split-merge options
        use_splitmerge = true, 
        n_split = 5, 
        n_merge = 5
    )

    n = length(y)

    if model_type=="MFM"
        log_v = MFM.coefficients(log_pk, gamma, n, t_max + 1)
        a = b = gamma
    elseif model_type=="DPM"
        log_v = float(1 : t_max + 1) * log(alpha) .- logabsgamma(alpha + n)[1] .+ logabsgamma(alpha)[1]
        a, b = 1., 0.
    else
        error("Invalid model_type: $model_type.")
    end

    module_ = getfield(BayesianMixtures_reg, Symbol(mode))
    
    return module_.Options(
        mode, model_type, 
        y, X, W, 
        n_total, n_burn, verbose,
        t_max, Initial_grouping, gamma, log_pk, 
        alpha, 
        use_splitmerge, n_split, n_merge, 
        a, b, log_v, n)
end

# Run the MCMC sampler with the specified options.
function run_sampler(options)
    o = options
    n,n_total = o.n, o.n_total
    module_ = getfield(BayesianMixtures_reg, Symbol(o.mode))

    if o.verbose
       println(o.mode, " ", o.model_type)
       println("n = $n, n_total = $n_total")
       print("Running... ")
    end
    
    # Main run
    elapsed_time = @elapsed t_r, N_r, z_r, beta_r, phi_r, eta_r = module_.sampler(o)
    time_per_step = elapsed_time / (n_total * n)

    if o.verbose
       println("complete.")
       println("Elapsed time = $elapsed_time seconds")
       println("Time per step ~ $time_per_step seconds")
    end

    return module_.Result(o, t_r, N_r, z_r, beta_r, phi_r, eta_r, elapsed_time, time_per_step)
end

# ===================================================================
# ===================================================================
# ================== Functions to analyze results ===================
# ===================================================================
# ===================================================================

function individual_theta(x, result; use_burnin = true)
    o = result.options
    n_total = o.n_total
    if use_burnin; n_burn = o.n_burn; else n_burn = 0; end
    n_use = n_total - n_burn
    p = size(result.beta)[2]
   
    sub_theta = zeros(n_use, p)
    z = result.z[x, :]
    for j = 1 : p
        for i = 1 : n_use
            sub_theta[i, j] = result.beta[z[i + n_burn], j, (i + n_burn)] 
        end
    end
    
    return sub_theta
end

end