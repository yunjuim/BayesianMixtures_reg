logsumexp(a,b) = (m = max(a,b); m == -Inf ? -Inf : log(exp(a-m) + exp(b-m)) + m)

function randp(p,k)
    s = 0.; for j = 1:k; s += p[j]; end
    u = rand()*s
    j = 1
    C = p[1]
    while u > C
        j += 1
        C += p[j]
    end
    @assert(j <= k)
    return j
end

function randlogp!(log_p,k)
    log_s = -Inf; for j = 1:k; log_s = logsumexp(log_s,log_p[j]); end
    p = log_p
    for j = 1:k; p[j] = exp(log_p[j]-log_s); end
    return randp(p,k)
end

function ordered_insert!(index,list,t)
    j = t
    while (j>0) && (list[j]>index)
        list[j+1] = list[j]
        j -= 1
    end
    list[j+1] = index
end

function ordered_remove!(index,list,t)
    for j = 1:t
        if list[j]>=index; list[j] = list[j+1]; end
    end
end

function ordered_next(list)
    j = 1
    while list[j] == j; j += 1; end
    return j
end

function restricted_Gibbs!(zsa,zsb,tia,tib,tja,tjb,cia,cib,cja,cjb,ni,nj,i,j,S,ns,y,X,W,eta,b,H,active)

    if !active; tia,tja = deepcopy(tia),deepcopy(tja); end
    log_p = 0.
    for ks = 1:ns
        k = S[ks]
        if k!=i && k!=j
            if zsa[k]==cia
                ni -= 1; Group_remove!(tia,y[k],X[k],W[k],eta) 
            else
                nj -= 1; Group_remove!(tja,y[k],X[k],W[k],eta) 
            end
            Li = log_m(y[k],X[k],W[k],eta,tia,H) - log_m(tia,H)
            Lj = log_m(y[k],X[k],W[k],eta,tja,H) - log_m(tja,H)
            Pi = exp(log(ni+b)+Li - logsumexp(log(ni+b)+Li,log(nj+b)+Lj))
            
            if active
                if rand() < Pi
                    zsb[k] = cib
                else
                    zsb[k] = cjb
                end
            end
            if zsb[k] == cib
                ni += 1
                Group_adjoin!(tia,y[k],X[k],W[k],eta)
                log_p += log(Pi)
            else
                nj += 1
                Group_adjoin!(tja,y[k],X[k],W[k],eta)
                log_p += log(1-Pi)
            end
        end
    end
    return log_p,ni,nj
end

function restricted_Gibbs!(zsa,zsb,tia,tib,tja,tjb,cia,cib,cja,cjb,ni,nj,i,j,S,ns,y,X,b,H,active)

    if !active; tia,tja = deepcopy(tia),deepcopy(tja); end
    log_p = 0.
    for ks = 1:ns
        k = S[ks]
        if k!=i && k!=j
            if zsa[k]==cia
                ni -= 1; Group_remove!(tia,y[k],X[k]) 
            else
                nj -= 1; Group_remove!(tja,y[k],X[k]) 
            end
            Li = log_m(y[k],X[k],tia,H) - log_m(tia,H)
            Lj = log_m(y[k],X[k],tja,H) - log_m(tja,H)
            Pi = exp(log(ni+b)+Li - logsumexp(log(ni+b)+Li,log(nj+b)+Lj))
            
            if active
                if rand()<Pi
                    zsb[k] = cib
                else
                    zsb[k] = cjb
                end
            end
            if zsb[k]==cib
                ni += 1
                Group_adjoin!(tia,y[k],X[k])
                log_p += log(Pi)
            else
                nj += 1
                Group_adjoin!(tja,y[k],X[k])
                log_p += log(1-Pi)
            end
        end
    end
    return log_p,ni,nj
end

function split_merge!(y,X,W,eta,z,zs,S,group,list,N,t,H,a,b,log_v,n_split,n_merge)
    n = length(y)
    
    # randomly choose a pair of indices
    i = round(Int, ceil(rand() * n))
    j = round(Int, ceil(rand() * (n-1))); if j >= i; j += 1; end
    ci0, cj0 = z[i], z[j]
    ti0, tj0 = group[ci0], group[cj0]
    
    # set S[1],...,S[ns] to the indices of the points in clusters ci0 and cj0
    ns = 0
    for k = 1 : n
        if z[k] == ci0 || z[k] == cj0; ns += 1; S[ns] = k; end
    end
    
    # find available cluster IDs for merge and split parameters
    k = 1
    while list[k] == k; k += 1; end; cm = k
    while list[k] == k+1; k += 1; end; ci = k+1
    while list[k] == k+2; k += 1; end; cj = k+2
    tm, ti, tj = group[cm], group[ci], group[cj]
    
    # merge state
    for ks = 1:ns; Group_adjoin!(tm, y[S[ks]], X[S[ks]], W[S[ks]], eta); end 
    
    # randomly choose the split launch state
    zs[i] = ci; Group_adjoin!(ti, y[i], X[i], W[i], eta); ni = 1
    zs[j] = cj; Group_adjoin!(tj, y[j], X[j], W[j], eta); nj = 1
    for ks = 1 : ns  # start with a uniformly chosen split
        k = S[ks]
        if k != i && k != j
            if rand() < 0.5; zs[k] = ci; Group_adjoin!(ti, y[k], X[k], W[k], eta); ni += 1
                else;        zs[k] = cj; Group_adjoin!(tj, y[k], X[k], W[k], eta); nj += 1
            end
        end
    end
    
    for rep = 1 : n_split  # make several moves
        log_p, ni, nj = restricted_Gibbs!(zs,zs,ti,ti,tj,tj,ci,ci,cj,cj,ni,nj,i,j,S,ns,y,X,W,eta,b,H,true)
    end
    
    # make proposal
    if ci0 == cj0  # propose a split
        # make one final sweep and compute it's probability density
        log_prop_ab, ni, nj = restricted_Gibbs!(zs,zs,ti,ti,tj,tj,ci,ci,cj,cj,ni,nj,i,j,S,ns,y,X,W,eta,b,H,true)
        
        # probability of going from merge state to original state
        log_prop_ba = 0.0  # log(1)
        
        # compute acceptance probability
        log_prior_b = log_v[t+1] + logabsgamma(ni + b)[1] + logabsgamma(nj + b)[1] - 2 * logabsgamma(a)[1]
        log_prior_a = log_v[t] + logabsgamma(ns + b)[1] - logabsgamma(a)[1]
        log_lik_ratio = log_m(ti, H) + log_m(tj, H) - log_m(ti0, H)
        p_accept = min(1.0, exp(log_prop_ba - log_prop_ab + log_prior_b-log_prior_a + log_lik_ratio))
        
        # accept or reject
        if rand() < p_accept # accept split
            for ks = 1:ns; z[S[ks]] = zs[S[ks]]; end
            ordered_remove!(ci0, list, t)
            ordered_insert!(ci, list, t-1)
            ordered_insert!(cj, list, t)
            N[ci0], N[ci], N[cj] = 0, ni, nj
            t += 1
            Group_clear!(ti0)
        else # reject split
            Group_clear!(ti)
            Group_clear!(tj)
        end
        Group_clear!(tm)
        
    else  # propose a merge
        # probability of going to merge state
        log_prop_ab = 0.0  # log(1)
        
        # compute probability density of going from split launch state to original state
        log_prop_ba, ni, nj = restricted_Gibbs!(zs,z,ti,ti0,tj,tj0,ci,ci0,cj,cj0,ni,nj,i,j,S,ns,y,X,W,eta,b,H,false)
        
        # compute acceptance probability
        log_prior_b = log_v[t-1] + logabsgamma(ns + b)[1] - logabsgamma(a)[1]
        log_prior_a = log_v[t] + logabsgamma(ni + b)[1] + logabsgamma(nj + b)[1] - 2 * logabsgamma(a)[1]
        log_lik_ratio = log_m(tm, H) - log_m(ti0, H) - log_m(tj0, H)
        p_accept = min(1.0, exp(log_prop_ba - log_prop_ab + log_prior_b-log_prior_a + log_lik_ratio))
   
        # accept or reject
        if rand() < p_accept # accept merge
            for ks = 1:ns; z[S[ks]] = cm; end
            ordered_remove!(ci0, list, t)
            ordered_remove!(cj0, list, t-1)
            ordered_insert!(cm, list, t-2)
            N[cm], N[ci0], N[cj0] = ns, 0, 0
            t -= 1
            Group_clear!(ti0)
            Group_clear!(tj0)
        else # reject merge
            Group_clear!(tm)
        end
        Group_clear!(ti)
        Group_clear!(tj)
    end
    return t
end

function split_merge!(y,X,z,zs,S,group,list,N,t,H,a,b,log_v,n_split,n_merge)
    n = length(y)
    
    # randomly choose a pair of indices
    i = round(Int, ceil(rand() * n))
    j = round(Int, ceil(rand() * (n-1))); if j >= i; j += 1; end
    ci0, cj0 = z[i], z[j]
    ti0, tj0 = group[ci0], group[cj0]
    
    # set S[1],...,S[ns] to the indices of the points in clusters ci0 and cj0
    ns = 0
    for k = 1 : n
        if z[k] == ci0 || z[k] == cj0; ns += 1; S[ns] = k; end
    end
    
    # find available cluster IDs for merge and split parameters
    k = 1
    while list[k] == k; k += 1; end; cm = k
    while list[k] == k+1; k += 1; end; ci = k+1
    while list[k] == k+2; k += 1; end; cj = k+2
    tm, ti, tj = group[cm], group[ci], group[cj]
    
    # merge state
    for ks = 1 : ns; Group_adjoin!(tm, y[S[ks]], X[S[ks]]); end # get the sufficient statistics
    
    # randomly choose the split launch state
    zs[i] = ci; Group_adjoin!(ti, y[i], X[i]); ni = 1
    zs[j] = cj; Group_adjoin!(tj, y[j], X[j]); nj = 1
    for ks = 1 : ns  # start with a uniformly chosen split
        k = S[ks]
        if k != i && k != j
            if rand() < 0.5; zs[k] = ci; Group_adjoin!(ti, y[k], X[k]); ni += 1
                else;        zs[k] = cj; Group_adjoin!(tj, y[k], X[k]); nj += 1
            end
        end
    end
    for rep = 1 : n_split  # make several moves
        log_p, ni, nj = restricted_Gibbs!(zs,zs,ti,ti,tj,tj,ci,ci,cj,cj,ni,nj,i,j,S,ns,y,X,b,H,true)
    end
    
    # make proposal
    if ci0 == cj0  # propose a split
        # make one final sweep and compute it's probability density
        log_prop_ab, ni, nj = restricted_Gibbs!(zs,zs,ti,ti,tj,tj,ci,ci,cj,cj,ni,nj,i,j,S,ns,y,X,b,H,true)
        
        # probability of going from merge state to original state
        log_prop_ba = 0.0  # log(1)
        
        # compute acceptance probability
        log_prior_b = log_v[t+1] + logabsgamma(ni+b)[1] + logabsgamma(nj+b)[1] - 2 * logabsgamma(a)[1]
        log_prior_a = log_v[t] + logabsgamma(ns+b)[1] - logabsgamma(a)[1]
        log_lik_ratio = log_m(ti, H) + log_m(tj, H) - log_m(ti0, H)
        p_accept = min(1.0, exp(log_prop_ba - log_prop_ab + log_prior_b-log_prior_a + log_lik_ratio))

        # accept or reject
        if rand() < p_accept # accept split
            for ks = 1 : ns; z[S[ks]] = zs[S[ks]]; end
            ordered_remove!(ci0,list,t)
            ordered_insert!(ci,list,t-1)
            ordered_insert!(cj,list,t)
            N[ci0], N[ci], N[cj] = 0, ni, nj
            t += 1
            Group_clear!(ti0)
        else # reject split
            Group_clear!(ti)
            Group_clear!(tj)
        end
        Group_clear!(tm)
        
    else  # propose a merge
        # probability of going to merge state
        log_prop_ab = 0.0  # log(1)
        
        # compute probability density of going from split launch state to original state
        log_prop_ba, ni, nj = restricted_Gibbs!(zs,z,ti,ti0,tj,tj0,ci,ci0,cj,cj0,ni,nj,i,j,S,ns,y,X,b,H,false)
        
        # compute acceptance probability
        log_prior_b = log_v[t-1] + logabsgamma(ns+b)[1] - logabsgamma(a)[1]
        log_prior_a = log_v[t] + logabsgamma(ni+b)[1] + logabsgamma(nj+b)[1] - 2 * logabsgamma(a)[1]
        log_lik_ratio = log_m(tm, H) - log_m(ti0, H) - log_m(tj0, H)
        p_accept = min(1.0, exp(log_prop_ba - log_prop_ab + log_prior_b-log_prior_a + log_lik_ratio))
        
        # accept or reject
        if rand() < p_accept # accept merge
            for ks = 1 : ns; z[S[ks]] = cm; end
            ordered_remove!(ci0,list,t)
            ordered_remove!(cj0,list,t-1)
            ordered_insert!(cm,list,t-2)
            N[cm], N[ci0], N[cj0] = ns,0,0
            t -= 1
            Group_clear!(ti0)
            Group_clear!(tj0)
        else # reject merge
            Group_clear!(tm)
        end
        Group_clear!(ti)
        Group_clear!(tj)
    end
    return t
end

function sampler(options)
    y, X, W, n = options.y, options.X, options.W, options.n
    n_total, t_max, Initial_grouping = options.n_total, options.t_max, options.Initial_grouping
    a, b, log_v = options.a, options.b, options.log_v
    use_splitmerge, n_split, n_merge = options.use_splitmerge, options.n_split, options.n_merge
    
    @assert(n == length(y))
    
    log_p = zeros(n + 1)
    log_Nb = log.((1:n) .+ b)
    
    if length(W[1]) == 0
        H = construct_hyperparameters(X)
        p, d = H.p, H.d
    else
        H = construct_hyperparameters(X, W)
        p, d = H.p, H.d
        eta = rand(d)
    end
    
    if Initial_grouping == "onecluster"
        t = 1 # number of clusters
        z = ones(Int, n)
        zs = copy(z) # temporary variable used for split-merge assignments
        S = zeros(Int,n) # temporary variable used for split-merge indices

        list = zeros(Int, t_max + 3); list[1] = 1

        c_next = 2
        N = zeros(Int, t_max + 3); N[1] = n # N[c] = size of cluster c
    end
    if Initial_grouping == "singleton"
        t = n # number of clustesrs
        z = zeros(Int, n); z[1 : n] = 1 : n
        zs = copy(z)
        S = zeros(Int, n)
        
        list = zeros(Int, t_max + 3); list[1 : n] = 1 : n
        c_next = n + 1
        N = zeros(Int, t_max + 3); N[1 : n] = fill(1, n)
    end
    
    group = [Group(p)::Group for c = 1 : t_max + 3]
    theta = [Theta(p)::Theta for c = 1 : t_max + 3]

    # Record-keeping variables
    t_r = zeros(Int64, n_total); 
    N_r = zeros(Int64, t_max + 3, n_total)
    z_r = zeros(Int16, n, n_total)
    beta_r = zeros(Float64, t_max + 3, p, n_total)
    phi_r = zeros(Float64, t_max + 3, n_total)
    eta_r = zeros(Float64, d, n_total)
    
    if length(W[1]) == 0 # if W does not exist
        if Initial_grouping == "singleton"
            for i = 1 : n; Group_adjoin!(group[i], y[i], X[i]); end
        end
        if Initial_grouping == "onecluster"
            for i = 1 : n; Group_adjoin!(group[1], y[i], X[i]); end
        end
        
        for iteration = 1:n_total
            # -------------- Resample z's --------------
            for i = 1:n
                # remove point i from it's cluster
                c = z[i]    
                N[c] -= 1
                Group_remove!(group[c], y[i], X[i])
                if N[c] > 0
                    c_prop = c_next
                else
                    c_prop = c
                    # remove cluster {i}, keeping list in proper order
                    ordered_remove!(c,list,t)
                    t -= 1
                end

                # compute probabilities for resampling
                for j = 1:t; cc = list[j]
                    log_p[j] = log_Nb[N[cc]] + log_m(y[i], X[i], group[cc], H) - log_m(group[cc], H)
                end
                log_p[t+1] = log_v[t + 1] - log_v[t] + log(a) + log_m(y[i], X[i], group[c_prop], H)

                # sample a new cluster for it
                j = randlogp!(log_p, t + 1)

                # add point i to it's new cluster
                if j <= t
                    c = list[j]
                else
                    c = c_prop
                    ordered_insert!(c,list,t)
                    t += 1
                    c_next = ordered_next(list)
                    @assert(t <= t_max, "Sampled t has exceeded t_max. Increase t_max and retry.")
                end
                # update sufficient statistics
                Group_adjoin!(group[c], y[i], X[i])
                z[i] = c 
                N[c] += 1
            end

            # -------------- Split/merge move --------------
            if use_splitmerge
                t = split_merge!(y,X,z,zs,S,group,list,N,t,H,a,b,log_v,n_split,n_merge)
                c_next = ordered_next(list)
                @assert(t<=t_max, "Sampled t has exceeded t_max. Increase t_max and retry.")
            end

            # -------------- update group-specific/common parameters -------------- 
            update_theta!(theta, t, list, group, H)

            for j = 1:t; c = list[j];
                N_r[c, iteration] = N[c]
                beta_r[c,:,iteration] = theta[c].beta
                phi_r[c, iteration] = theta[c].phi
            end
            
            t_r[iteration] = t
            z_r[:, iteration] = z

            for j = 1:t_max; Group_clear!(group[j]); Theta_clear!(theta[j]); end
            for j = 1:t; c = list[j]; for k = 1:n; if z[k] == c; Group_adjoin!(group[c],y[k],X[k]); end; end; end
        end
    else # If W exists
        if Initial_grouping == "singleton"
            for i = 1 : n; Group_adjoin!(group[i],y[i],X[i],W[i],eta); end
        end
        if Initial_grouping == "onecluster"
            for i = 1 : n; Group_adjoin!(group[1],y[i],X[i],W[i],eta); end
        end
        
        for iteration = 1:n_total
            # -------------- Resample z's --------------
            for i = 1:n
                # remove point i from it's cluster
                c = z[i]    
                N[c] -= 1
                Group_remove!(group[c],y[i],X[i],W[i],eta)
                if N[c]>0
                    c_prop = c_next
                else
                    c_prop = c
                    # remove cluster {i}, keeping list in proper order
                    ordered_remove!(c,list,t)
                    t -= 1
                end

                # compute probabilities for resampling
                for j = 1:t; cc = list[j]
                    log_p[j] = log_Nb[N[cc]] + log_m(y[i], X[i], W[i], eta, group[cc], H) - log_m(group[cc], H)
                end
                log_p[t+1] = log_v[t + 1]-log_v[t] + log(a) + log_m(y[i], X[i], W[i], eta, group[c_prop], H)

                # sample a new cluster for it
                j = randlogp!(log_p, t + 1)

                # add point i to it's new cluster
                if j <= t
                    c = list[j]
                else
                    c = c_prop
                    ordered_insert!(c, list, t)
                    t += 1
                    c_next = ordered_next(list)
                    @assert(t <= t_max, "Sampled t has exceeded t_max. Increase t_max and retry.")
                end
                
                Group_adjoin!(group[c], y[i], X[i], W[i], eta)
                z[i] = c 
                N[c] += 1
            end

            # -------------- Split/merge move --------------
            if use_splitmerge
                t = split_merge!(y,X,W,eta,z,zs,S,group,list,N,t,H,a,b,log_v,n_split,n_merge)
                c_next = ordered_next(list)
                @assert(t <= t_max, "Sampled t has exceeded t_max. Increase t_max and retry.")
            end

            # -------------- update group-specific/common parameters -------------- 
            update_theta!(theta,t,list,group,H)
            update_eta!(y,X,W,eta,theta,z,t,list,n,H)

            for j = 1 : t; c = list[j];
                N_r[c, iteration] = N[c]
                beta_r[c, :, iteration] = theta[c].beta
                phi_r[c, iteration] = theta[c].phi
            end

            t_r[iteration] = t
            eta_r[:, iteration] = eta
            z_r[:, iteration] = z
            
            # for j = 1 : t_max; Group_clear!(group[j]); Theta_clear!(theta[j]); end
            # for j = 1 : t; c = list[j]; for k = 1 : n; 
            #         if z[k] == c; Group_adjoin!(group[c], y[k], X[k], W[k], eta); end; end; 
            # end
            for j = 1:t; c = list[j]; 
               Group_clear!(group[c]); Theta_clear!(theta[c])
               for k = 1:n; if z[k] == c; Group_adjoin!(group[c],y[k],X[k],W[k],eta); end; end; 
            end
            
        end

    end
    
    return t_r, N_r, z_r, beta_r, phi_r, eta_r
end