struct Options
    mode::String
    model_type::String
    y::Array{Float64,1}
    X::Array{Array{Float64,1},1}
    W::Array{Array{Float64,1},1}
    n_total::Int64
    n_burn::Int64
    verbose::Bool # display information or not
    t_max::Int64
    Initial_grouping::String
    
    # MFM options
    gamma::Float64
    log_pk::Function
    
    # DPM options
    alpha::Float64
    
    # Jain-Neal split-merge options
    use_splitmerge::Bool
    n_split::Int64
    n_merge::Int64
    
    # Partition distribution values
    a::Float64
    b::Float64
    log_v::Array{Float64,1}
    
    # Other
    n::Int64
end

struct Result
    options::Options
    t::Array{Int64, 1}
    N::Array{Int64, 2}
    z::Array{Int64, 2}
    beta::Array{Float64, 3}
    phi::Array{Float64, 2}
    eta::Array{Float64, 2}
    elapsed_time::Float64
    time_per_step::Float64
end