module MultiVarMoments
export MultiVarMomentsAcc, centered_sum_multivar!

include("Utils.jl")
include("Moments.jl")
using .Utils, .Moments

# REFERENCE FUNCTIONS FOR TESTING
struct MultiVarMomentsAccReference{Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    totals::Ta
    SCPs::Ta  # sums of centered products
    α::Vector{UInt}  # order vector
    ns::UInt  # number of samples per trace (and therefore the variateness of sums of centered prods)
    nl::UInt
    _totals::Ta
    _SCPs::Ta
    _sums::Ta

    function MultiVarMomentsAccReference{Tt, Tl, Ta}(order::UInt, ns, nl) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
        totals = fill!(Ta{UInt32, 1}(undef, nl), 0)
        SCPs = fill!(Ta{Tt, 3}(undef, nl, 1, 1), 0)
        α = fill!(Ta{UInt, 1}(undef, ns), order)  # the same order is calculated for each sample position 
        _totals = similar(totals)
        _SCPs = similar(SCPs)
        _sums = Ta{Tt, 2}(undef, nl, ns)
        new(totals, SCPs, α, ns, nl, _totals, _SCPs, _sums)
    end
end

function _centered_sum_multivar_reference!(SCPs::AbstractArray{Tt, 3}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, order::AbstractVector{UInt}, means::AbstractMatrix{Tt}) where {Tt<:AbstractFloat, Tl<:Integer}
    @boundscheck begin
        checkbounds(traces, 1, size(order, 1))
        checkbounds(order, size(traces, 2))
    end
    
    for i in axes(traces, 1)
        l_i = convert(Int32, labels[i]+1)
        cp = prod((traces[i, :] .- means[l_i, :]) .^ order)
        SCPs[l_i, 1, 1] += cp
    end
end

function _centered_sum_update_reference!(acc::MultiVarMomentsAccReference{Tt, Tl, Ta}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    fill!(acc._sums, 0)
    fill!(acc._totals, 0)
    fill!(acc._SCPs, 0)
    
    # Pass 1, calculate labels wise sums
    Moments.label_wise_sum_ak!(traces, labels, acc._sums, acc._totals)

    # Pass 2: find means and calculate sums of centered prods
    means = acc._sums ./ acc._totals
    _centered_sum_multivar_reference!(acc._SCPs, traces, labels, acc.α, means)

    acc.SCPs .= acc._SCPs
    acc.totals .= acc._totals
end

# PRACTICAL IMPLEMENTATIONS
struct MultiVarMomentsAcc{Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    totals::Ta
    SCPs::Ta  # sums of centered products
    α::Matrix{Int}  # order vectors (vector rows)
    ns::UInt  # number of samples per trace (and therefore the variateness of sums of centered prods)
    nl::UInt
    _totals::Ta
    _SCPs::Ta
    _sums::Ta

    function MultiVarMomentsAcc{Tt, Tl, Ta}(order::Union{Int, AbstractVector{Int}, AbstractMatrix{Int}}, ns::Integer, nl::Integer) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
        if typeof(order) == Int
            α = fill!(Ta{Int, 2}(undef, 1, ns), order)  # the same order is calculated for each sample position 
        elseif typeof(order) <: AbstractVector{Int}
            α = reshape(order, 1, ns)
        else typeof(order) <: AbstractMatrix{Int}
            checkbounds(order, 1, ns)
            α = order
        end
        
        totals = fill!(Ta{UInt32, 1}(undef, nl), 0)
        SCPs = fill!(Ta{Tt, 3}(undef, nl, size(α, 1), 1), 0)

        _totals = similar(totals)
        _SCPs = similar(SCPs)
        _sums = Ta{Tt, 2}(undef, nl, ns)
        new(totals, SCPs, α, ns, nl, _totals, _SCPs, _sums)
    end
end

function centered_sum_multivar!(SCPs::AbstractArray{Tt, 3}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, order::AbstractMatrix{Int}, means::AbstractMatrix{Tt}) where {Tt<:AbstractFloat, Tl<:Integer}
    @boundscheck begin
        # make sure order vector and measurement vector are of equal length
        checkbounds(traces, 1, size(order, 2)); checkbounds(order, 1, size(traces, 2))  
    end
    
    for i in axes(traces, 1)
        l_i = convert(Int32, labels[i]+1)
        centered = traces[i, :] .- means[l_i, :]
        for oi in axes(order, 1)
            centered_product = prod(centered .^ order[oi, :])
            SCPs[l_i, oi, 1] += centered_product
        end
    end
end

function centered_sum_update!(acc::MultiVarMomentsAcc{Tt, Tl, Ta}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    fill!(acc._sums, 0)
    fill!(acc._totals, 0)
    fill!(acc._SCPs, 0)
    
    # Pass 1, calculate labels wise sums
    Moments.label_wise_sum_ak!(traces, labels, acc._sums, acc._totals)

    # Pass 2: find means and calculate sums of centered prods
    means = acc._sums ./ acc._totals
    centered_sum_multivar!(acc._SCPs, traces, labels, acc.α, means)

    acc.SCPs .= acc._SCPs
    acc.totals .= acc._totals
end





end