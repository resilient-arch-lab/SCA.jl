module MultiVarMoments
export MultiVarMomentsAcc, centered_sum_multivar!

include("Utils.jl")
include("Moments.jl")
using .Utils, .Moments

struct MultiVarMomentsAcc{Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    totals::Ta
    SCPs::Ta  # sums of centered products
    α::Vector{UInt}
    ns::UInt  # number of samples per trace (and therefore the variateness of sums of centered prods)
    nl::UInt
    _totals::Ta
    _SCPs::Ta
    _sums::Ta

    function MultiVarMomentsAcc{Tt, Tl, Ta}(order::UInt, ns, nl) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
        totals = fill!(Ta{UInt32, 1}(undef, nl), 0)
        SCPs = fill!(Ta{Tt, 3}(undef, nl, 1, 1), 0)
        α = fill!(Ta{UInt, 1}(undef, ns), order)  # the same order is calculated for each sample position 
        _totals = similar(totals)
        _SCPs = similar(SCPs)
        _sums = Ta{Tt, 2}(undef, nl, ns)
        new(totals, SCPs, α, ns, nl, _totals, _SCPs, _sums)
    end
end

function centered_sum_multivar!(SCPs::AbstractArray{Tt, 3}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, order::AbstractVector{UInt}, means::AbstractMatrix{Tt}) where {Tt<:AbstractFloat, Tl<:Integer}
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