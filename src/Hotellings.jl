module Hotellings
export HotellingsT2, hotellings_t2

include("Moments.jl")
using .Moments
using Statistics
using LinearAlgebra


struct HotellingsT2{Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    sums::Ta
    totals::Ta
    
end

function HotellingsT2{Tt, Tl, Ta}(ns, ldim) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    sums = fill!(Ta{Tt, 3}(undef, ldim, 2, ns), 0)
    totals = fill!(Ta{Int32, 2}(undef, ldim, 2), 0)

    HotellingsT2{Tt, Tl, Ta}(sums, totals)
end


function hotellings_t2(test::HotellingsT2{Tt, Tl, Ta}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    
    
    # Moments.label_wise_sum_ak!(traces, labels, test.sums, test.totals)
    
    # tmp1 = prod(test.totals) / sum(test.totals)
    # @views tmp2 = (test.sums[:, :, 1] ./ test.totals[:, 1]) .- (test.sums[:, :, 2] ./ test.totals[:, 2])
    # @views tmp2 = (test.sums[:, :, 1] ./ test.totals[:, 1]) .- (test.sums[:, :, 2] ./ test.totals[:, 2])
    res = tmp1 .* (transpose(tmp2) * inv(cov(tmp2)))
    return res
end



end