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
    
    Moments.label_wise_sum_ak!(traces, labels, test.sums, test.totals)
    
    # assumes ldim = 1 for now
    x1bar = test.sums[1, 1, :] ./ test.totals[1, 1]; x2bar = test.sums[1, 2, :] ./ test.totals[1, 2]
    
    # V could these me computed with the moments accumulator? They should be able to be
    Σ1 = mapreduce(x -> ((x.-x1bar)*transpose(x.-x1bar)), .+, eachrow(traces[labels[:, 1] .== 0, :]), init=zeros(Tt, length(x1bar), length(x1bar)))
    Σ2 = mapreduce(x -> ((x.-x2bar)*transpose(x.-x2bar)), .+, eachrow(traces[labels[:, 2] .== 1, :]), init=zeros(Tt, length(x2bar), length(x2bar)))
    
    Σpooled = (((test.totals[1, 1] - 1).*Σ1) .+ ((test.totals[1, 2] - 1).*Σ2)) ./ (test.totals[1, 1] + test.totals[1, 2] - 2)
    tmp1 = prod(test.totals[1, :]) / sum(test.totals[1, :])
    tmp2 = x1bar .- x2bar
    res = tmp1 .* (transpose(tmp2) * inv(Σpooled) * tmp2)
    
    return res
end



end