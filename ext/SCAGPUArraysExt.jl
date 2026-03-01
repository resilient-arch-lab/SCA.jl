module SCAGPUArraysExt

using SCA: Moments
using GPUArrays
using KernelAbstractions

function Moments.centered_sum_update!(acc::Moments.UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractGPUMatrix{Tt}, labels::AbstractGPUVector{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    # Initialize intermediate values
    sums = fill!(similar(traces, Tt, acc.nl, acc.ns), 0)
    moments = fill!(similar(traces, Tt, size(acc.moments)), 0)
    totals = fill!(similar(traces, UInt32, size(acc.totals)), 0)

    # compile kernels
    _label_wise_sum_kern = Moments.label_wise_sum_shared!(get_backend(traces), (4, 64))
    _centered_sum_kern = Moments.centered_sum_kern!(get_backend(moments), (1, 256))

    _label_wise_sum_kern(traces, labels, sums, totals, ndrange=size(traces))
    # label_wise_sum_cpu!(traces, labels, sums, totals)
    # KernelAbstractions.synchronize(get_backend(sums))

    # find means
    @. moments[:, 1, :] = sums / totals

    # compute centered sums
    _centered_sum_kern(moments, traces, labels, ndrange=size(traces))
    # KernelAbstractions.synchronize(get_backend(sums))

    # This has to be performed on CPU for now, its a pretty complicated OP
    Moments.merge_from!(acc, Tarray(moments), Tarray(totals))
end




end