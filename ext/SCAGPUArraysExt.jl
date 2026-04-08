module SCAGPUArraysExt

using SCA
using SCA.Moments: UniVarMomentsAcc, label_wise_sum_ak!, centered_sum_kern_ak_transposed!, merge_from_ak!
using GPUArrays
using KernelAbstractions

function centered_sum_update!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractGPUArray{Tt}, labels::AbstractGPUArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractGPUArray}
    # initialize intermediate values (these could be allocated on `acc` construction)
    fill!(acc._sums, 0)
    fill!(acc._moments, 0)
    fill!(acc._totals, 0)

    if get_backend(traces) != get_backend(acc._sums) || get_backend(labels) != get_backend(acc._sums)
        traces = Tarray(traces)
        labels = Tarray(labels)
    end

    @boundscheck begin
        checkbounds(acc._sums, acc.nl, size(traces, 2))
        checkbounds(acc._moments, acc.nl, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1))
    end

    label_wise_sum_ak!(traces, labels, acc._sums, acc._totals)

    # find means
    @. acc._moments[:, 1, :] = acc._sums / acc._totals

    # compute centered sums
    # centered_sum_kern_ak!(acc._moments, traces, labels)
    # about 30% of centered_sum_update! runtime
    centered_sum_kern_ak_transposed!(acc._moments, traces, labels)

    # merge centered sum estimations
    init_ls = acc.totals .== 0
    update_ls = acc.totals .!= 0
    if any(init_ls)
        @inbounds acc.moments[init_ls, :, :] .= acc._moments[init_ls, :, :]
        @inbounds acc.totals[init_ls] .= acc._totals[init_ls]
    end
    if any(update_ls)
        Threads.@threads for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
            @inbounds merge_from_ak!(view(acc.moments, l, :, :), view(acc.totals, l), view(acc._moments, l, :, :), view(acc._totals, l))
            # roughly 40% of centered_sum_update! runtime (was 60 before I removed the δ_pows allocation)
            # Also, this is runtime dispatched and garbage collected?
        end
        @inbounds acc.totals[update_ls] .+= acc._totals[update_ls]
    end
end


end