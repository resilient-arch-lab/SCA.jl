"""
Parallel estimation of statistical moments. based on the implementation from
[SCALib](https://github.com/simple-crypto/SCALib).
"""

module Moments
export UniVarMomentsAcc, centered_sum_update!, merge_from!, get_mean_and_var

include("Utils.jl")
using .Utils

using Random
using KernelAbstractions, Atomix
import AcceleratedKernels as AK
using Base: convert

# TODO: I'm not convinced this actually needs to be parameterized on the array type, and it
# does complicate things slightly.
mutable struct UniVarMomentsAcc{Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    totals::Tarray  # Tarray is a typevar, which you can't parameterize directly 
    moments::Tarray
    const order::UInt
    const ns::UInt
    const nl::UInt

    function UniVarMomentsAcc{Tt, Tl, Tarray}(order, ns, nl) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
        totals = fill!(Tarray{UInt32, 1}(undef, nl), 0)
        moments = fill!(Tarray{Tt, 3}(undef, nl, order, ns), 0)
        new(totals, moments, order, ns, nl)
    end
end

# works on CPU and GPU
@kernel function label_wise_sum_shared!(traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, sums::AbstractMatrix{Tt}, totals::AbstractVector{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
    I, J = @index(Global, NTuple)  # I: trace, J: y_offset
    i, j = @index(Local, NTuple)

    nt = @uniform @groupsize()[1]  # this is compile time constant if the kernel is compiled with a static workgroup size
    ns = @uniform @groupsize()[2]
    t_sh = @localmem Tt (nt, ns)
    l_sh = @localmem Int32 nt
    @inbounds t_sh[i, j] = convert(Tt, traces[I, J])
    if j == 1
        @inbounds l_sh[i] = convert(Int32, labels[I]+1)
    end
    @synchronize()

    @inbounds l_idx = l_sh[i]
    @inbounds Atomix.@atomic sums[l_idx, J] += t_sh[i, j]
    if J == 1
        @inbounds Atomix.@atomic totals[l_idx] += 1
    end
end

# Works on CPU and GPU
function label_wise_sum_ak!(traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, sums::AbstractMatrix{Tt}, totals::AbstractVector{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
    @inbounds AK.foraxes(traces, 1) do i
        l_i = convert(Int32, labels[i]+1)
        Atomix.@atomic totals[l_i] += 1
        for j in axes(traces, 2)
            Atomix.@atomic sums[l_i, j] += traces[i, j]
        end
    end
end

function label_wise_sum_cpu!(traces::AbstractArray, labels::AbstractArray, sums::AbstractArray, totals::AbstractArray)
    # each thread gets a column of data
    samples_per_thread = cld(size(traces, 2), Threads.nthreads())
    trace_tiles = tiled_view(traces, (size(traces, 1), samples_per_thread))
    sum_tiles = tiled_view(sums, (size(traces, 1), samples_per_thread))
    
    Threads.@threads for tile_idx in axes(trace_tiles, 2)
        trace_tile = trace_tiles[1, tile_idx]
        sum_tile = sum_tiles[1, tile_idx]

        if tile_idx == 1
            for trace in axes(traces, 1)
                @inbounds l = Int(labels[trace]) + 1
                @inbounds totals[l] += 1
            end
        end

        for trace in axes(traces, 1)
            @inbounds l = Int(labels[trace]) + 1
            @inbounds @views sum_tile[l, :] .+= trace_tile[trace, :]
        end
    end
end

# Centered sum update kernel
@kernel function centered_sum_kern!(moments::AbstractArray{Tt, 3}, traces::AbstractArray, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    i, j = @index(Local, NTuple)  # i: assumed to be 1, j: trace_y_offset_local
    I, J = @index(Global, NTuple)  # I: trace, J: trace_y_offset_global

    order = @uniform size(moments, 2)
    tmp_shape = @uniform @groupsize()
    t = @localmem Tt tmp_shape
    pow = @localmem Tt tmp_shape

    @inbounds @private l_i = unsafe_trunc(Int, labels[I]+1)
    @inbounds @private t_i = convert(Tt, traces[I, J])

    @inbounds @private t_update = t_i - moments[l_i, 1, J]
    @inbounds t[i, j] = t_update
    @inbounds pow[i, j] = t[i, j]

    for d in 2:order
        @inbounds pow[i, j] *= t[i, j]
        @inbounds Atomix.@atomic moments[l_i, d, J] += pow[i, j]
    end
end

function centered_sum_kern_ak!(moments::AbstractArray{Tt, 3}, traces::AbstractArray, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 2)

    @inbounds AK.foraxes(traces, 1) do i
        l_i = convert(Int32, labels[i]+1)
        for j in axes(traces, 2)
            t_update = traces[i, j] - moments[l_i, 1, j]
            pow = t_update
            for d in 2:order
                pow *= t_update
                Atomix.@atomic moments[l_i, d, j] += pow
            end
        end
    end

end

# Update the estimation of centered sums in `acc`
# Note: Tarray must be `Array`, as the struct must live in CPU memory. However, if
# `traces` and `labels` are GPU arrays, as much computation as possible will be done
# on GPU before finalizing results on the CPU.
function centered_sum_update!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    # Initialize intermediate values
    sums = fill!(similar(traces, Tt, acc.nl, acc.ns), 0)
    moments = fill!(similar(traces, Tt, size(acc.moments)), 0)
    totals = fill!(similar(traces, UInt32, size(acc.totals)), 0)

    # compile kernels
    _label_wise_sum_kern = label_wise_sum_shared!(get_backend(traces), (4, 64))
    _centered_sum_kern = centered_sum_kern!(get_backend(moments), (1, 256))

    _label_wise_sum_kern(traces, labels, sums, totals, ndrange=size(traces))
    # label_wise_sum_cpu!(traces, labels, sums, totals)
    # KernelAbstractions.synchronize(get_backend(sums))

    # find means
    @. moments[:, 1, :] = sums / totals

    # compute centered sums
    _centered_sum_kern(moments, traces, labels, ndrange=size(traces))
    # KernelAbstractions.synchronize(get_backend(sums))

    # This has to be performed on CPU for now, its a pretty complicated OP
    merge_from!(acc, Tarray(moments), Tarray(totals))
end

function centered_sum_update_ak!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    # Initialize intermediate values
    sums = fill!(similar(traces, Tt, acc.nl, acc.ns), 0)
    moments = fill!(similar(traces, Tt, size(acc.moments)), 0)
    totals = fill!(similar(traces, UInt32, size(acc.totals)), 0)

    label_wise_sum_ak!(traces, labels, sums, totals)

    # find means
    @. moments[:, 1, :] = sums / totals

    # compute centered sums
    centered_sum_kern_ak!(moments, traces, labels)

    # This has to be performed on CPU for now, its a pretty complicated OP
    merge_from!(acc, Tarray(moments), Tarray(totals))
end

# Precision (even with Float64) seems to degrade from performing the same 
# computation in a single centered_sum_update! for the same data. Use of 
# this should be minimized, prefer larger update batches whenever possible
function merge_from!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, M_new::Array{Tt, 3}, totals_new::Array{UInt32, 1}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    if all(totals_new .== 0)
        return nothing
    end
    if all(acc.totals .== 0)
        # If this is the first estimation, the acc values can be updated directly
        acc.moments .= M_new
        acc.totals .= totals_new
        return nothing
    end

    δ = view(M_new, :, 1, :) - view(acc.moments, :, 1, :)
    δ_pows = fill!(Tarray{Tt, 2}(undef, acc.order+1, acc.ns), 0)
    M_old, totals_old = view(acc.moments, :, :, :), view(acc.totals, :)
    totals_result = acc.totals .+ totals_new
    kern_order = Int(acc.order)

    for l_idx in axes(totals_old, 1)
        M_old_i = view(M_old, l_idx, :, :)
        M_new_i = view(M_new, l_idx, :, :)

        if totals_new[l_idx] == 0
            continue
        end
        if totals_old[l_idx] == 0
            M_old_i .= M_new_i
            totals_old[l_idx] = totals_new[l_idx]
            continue
        end

        for j in axes(δ_pows, 1)
            view(δ_pows, j, :) .= view(δ, l_idx, :).^j
        end
        
        for p in kern_order:-1:2
            (as_input1, to_update1) = view(M_old_i, 1:p-1, :), view(M_old_i, p, :)
            (as_input2, to_update2) = view(M_new_i, 1:p-1, :), view(M_new_i, p, :)

            to_update1 .+= to_update2

            for k in 1:p-2
                δ_pows_k = δ_pows[k, :]
                cst = binomial(k, p)
                tmp2 = view(as_input1, p-k, :) .* ((-totals_new[l_idx]/totals_result[l_idx]).^k)
                tmp3 = view(as_input2, p-k, :) .* ((totals_old[l_idx]/totals_result[l_idx]).^k)
                x = tmp2 .+ tmp3
                to_update1 .+= (δ_pows_k .* cst) .* x
            end
            tmp = (1/(totals_new[l_idx]^(p-1))) - ((-1/totals_old[l_idx])^(p-1))
            tmp *= ((totals_old[l_idx] * totals_new[l_idx])/totals_result[l_idx])^p

            to_update1 .+= δ_pows[p, :] .* tmp
        end
        view(M_old_i, 1, :) .+= (view(δ, l_idx, :) .* (totals_new[l_idx]/totals_result[l_idx]))  # update mean seperately
    end
    totals_old .= totals_result
    return nothing
end

function merge_from!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, acc_new::UniVarMomentsAcc{Tt, Tl, Tarray}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    merge_from!(acc, acc_new.moments, acc_new.totals)
end

function get_mean_and_var(m::UniVarMomentsAcc, d::Int)
    if d == 1
        @inbounds μ = @view m.moments[:, 1, :]
        @inbounds σ2 = m.moments[:, 2, :] ./ m.totals
        return μ, σ2
    elseif d == 2
        @inbounds μ = m.moments[:, 2, :] ./ m.totals
        @inbounds σ2 = m.moments[:, 4, :] ./ m.totals
        return μ, σ2
    elseif d > 2
        @inbounds μ = (m.moments[:, d, :] ./ m.totals) ./ ((m.moments[:, 2, :] ./ m.totals).^(d/2))
        @inbounds σ2 = ((m.moments[:, 2*d, :] ./ m.totals) .- ((m.moments[:, d, :] ./ m.totals).^2)) ./ ((m.moments[:, 2, :] ./ m.totals).^d)
        return μ, σ2
    end
end

end  # module Moments