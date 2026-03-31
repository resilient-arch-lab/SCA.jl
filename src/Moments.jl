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
using FixedSizeArrays

# TODO: I'm not convinced this actually needs to be parameterized on the array type, and it
# does complicate things slightly.
# TODO: This should be able to handle mutli-dimensional labels (e.g. vector labels)
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

# right now this is slower than running a UniVarMomentsAccs for each label element
mutable struct UniVarMomentsAccNDLabel{Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    totals::Tarray  # Tarray is a typevar, which you can't parameterize directly 
    moments::Tarray
    const order::UInt
    const ns::UInt
    const nl::UInt
    const label_shape::NTuple{LD}

    function UniVarMomentsAccNDLabel{Tt, Tl, Tarray, LD}(order, ns, nl, label_shape) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD} 
        totals = fill!(Tarray{UInt32, 1+LD}(undef, label_shape..., nl), 0)
        moments = fill!(Tarray{Tt, 3+LD}(undef, label_shape..., nl, order, ns), 0)
        new{Tt, Tl, Tarray, LD}(totals, moments, order, ns, nl, label_shape)
    end
end

# works on CPU and GPU
# Depricated in favor of AcceleratedKernels kernels (label_wise_sum_ak!)
@kernel function label_wise_sum_shared!(@Const(traces::AbstractMatrix{Tt}), @Const(labels::AbstractVector{Tl}), sums::AbstractMatrix{Tt}, totals::AbstractVector{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
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
    @inbounds AK.foraxes(traces, 1, min_elems=5000) do i
        l_i = convert(Int32, labels[i]+1)
        Atomix.@atomic totals[l_i] += 1
        for j in axes(traces, 2)
            Atomix.@atomic sums[l_i, j] += traces[i, j]
        end
    end
end

function label_wise_sum_ak!(traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
    @inbounds AK.foraxes(traces, 1, min_elems=5000) do i
        for l in axes(labels, 2)
            l_i = convert(Int32, labels[i, l]+1)
            Atomix.@atomic totals[l, l_i] += 1
            for j in axes(traces, 2)
                Atomix.@atomic sums[l, l_i, j] += traces[i, j]
            end
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

function label_wise_sum_scalar!(traces::AbstractVector{Tt}, labels::AbstractVector{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
    @inbounds AK.foraxes(traces, 1, min_elems=5000) do i
        for l in axes(labels, 2)
            l_i = convert(Int32, labels[i, l]+1)
            Atomix.@atomic totals[l, l_i] += 1
            for j in axes(traces, 2)
                Atomix.@atomic sums[l, l_i, j] += traces[i, j]
            end
        end
    end
end

# Centered sum update kernel
# Depricated in favor of AcceleratedKernels kernels (centered_sum_kern_ak!)
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

function centered_sum_kern_ak!(moments::AbstractArray{Tt, 3}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 2)

    @inbounds AK.foraxes(traces, 1, min_elems=5000) do i
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

function centered_sum_kern_ak!(moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 3)

    @inbounds AK.foraxes(traces, 1, min_elems=5000) do i
        for l in axes(labels, 2)
            l_i = convert(Int32, labels[i, l]+1)
            for j in axes(traces, 2)
                t_update = traces[i, j] - moments[l, l_i, 1, j]
                pow = t_update
                for d in 2:order
                    pow *= t_update
                    Atomix.@atomic moments[l, l_i, d, j] += pow
                end
            end
        end
    end
end

function centered_sum_cpu!(moments::AbstractArray{Tt, 3}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 2)
    samples_per_thread = cld(size(traces, 2), Threads.nthreads())
    trace_tiles = tiled_view(traces, (size(traces, 1), samples_per_thread))
    moment_tiles = tiled_view(moments, (size(moments)[1:2]..., samples_per_thread))

    @inbounds Threads.@threads for tile_idx in axes(trace_tiles, 2)
        trace_tile = trace_tiles[1, tile_idx]
        moment_tile = moment_tiles[1, 1, 1, tile_idx]
        for i in axes(trace_tile, 1)
            l_i = convert(Int32, labels[i]+1)
            t_update = trace_tile[i, :] .- moment_tile[l_i, 1, :]
            pow = copy(t_update)

            for d in 2:order
                pow .*= t_update
                moment_tile[l_i, d, :] .+= pow
            end
        end
    end
end

# still VERY slow for some reason
# (insane amount of allocations, figure out from where)
function centered_sum_cpu!(moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 3)
    samples_per_thread = cld(size(traces, 2), Threads.nthreads())
    trace_tiles = tiled_view(traces, (size(traces, 1), samples_per_thread))
    moment_tiles = tiled_view(moments, (size(moments)[1:3]..., samples_per_thread))

    Threads.@threads for tile_idx in axes(trace_tiles, 2)
        @inbounds begin
            trace_tile = trace_tiles[1, tile_idx]
            moment_tile = moment_tiles[1, 1, 1, tile_idx]
            l_i = Vector{UInt32}(undef, size(labels, 2))
            t_update = Array{Tt, 2}(undef, size(l_i, 1), size(trace_tile, 2))
            pow = similar(t_update)
            
            for i in axes(trace_tile, 1)
                l_i .= convert.(Int32, labels[i, :].+1)
                moment_idx = CartesianIndex.(axes(l_i, 1), l_i)

                t_update .= @views reshape(trace_tile[i, :], 1, :) .- moment_tile[moment_idx, 1, :]
                pow .= t_update

                for d in 2:order
                    pow .*= t_update
                    moment_tile[moment_idx, d, :] .+= pow
                end
            end
        end
    end
end

# Update the estimation of centered sums in `acc`
# Note: Tarray must be `Array`, as the struct must live in CPU memory. However, if
# `traces` and `labels` are GPU arrays, as much computation as possible will be done
# on GPU before finalizing results on the CPU.
function centered_sum_update_old!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
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

# works end-to-end on CPU or GPU
function centered_sum_update!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    # initialize intermediate values (these could be allocated on `acc` construction)
    sums = fill!(similar(traces, Tt, acc.nl, acc.ns), 0)
    moments = fill!(similar(traces, Tt, size(acc.moments)), 0)
    totals = fill!(similar(traces, UInt32, size(acc.totals)), 0)

    @boundscheck begin
        checkbounds(sums, acc.nl, size(traces, 2))
        checkbounds(moments, acc.nl, acc.order, size(traces, 2))
    end

    label_wise_sum_ak!(traces, labels, sums, totals)

    # find means
    @. moments[:, 1, :] = sums / totals

    # compute centered sums
    centered_sum_kern_ak!(moments, traces, labels)

    # merge centered sum estimations
    init_ls = acc.totals .== 0
    update_ls = acc.totals .!= 0
    moments = Tarray(moments)  # cast to same memory as acc if not already there
    totals = Tarray(totals)  # cast to same memory as acc if not already there
    if any(init_ls)
        acc.moments[init_ls, :, :] .= moments[init_ls, :, :]  # scalar indexing
        acc.totals[init_ls] .= totals[init_ls]
        # even moving `init_ls` to the same device as `moments` on the second side doesn't make the 
        # indexing non-scalar. However, if acc is on device memory along with input data this works fine.  
    end
    if any(update_ls)
        for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
            merge_from_ak_gpu!(view(acc.moments, l, :, :), view(acc.totals, l), view(moments, l, :, :), view(totals, l))
        end
        acc.totals[update_ls] .+= totals[update_ls]
    end
end

# This is still horrendously slow on CPU, must fix. 
# No overhead compared to scalar labels on GPUs though
function centered_sum_update!(acc::UniVarMomentsAccNDLabel{Tt, Tl, Tarray, LD}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    # Initialize intermediate values
    sums = fill!(similar(traces, Tt, acc.label_shape..., acc.nl, acc.ns), 0)
    moments = fill!(similar(traces, Tt, size(acc.moments)), 0)
    totals = fill!(similar(traces, UInt32, size(acc.totals)), 0)

    @time "label_wise_sum_ak!" label_wise_sum_ak!(traces, labels, sums, totals)

    # find means
    @time "means" @. moments[:, :, 1, :] = sums / totals

    # compute centered sums
    # @time "centered_sum_kern_ak!" centered_sum_kern_ak!(moments2, traces, labels)  # 15s
    @time "centered_sum_kern_ak!" @sync for l in axes(labels, 2)
        Threads.@spawn centered_sum_kern_ak!(view(moments, l, :, :, :), traces, labels[:, l])
    end

    # merge centered sum estimations
    init_ls = acc.totals .== 0
    update_ls = acc.totals .!= 0
    moments = Tarray(moments)  # cast to same memory as acc if not already there
    totals = Tarray(totals)  # cast to same memory as acc if not already there
    if any(init_ls)
        acc.moments[init_ls, :, :] .= moments[init_ls, :, :]
        acc.totals[init_ls] .= totals[init_ls]
    end
    if any(update_ls)
        @sync for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
            Threads.@spawn merge_from_ak_gpu!(view(acc.moments, l, :, :), view(acc.totals, l), view(moments, l, :, :), view(totals, l))
        end
        acc.totals[update_ls] .+= totals[update_ls]
    end
end

function centered_sum_update_combined!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    sums = fill!(similar(traces, Tt, acc.nl, acc.ns), 0)
    moments = fill!(similar(traces, Tt, size(acc.moments)), 0)
    totals = fill!(similar(traces, UInt32, size(acc.totals)), 0)

    traces_tiles = FixedSizeArray.(tiled_view(traces, (size(traces, 1), 1)))
    sums_tiles = FixedSizeArray.(tiled_view(sums, (size(sums, 1), 1)))
    moments_tiles = FixedSizeArray.(tiled_view(moments, (size(moments)[1:2]..., 1)))
    
    @time AK.foraxes(traces, 2) do j
        t_tile = traces_tiles[1, j]
        s_tile = sums_tiles[1, j]
        m_tile = moments_tiles[1, 1, j]
        l_tile = labels
        order = size(m_tile, 2)
        
        # label wise sum
        for i in axes(t_tile, 1)
            l_i = convert(Int32, l_tile[i]+1)
            Atomix.@atomic t_tile[l_i] += 1  # these probably don't have to be atomic since I'm parallelizing over time here
            Atomix.@atomic s_tile[l_i] += t_tile[i]
        end
        
        # compute means
        for l in axes(m_tile, 1)
            m_tile[l] = sums[l] / totals[l]
        end

        # centered sum kernel
        for i in axes(t_tile, 1)
            l_i = convert(Int32, l_tile[i]+1)
            t_update = t_tile[i] - m_tile[l_i, 1]
            pow = t_update
            for d in 2:order
                pow *= t_update
                Atomix.@atomic m_tile[l_i, d] += pow
            end
        end

        # merge from
        # for l in axes(m_tile, 1)
            
        # end
    end

    # merge centered sum estimations
    init_ls = acc.totals .== 0
    update_ls = acc.totals .!= 0
    moments = Tarray(moments)  # cast to same memory as acc if not already there
    totals = Tarray(totals)  # cast to same memory as acc if not already there
    if any(init_ls)
        acc.moments[init_ls, :, :] .= moments[init_ls, :, :]
        acc.totals[init_ls] .= totals[init_ls]
    end
    if any(update_ls)
        @sync for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
            Threads.@spawn merge_from_ak_gpu!(view(acc.moments, l, :, :), view(acc.totals, l), view(moments, l, :, :), view(totals, l))
        end
        acc.totals[update_ls] .+= totals[update_ls]
    end
end

# Precision (even with Float64) seems to degrade from performing the same 
# computation in a single centered_sum_update! for the same data. Use of 
# this should be minimized, prefer larger update batches whenever possible
# TODO: depricate in favor of merge_from_kern!
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
    totals_result = totals_old .+ totals_new
    kern_order = Int(acc.order)

    # I'm pretty sure this can't be threaded like this, because the loop modifies δ_pows
    # Threads.@threads for l_idx in axes(totals_old, 2)
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

function merge_from!(acc::UniVarMomentsAccNDLabel{Tt, Tl, Tarray, 1}, M_new::Array{Tt, 4}, totals_new::Array{UInt32, 2}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    checkbounds(M_new, size(acc.moments)...)
    checkbounds(totals_new, size(acc.totals)...)
    
    if all(totals_new .== 0)
        return nothing
    end
    if all(acc.totals .== 0)
        # If this is the first estimation, the acc values can be updated directly
        acc.moments .= M_new
        acc.totals .= totals_new
        return nothing
    end

    kern_order = Int(acc.order)

    @inbounds for l in axes(acc.totals, 1)
        δ = view(M_new, l, :, 1, :) - view(acc.moments, l, :, 1, :)
        δ_pows = fill!(Tarray{Tt, 2}(undef, acc.order+1, acc.ns), 0)
        M_old_l, totals_old_l = view(acc.moments, l, :, :, :), view(acc.totals, l, :)
        totals_new_l = view(totals_new, l, :)
        totals_result_l = totals_old_l .+ totals_new_l

        for l_idx in axes(totals_old_l, 1)
            M_old_i = view(M_old_l, l_idx, :, :)
            M_new_i = view(M_new, l, l_idx, :, :)

            if totals_new_l[l_idx] == 0
                continue
            end
            if totals_old_l[l_idx] == 0
                M_old_i .= M_new_i
                totals_old_l[l_idx] = totals_new_l[l_idx]
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
                    δ_pows_k = @views δ_pows[k, :]
                    cst = binomial(k, p)
                    tmp2 = view(as_input1, p-k, :) .* ((-totals_new_l[l_idx]/totals_result_l[l_idx]).^k)
                    tmp3 = view(as_input2, p-k, :) .* ((totals_old_l[l_idx]/totals_result_l[l_idx]).^k)
                    to_update1 .+= (δ_pows_k .* cst) .* (tmp2 .+ tmp3)
                end
                tmp = (1/(totals_new_l[l_idx]^(p-1))) - ((-1/totals_old_l[l_idx])^(p-1))
                tmp *= ((totals_old_l[l_idx] * totals_new_l[l_idx])/totals_result_l[l_idx])^p

                to_update1 .+= δ_pows[p, :] .* tmp
            end
            view(M_old_i, 1, :) .+= (view(δ, l_idx, :) .* (totals_new_l[l_idx]/totals_result_l[l_idx]))  # update mean seperately
        end
        totals_old_l .= totals_result_l
    end
    return nothing
end

# Merge a single label estimation
function merge_from_kern!(M_old::AbstractArray{Tt, 2}, total_old::AbstractArray{UInt32, 0}, M_new::AbstractArray{Tt, 2}, total_new::AbstractArray{UInt32, 0}) where { Tt<:AbstractFloat }
    checkbounds(M_new, size(M_old)...)
    checkbounds(total_new, size(total_old)...)

    if total_new[1] == 0
        # println("empty update")
        return nothing
    end
    if total_old[1] == 0
        # println("fresh update")
        M_old .= M_new
        total_old[1] = total_new[1]
        return nothing
    end

    order = size(M_old, 1)
    δ = M_new[1, :] .- M_old[1, :]
    δ_pows = zeros(Tt, order + 1, size(M_old, 2))
    total_result = total_old .+ total_new
    (tmp1, tmp2, tmp3) = (zeros(Tt, size(M_old, 2)) for _ in 1:3)
    @inbounds begin
        for j in axes(δ_pows, 1)
            δ_pows[j, :] .= δ.^j
        end

        for p in order:-1:2
            (as_input1, to_update1) = view(M_old, 1:p-1, :), view(M_old, p, :)
            (as_input2, to_update2) = view(M_new, 1:p-1, :), view(M_new, p, :)

            to_update1 .+= to_update2

            for k in 1:p-2
                cst = binomial(k, p)
                tmp1 .= as_input1[p-k, :] .* ((-total_new[1]/total_result[1])^k)
                tmp2 .= as_input2[p-k, :] .* ((total_old[1]/total_result[1])^k)
                tmp3 .= tmp1 .+ tmp2
                to_update1 .+= (δ_pows[k, :] .* cst) .* tmp3
            end
            tmp1[1] = (1/(total_new[1]^(p-1))) - ((-1/total_old[1])^(p-1))
            tmp1[1] *= ((total_old[1] * total_new[1])/total_result[1])^p

            to_update1 .+= δ_pows[p, :] .* tmp1[1]
        end
        view(M_old, 1, :) .+= (δ .* (total_new[1]/total_result[1]))  # update mean seperately
        total_old[1] = total_result[1]
    end
end

function merge_from_ak!(M_old::AbstractArray{Tt, 2}, total_old::AbstractArray{UInt32, 0}, M_new::AbstractArray{Tt, 2}, total_new::AbstractArray{UInt32, 0}) where { Tt<:AbstractFloat }
    checkbounds(M_new, size(M_old)...)
    checkbounds(total_new, size(total_old)...)
 
    if total_new .== 0
        # println("empty update")
        return nothing
    end
    if total_old .== 0
        # println("fresh update")
        M_old .= M_new
        total_old .= total_new
        return nothing
    end

    order = size(M_old, 1)
    δ = M_new[1, :] .- M_old[1, :]
    δ_pows = zeros(Tt, order + 1, size(M_old, 2))
    total_result = total_old .+ total_new
    (tmp1, tmp2, tmp3) = (zeros(Tt, size(M_old, 2)) for _ in 1:3)
    @inbounds AK.foraxes(M_old, 2, min_elems=1000) do i
        for j in axes(δ_pows, 1)
            δ_pows[j, i] = δ[i]^j
        end

        for p in order:-1:2
            (as_input1, to_update1) = view(M_old, 1:p-1, :), view(M_old, p, :)
            (as_input2, to_update2) = view(M_new, 1:p-1, :), view(M_new, p, :)

            to_update1[i] += to_update2[i]

            for k in 1:p-2
                cst = binomial(k, p)
                tmp1[i] = as_input1[p-k, i] * ((-total_new[1]/total_result[1])^k)
                tmp2[i] = as_input2[p-k, i] * ((total_old[1]/total_result[1])^k)
                tmp3[i] = tmp1[i] + tmp2[i]
                to_update1[i] += (δ_pows[k, i] * cst) * tmp3[i]
            end
            tmp = (1/(total_new[1]^(p-1))) - ((-1/total_old[1])^(p-1))
            tmp *= ((total_old[1] * total_new[1])/total_result[1])^p

            to_update1[i] += δ_pows[p, i] * tmp
        end
        M_old[1, i] += (δ[i] * (total_new[1]/total_result[1]))  # update mean seperately
    end
    @inbounds total_old .= total_result
end

function merge_from_ak_gpu!(M_old::AbstractArray{Tt, 2}, total_old::AbstractArray{UInt32, 0}, M_new::AbstractArray{Tt, 2}, total_new::AbstractArray{UInt32, 0}) where { Tt<:AbstractFloat }
    checkbounds(M_new, size(M_old)...)
    checkbounds(total_new, size(total_old)...)

    order = size(M_old, 1)
    δ_pows = similar(M_old, order + 1, size(M_old, 2))
    # LLVM error: Undefined external symbol "__divti3"
    # missing external symbol for 128bit integer division, probably from `binomial`, fixed by casting `binomial` args to Int32
    @inbounds AK.foraxes(M_old, 2, min_elems=1000) do i
        δ = M_new[1, i] - M_old[1, i]
        total_result = total_old[1] + total_new[1]
        for j in axes(δ_pows, 1)
            δ_pows[j, i] = δ^j
        end

        for p in order:-1:2
            (as_input1, to_update1) = view(M_old, 1:p-1, :), view(M_old, p, :)
            (as_input2, to_update2) = view(M_new, 1:p-1, :), view(M_new, p, :)

            to_update1[i] += to_update2[i]

            for k in 1:p-2
                cst = binomial(Int32(k), Int32(p))  # explicity Int32 cast avoids unnecessary use of arbitrary precision arithmetic 
                tmp1 = as_input1[p-k, i] * ((-total_new[1]/total_result[1])^k)
                tmp2 = as_input2[p-k, i] * ((total_old[1]/total_result[1])^k)
                tmp3 = tmp1 + tmp2
                to_update1[i] += (δ_pows[k, i] * cst) * tmp3
            end
            tmp = (1/(total_new[1]^(p-1))) - ((-1/total_old[1])^(p-1))
            tmp *= ((total_old[1] * total_new[1])/total_result[1])^p

            to_update1[i] += δ_pows[p, i] * tmp
        end
        M_old[1, i] += (δ * (total_new[1]/total_result[1]))  # update mean seperately
    end
    return nothing
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