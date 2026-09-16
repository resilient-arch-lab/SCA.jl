"""
Parallel estimation of statistical moments. based on the implementation from
[SCALib](https://github.com/simple-crypto/SCALib).
"""

module Moments
export UniVarMomentsAcc, centered_sum_update!, merge_from!, get_mean_and_var, UniVarMomentsAccVecLabel, centered_sum_update_pass_1!, centered_sum_update_pass_2!

include("Utils.jl")
using .Utils

using Random
using KernelAbstractions, Atomix
import AcceleratedKernels as AK
using Base: convert

# TODO: I'm not convinced this actually needs to be parameterized on the array type, and it
# does complicate things slightly.
# TODO: This should be able to handle mutli-dimensional labels (e.g. vector labels)
struct UniVarMomentsAcc{Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    totals::Tarray
    moments::Tarray
    order::UInt
    ns::UInt
    nl::UInt
    _totals::Tarray
    _moments::Tarray
    _sums::Tarray

    function UniVarMomentsAcc{Tt, Tl, Tarray}(order, ns, nl) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
        totals = fill!(Tarray{UInt32, 1}(undef, nl), 0)
        moments = fill!(Tarray{Tt, 3}(undef, nl, order, ns), 0)
        _totals = similar(totals)
        _moments = similar(moments)
        _sums = Tarray{Tt, 2}(undef, nl, ns)
        new(totals, moments, order, ns, nl, _totals, _moments, _sums)
    end
end

struct UniVarMomentsAccVecLabel{Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    totals::Tarray
    moments::Tarray
    order::UInt
    ns::UInt
    nl::UInt
    _totals::Tarray
    _moments::Tarray
    _sums::Tarray

    function UniVarMomentsAccVecLabel{Tt, Tl, Tarray, LD}(order, ns, nl) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD} 
        totals = fill!(Tarray{UInt32, 2}(undef, LD, nl), 0)
        moments = fill!(Tarray{Tt, 4}(undef, LD, nl, order, ns), 0)
        _totals = fill!(similar(totals), 0)
        _moments = fill!(similar(moments), 0)
        _sums = fill!(Tarray{Tt, 3}(undef, LD, nl, ns), 0)
        new{Tt, Tl, Tarray, LD}(totals, moments, order, ns, nl, _totals, _moments, _sums)
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

# Faster on CPU than non-transposed
function label_wise_sum_ak_transposed!(traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, sums::AbstractMatrix{Tt}, totals::AbstractVector{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
    @inbounds AK.foraxes(traces, 2) do j
        for i in axes(traces, 1)
            l_i = convert(Int32, labels[i]+1)
            if j == 1
                totals[l_i] += 1
            end
            sums[l_i, j] += traces[i, j]
        end
    end
end

function label_wise_sum_ak_transposed!(traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, nl::Int)::Tuple{AbstractMatrix{Tt}, AbstractVector{UInt32}} where {Tt<:AbstractFloat, Tl<:Integer}
    sums = zeros(eltype(traces), nl, size(traces, 2))
    totals = zeros(UInt32, nl)
    
    @inbounds AK.foraxes(traces, 2) do j
        for i in axes(traces, 1)
            l_i = convert(Int32, labels[i]+1)
            if j == 1
                totals[l_i] += 1
            end
            sums[l_i, j] += traces[i, j]
        end
    end

    return sums, totals
end

function label_wise_sum_ak_transposed!(traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
    @inbounds AK.foraxes(traces, 2) do j
        for i in axes(traces, 1)
            for l in axes(labels, 2)
                l_i = convert(Int32, labels[i, l]+1)
                if j == 1
                    totals[l, l_i] += 1
                end
                sums[l, l_i, j] += traces[i, j]
            end
        end
    end
end

# For multi-element labels
function label_wise_sum_ak!(traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
    @inbounds AK.foraxes(traces, 1) do i
        for l in axes(labels, 2)
            l_i = convert(Int32, labels[i, l]+1)
            Atomix.@atomic totals[l, l_i] += 1
            for j in axes(traces, 2)
                Atomix.@atomic sums[l, l_i, j] += traces[i, j]
            end
        end
    end
end

# simple, sequential label-wise sum op for cpu
@inline function label_wise_sum!(traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix) where {Tt<:AbstractFloat, Tl<:Integer}
    for i in axes(traces, 1)
        for l in axes(labels, 2)
            l_i = convert(Int, labels[i, l])+1
            for j in axes(traces, 2)
                sums[l, l_i, j] += traces[i, j]
            end
            totals[l, l_i] += 1
        end
    end
end

function centered_sum_kern_ak!(moments::AbstractArray{Tt, 3}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 2)

    @inbounds AK.foraxes(traces, 1) do i
        l_i = convert(Int32, labels[i]+1)
        for j in axes(traces, 2)
            t_update = traces[i, j] - moments[l_i, 1, j]
            pow = t_update
            for d in 2:order
                pow *= t_update
                Atomix.@atomic moments[l_i, d, j] += pow  # this line is like 90% of this functions runtime
            end
        end
    end
end

# way better CPU performance (and better GPU performance) than non
# transposed version due to elimination of atomic adds
function centered_sum_kern_ak_transposed!(moments::AbstractArray{Tt, 3}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 2)

    @inbounds AK.foraxes(traces, 2) do j
        for i in axes(traces, 1)
            l_i = convert(Int32, labels[i]+1)
            t_update = traces[i, j] - moments[l_i, 1, j]
            pow = t_update
            for d in 2:order
                pow *= t_update
                moments[l_i, d, j] += pow
            end
        end
    end
end

function centered_sum_kern_ak!(moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 3)
    itr_view = @view moments[:, 1, 1, :]

    @inbounds AK.foreachindex(itr_view) do idx
        (l, j) = CartesianIndices(itr_view)[idx].I
        for ti in axes(traces, 1)
            t_i = traces[ti, j]
            l_i = convert(Int32, labels[ti, l]+1)
            t_update = t_i - moments[l, l_i, 1, j]
            pow = t_update
            for d in 2:order
                pow *= t_update
                moments[l, l_i, d, j] += pow
            end
        end
    end
end

function centered_sum_kern_ak_atomic!(moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 3)

    @inbounds AK.foreachindex(traces) do idx
        (i, j) = CartesianIndices(traces)[idx].I
        t_i = traces[i, j]
        for l in axes(moments, 1)
            l_i = convert(Int32, labels[i, l]+1)
            t_update = t_i - moments[l, l_i, 1, j]
            pow = t_update
            for d in 2:order
                pow *= t_update
                Atomix.@atomic moments[l, l_i, d, j] += pow
            end
        end
    end
end

# simple, sequential centered sum update op for cpu
@inline function centered_sum!(moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 3)
    
    for i in axes(traces, 1)
        for l in axes(labels, 2)
            l_i = convert(Int, labels[i, l])+1
            for j in axes(traces, 2)
                t_update = traces[i, j] - moments[l, l_i, 1, j]
                pow = t_update
                for d in 2:order
                    pow *= t_update
                    moments[l, l_i, d, j] += pow
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
    merge_from_old!(acc, Tarray(moments), Tarray(totals))
end

# works end-to-end on CPU or GPU
# TODO: Figure out why this segfaults with AMDGPU when Tarray is ROCArray
#   Works with CUDA, weird...
#   - It happens during merging, on init
function centered_sum_update!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    fill!(acc._sums, 0)
    fill!(acc._moments, 0)
    fill!(acc._totals, 0)

    if get_backend(traces) != get_backend(acc._sums)
        traces = Tarray(traces)
    end
    if get_backend(labels) != get_backend(acc._sums)
        labels = Tarray(labels)
    end

    @boundscheck begin
        checkbounds(acc._sums, acc.nl, size(traces, 2))
        checkbounds(acc._moments, acc.nl, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1))
    end

    label_wise_sum_ak_transposed!(traces, labels, acc._sums, acc._totals)

    # find means
    @. acc._moments[:, 1, :] = acc._sums / acc._totals

    # compute centered sums
    centered_sum_kern_ak_transposed!(acc._moments, traces, labels)  # about 30% of centered_sum_update! runtime

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
    return nothing
end

# First pass in two pass approach
function centered_sum_update_pass_1!(acc::UniVarMomentsAccVecLabel{Tt, Tl, Tarray, LD}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    @boundscheck begin
        checkbounds(acc._sums, LD, acc.nl, size(traces, 2))
        checkbounds(acc._moments, LD, acc.nl, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), LD)
    end
    
    fill!(acc._moments, 0)
    fill!(acc._sums, 0)
    fill!(acc._totals, 0)

    label_wise_sum_ak_transposed!(traces, labels, acc._sums, acc._totals)

    return
end

function centered_sum_update_pass_1!(sums::AbstractArray{Tt}, totals::AbstractArray{UInt32}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    label_wise_sum_ak_transposed!(traces, labels, sums, totals)
    return
end

# TODO: Make this support merging
# Second pass in two pass approach
function centered_sum_update_pass_2!(acc::UniVarMomentsAccVecLabel{Tt, Tl, Tarray, LD}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    @boundscheck begin
        checkbounds(acc._sums, LD, acc.nl, size(traces, 2))
        checkbounds(acc._moments, LD, acc.nl, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), LD)
    end

    @. acc._moments[:, :, 1, :] = acc._sums / acc._totals

    centered_sum_kern_ak!(acc._moments, traces, labels)

    # merge centered sum estimations
    init_ls = acc.totals .== 0
    update_ls = acc.totals .!= 0
    if any(init_ls)
        @inbounds @views acc.moments[init_ls, :, :] .= acc._moments[init_ls, :, :]
        @inbounds @views acc.totals[init_ls] .= acc._totals[init_ls]
    end
    if any(update_ls)
        for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
            @inbounds merge_from_ak!(view(acc.moments, l, :, :), view(acc.totals, l), view(acc._moments, l, :, :), view(acc._totals, l))
        end
        @inbounds @views acc.totals[update_ls] .+= acc._totals[update_ls]
    end

    return
end

function centered_sum_update_pass_2!(moments::AbstractArray{Tt}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    centered_sum_kern_ak!(moments, traces, labels)
    return
end

function centered_sum_update!(acc::UniVarMomentsAccVecLabel{Tt, Tl, Tarray, LD}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    centered_sum_update_pass_1!(acc, traces, labels)
    centered_sum_update_pass_2!(acc, traces, labels)
end

# This scratch storage uses up a ton of memory, runtime is a lot slower than AK implementations.
function centered_sum_update(traces::Matrix{Tt}, labels::Matrix{Tl}, nl::Int, order::Int)::Array{Tt, 4} where {Tt<:AbstractFloat, Tl<:Integer}
    tile_size = (max(4096, cld(size(traces, 1), cld(Threads.nthreads(), sizeof(Tt)))), cld(1024, sizeof(Tt)))
    trace_tiles = Utils.tiled_view(traces, tile_size)
    label_tiles = Utils.tiled_view(labels, (tile_size[1], size(labels, 2)))

    # pre-allocate buffers
    sums = Array{Tt, 3}(undef, size(labels, 2), nl, size(traces, 2))
    totals = Matrix{Int}(undef, size(labels, 2), nl)
    moments = fill!(Array{Tt, 4}(undef, size(labels, 2), nl, order, size(traces, 2)), 0)
    sums_scratch = [fill!(similar(trace_tiles[x, y], size(labels, 2), nl, size(trace_tiles[x, y], 2)), 0) for x in axes(trace_tiles, 1), y in axes(trace_tiles, 2)]
    totals_scratch = [fill!(Matrix{Int}(undef, size(labels, 2), nl), 0) for _ in axes(trace_tiles, 1), _ in axes(trace_tiles, 2)]

    @sync for itile in axes(trace_tiles, 1)
        for jtile in axes(trace_tiles, 2)
            Threads.@spawn @views label_wise_sum!(trace_tiles[itile, jtile], label_tiles[itile, 1], sums_scratch[itile, jtile], totals_scratch[itile, jtile])
        end
    end
    sums .= cat(sum(sums_scratch, dims=(1))..., dims=(3))
    totals .= sum(@view(totals_scratch[:, 1]))

    # calculate means
    @views moments[:, :, 1, :] .= sums ./ totals
    moments_scratch = [moments[:, :, :, ((y-1)*tile_size[2])+1:(min(y*tile_size[2], size(moments, 4)))] for x in axes(trace_tiles, 1), y in axes(trace_tiles, 2)]
    
    @sync for itile in axes(trace_tiles, 1)
        for jtile in axes(trace_tiles, 2)
            Threads.@spawn @views centered_sum!(moments_scratch[itile, jtile], trace_tiles[itile, jtile], label_tiles[itile, 1])
        end
    end

    @views moments[:, :, 2:end, :] .= cat(sum(map(m -> m[:, :, 2:end, :], moments_scratch), dims=(1))..., dims=(4))

    return moments
end

# Precision (even with Float64) seems to degrade from performing the same 
# computation in a single centered_sum_update! for the same data. Use of 
# this should be minimized, prefer larger update batches whenever possible
function merge_from_old!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, M_new::Array{Tt, 3}, totals_new::Array{UInt32, 1}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
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

# TODO: This seems to be consistently innaccurate, not due to floating point precision issues. I should
# figure out why that is.
function merge_from_ak!(M_old::AbstractArray{Tt, 2}, total_old::AbstractArray{UInt32, 0}, M_new::AbstractArray{Tt, 2}, total_new::AbstractArray{UInt32, 0}) where { Tt<:AbstractFloat }
    @boundscheck begin
        checkbounds(M_new, size(M_old)...)
        checkbounds(total_new, size(total_old)...)
    end
    
    order = size(M_old, 1)

    @inbounds AK.foraxes(M_old, 2) do j  # most allocations here 
        δ = M_new[1, j] - M_old[1, j]
        total_result = total_old[1] + total_new[1]

        for p in order:-1:2
            M_old[p, j] += M_new[p, j]

            # This loop seems to be where the error is coming from. orders 1 and 2 are accurate but 3 is where extreme error starts happening
            # Error also seems to be worst at orders 3, 5, 7, ...
            # At orders 3, 5, 7, ..., the error appears to be more data dependent than the subtle error at even orders
            # Error seems to decrease on average as order rises beyond 3.
            M_tmp = 0
            # for k in p-2:-1:1
            for k in 1:p-2
                k_choose_p = binomial(Int32(p), Int32(k))  # explicit Int32 cast avoids unnecessary use of arbitrary precision arithmetic 
                tmp1 = M_old[p-k, j] * ((-total_new[1]/total_result[1])^k)
                tmp2 = M_new[p-k, j] * ((total_old[1]/total_result[1])^k)
                tmp3 = tmp1 + tmp2
                M_tmp += ((δ^k) * k_choose_p) * tmp3
            end
            M_old[p, j] += M_tmp

            # with batches of size 10000, this section is stable with float64 up to at least order 16 within 5 decimal places
            # tmp = (1/(total_new[1]^(p-1))) - ((-1/total_old[1])^(p-1))  # this is how its shown in the paper
            tmp = ((1/total_new[1])^(p-1)) - ((-1/total_old[1])^(p-1))  # this is not how its shown in the paper, but is how scalib implements it.
            # ^ This improves numerical stability at orders > 4 by avoiding division of 1 by total_new[1]^(p-1), which is quite large at p>4
            tmp *= (((total_old[1] * total_new[1])/total_result[1]) * δ)^p
            M_old[p, j] += tmp
        end

        M_old[1, j] += (δ * (total_new[1]/total_result[1]))  # update mean seperately
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

function get_mean_and_var(m::UniVarMomentsAccVecLabel, d::Int)
    if d == 1
        @inbounds μ = @view m.moments[:, :, 1, :]
        @inbounds σ2 = m.moments[:, :, 2, :] ./ m.totals
        return μ, σ2
    elseif d == 2
        @inbounds μ = m.moments[:, :, 2, :] ./ m.totals
        @inbounds σ2 = m.moments[:, :, 4, :] ./ m.totals
        return μ, σ2
    elseif d > 2
        @inbounds μ = (m.moments[:, :, d, :] ./ m.totals) ./ ((m.moments[:, :, 2, :] ./ m.totals).^(d/2))
        @inbounds σ2 = ((m.moments[:, :, 2*d, :] ./ m.totals) .- ((m.moments[:, :, d, :] ./ m.totals).^2)) ./ ((m.moments[:, :, 2, :] ./ m.totals).^d)
        return μ, σ2
    end
end

end  # module Moments