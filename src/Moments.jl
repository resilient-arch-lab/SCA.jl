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
using KernelAbstractions.Extras.LoopInfo: @unroll
import AcceleratedKernels as AK
using Base: convert
using FixedSizeArrays
using StaticArrays

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

# right now this is slower than running a UniVarMomentsAccs for each label element
struct UniVarMomentsAccNDLabel{Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    totals::Tarray  # Tarray is a typevar, which you can't parameterize directly 
    moments::Tarray
    order::UInt
    ns::UInt
    nl::UInt
    label_shape::NTuple{LD}
    _totals::Tarray
    _moments::Tarray
    _sums::Tarray

    function UniVarMomentsAccNDLabel{Tt, Tl, Tarray, LD}(order, ns, nl, label_shape) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD} 
        totals = fill!(Tarray{UInt32, 1+LD}(undef, label_shape..., nl), 0)
        moments = fill!(Tarray{Tt, 3+LD}(undef, label_shape..., nl, order, ns), 0)
        _totals = similar(totals)
        _moments = similar(moments)
        _sums = Tarray{Tt, 2+LD}(undef, label_shape..., nl, ns)
        new{Tt, Tl, Tarray, LD}(totals, moments, order, ns, nl, label_shape, _totals, _moments, _sums)
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
# Faster on GPU than transposed version
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

function label_wise_sum_ak_transposed!(traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, nl::Int) where {Tt<:AbstractFloat, Tl<:Integer}
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

# Use shared memory reduction
#   label chunk size must kept small enough to not exploade shared memory usage from the shared `moments` tile
#   moments_acc shape: tile_size[3], nl, order, tile_size[2]
#   block shape (threads contiguous on dim x): (tile_size[2], 256 // tile_size[2] = tile_size[1])
#   tile_size[3] must be divisible by tile_size[1] and acc_labels_per_thread = tile_size[3] // tile_size[1]
#       This only works if tile_size[1] < tile_size[3], which is not true for reasonably small tile_size[2]
#   alternative: nl must be divisible by tile_size[1], and acc_labels_per_thread = nl // tile_size[1]
#       error on thread (65, 1), block (127, 1)
@kernel function centered_sum_kern_KA_2!(
    moments::AbstractArray{Tt, 4}, @Const(traces::AbstractMatrix{Tt}), @Const(labels::AbstractMatrix{Tl}),
    ::Val{tile_size}, ::Val{tiler_size}, ::Val{order}, ::Val{lsize}, ::Val{nl}, ::Val{acc_labels_per_thread}) where {Tt<:AbstractFloat, Tl<:Integer, tile_size, tiler_size, order, lsize, nl, acc_labels_per_thread}

    j, i = @index(Local, NTuple)
    J, I = @index(Group, NTuple) # block indexes in grid, accounting for tile_size and tiler
    # @print("(j, i): $((j, i))\t(J, I): $((J, I))\n")  # griddim is right

    moments_acc = @localmem Tt (tile_size[3], nl, order, tile_size[2])
    
    for i_tile in 1:tiler_size[1]
        for j_tile in 1:tiler_size[2]
            i_tile_global_offset =  i + ((i_tile-1)*tile_size[1]) + ((I-1) * tile_size[1]*tiler_size[1])
            j_tile_global_offset =  j + ((j_tile-1)*tile_size[2]) + ((J-1) * tile_size[2]*tiler_size[2])
            t = traces[i_tile_global_offset, j_tile_global_offset]
            for l_tile in 1:tiler_size[3]
                if (@index(Local, Linear) == 29) & (@index(Group, Linear) == 16)
                    @print("i_tile: $(i_tile)\t j_tile: $(j_tile)\t l_tile: $(l_tile)\t(j, i): $((j, i))\t(J, I): $((J, I))\n")
                end
                @synchronize()
                # zero moments_acc
                for l in 1:acc_labels_per_thread
                    for lidx in 1:tile_size[3]
                        moments_acc[lidx, (l-1)+i, 1, j] = moments[(l_tile-1)*tile_size[3] + lidx, (l-1)+i, 1, j_tile_global_offset]
                        for d in 2:order
                            moments_acc[lidx, (l-1)+i, d, j] = 0
                        end
                    end
                end
                @synchronize()

                # accumulate to moments_acc
                for lidx in 1:tile_size[3]
                    l = convert(Int32, (labels[i_tile_global_offset, (l_tile-1)*tile_size[3] + lidx]+1)&0xff)
                    if l <= size(moments_acc, 2)
                        t_update = t - moments_acc[lidx, l, 1, j]
                        t_power = t_update
                        for d in 2:order
                            t_power *= t_update
                            Atomix.@atomic moments_acc[lidx, l, d, j] += t_power
                        end
                    end
                end
                @synchronize()

                # atomic_add to global mem
                for lidx in 1:tile_size[3]
                    for l in 1:acc_labels_per_thread
                        for d in 2:order
                            Atomix.@atomic moments[(l_tile-1)*tile_size[3] + lidx, (l-1)+i, d, j_tile_global_offset] += moments_acc[lidx, (l-1)+i, d, j]
                        end
                    end
                end
            end
        end
    end
end

function centered_sum_KA_wrapper!(
    moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}, 
    ::Val{tile_size}, ::Val{tiler_size}, ::Val{order}, ::Val{lsize}, ::Val{nl}, ::Val{acc_labels_per_thread}) where {Tt<:AbstractFloat, Tl<:Integer, tile_size, tiler_size, order, lsize, nl, acc_labels_per_thread}
    
    @assert order == size(moments, 3) "order must equal the size of `moments` in third dim"
    @assert lsize == size(labels, 2) "lsize must equal size of `labels` in second dim"
    # @assert size(traces) == tile_size .* tiler_size

    kernel_ndrange = (size(traces, 2) ÷ (tiler_size[2]), size(traces, 1) ÷ (tiler_size[1]))
    block_shape = (tile_size[2], tile_size[1])
    traces_tiles = tiled_view(traces, (tile_size[1], tile_size[2]))
    # trace_tile_sizes = unique(size.([traces_tiles[1, 1] traces_tiles[1, end]; traces_tiles[end, 1] traces_tiles[end, end]]))
    labels_tiles = tiled_view(labels, (tile_size[1], tile_size[3]))

    println("tile size: $(tile_size)")
    println("tiler size: $(tiler_size)")
    println("size product: $(tile_size .* tiler_size)")
    println("acc_labels_per_thread: $(acc_labels_per_thread)")
    println("Kernel ndrange: $(kernel_ndrange)")
    println("Kernel block shape: $(block_shape)")

    dev = get_backend(moments)
    kernel = centered_sum_kern_KA_2!(dev, block_shape, kernel_ndrange)
    kernel(moments, traces, labels, Val(tile_size), Val(tiler_size), Val(order), Val(lsize), Val(nl), Val(acc_labels_per_thread), ndrange=kernel_ndrange)
    # KernelAbstractions.synchronize(dev)
end

# non-atomic shared memory reduction
# n x m block:
#   1. fetches tile of traces[n*samples_per_thread, m*traces_per_thread] to shared memory
#       thd_trace_tile_size = [samples_per_thread, traces_per_thread]
#   2. accumulate to moments_acc
#       
#   3. accumulate to global mem
#   
# block_shape: [traces_per_block]
@kernel function centered_sum_kern_KA_3!(moments::AbstractArray{Tt, 4}, @Const(traces::AbstractMatrix{Tt}), @Const(labels::AbstractMatrix{Tl}),
    ::Val{thd_trace_tile_size}, ::Val{thd_label_tile_size}, ::Val{order}, ::Val{lsize}, ::Val{nl}) where {Tt<:AbstractFloat, Tl<:Integer, thd_trace_tile_size, thd_label_tile_size, order, lsize, nl}

    @assert thd_trace_tile_size[1] == thd_trace_tile_size[1] "trace and label tile sizes must match on index 1"

    trace_shmem = @localmem Tt (thd_trace_tile_size[1] * @groupsize()[1], thd_trace_tile_size[2] * @groupsize()[2])
    label_shmem = @localmem Tl (thd_label_tile_size[1] * @groupsize()[1], thd_label_tile_size[2] * @groupsize()[2])
    moments_acc = @localmem Tt (@groupsize()[2], nl, order, @groupsize()[1])

    j, i = @index(Local, NTuple)
    J, I = @index(Group, NTuple)

    trace_global_offset = (I * @groupsize()[2] * thd_trace_tile_size[2], J * @groupsize()[1] * thd_trace_tile_size[1])
    label_global_offset = (I * @groupsize()[2] * thd_label_tile_size[2], J * @groupsize()[1] * thd_label_tile_size[1])
    
    trace_shmem[]

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

    traces_per_thread = 10
    trace_tiles = tiled_view(traces, (traces_per_thread, size(traces, 2)))
    ntiles = size(trace_tiles, 1)
    label_tiles = tiled_view(labels, (traces_per_thread, size(labels, 2)))

    @inbounds AK.foreachindex(itr_view) do idx
        (l, j) = CartesianIndices(itr_view)[idx].I
        for ti in 1:ntiles
            for i in 1:traces_per_thread
                t_i = traces[((ti-1)*traces_per_thread)+i, j]
                l_i = convert(Int32, labels[((ti-1)*traces_per_thread)+i, l]+1)
                t_update = t_i - moments[l, l_i, 1, j]
                pow = t_update
                for d in 2:order
                    pow *= t_update
                    moments[l, l_i, d, j] += pow
                end
            end
        end
    end
end

function centered_sum_kern_ak_atomic!(moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(moments, 3)

    @inbounds AK.foreachindex(traces') do idx
        (j, i) = CartesianIndices((size(traces, 2), size(traces, 1)))[idx].I
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

    label_wise_sum_ak_transposed!(traces, labels, acc._sums, acc._totals)

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
    return nothing
end

# This is still horrendously slow on CPU, must fix. 
# No overhead compared to scalar labels on GPUs though
function centered_sum_update!(acc::UniVarMomentsAccNDLabel{Tt, Tl, Tarray, LD}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    # Initialize intermediate values
    fill!(acc._sums, 0)
    fill!(acc._moments, 0)
    fill!(acc._totals, 0)

    if get_backend(traces) != get_backend(acc._sums) || get_backend(labels) != get_backend(acc._sums)
        print("backend mismatch")
        traces = Tarray(traces)
        labels = Tarray(labels)
    end

    @time "label_wise_sum_ak!" label_wise_sum_ak_transposed!(traces, labels, acc._sums, acc._totals)

    # find means
    @time "means" @. acc._moments[:, :, 1, :] = acc._sums / acc._totals

    # compute centered sums
    @time "centered_sum_kern_ak!" centered_sum_kern_ak!(acc._moments, traces, labels)  # 15s
    # @time "centered_sum_kern_ak!" for l in axes(labels, 2)
    # centered_sum_kern_ak_transposed!(view(acc._moments, l, :, :, :), traces, view(labels, :, l))
    # end

    # merge centered sum estimations
    @time "merge" begin
        init_ls = acc.totals .== 0
        update_ls = acc.totals .!= 0
        if any(init_ls)
            @inbounds acc.moments[init_ls, :, :] .= acc._moments[init_ls, :, :]
            # @inbounds acc.totals[init_ls] .= acc._totals[init_ls]
        end
        if any(update_ls)
            for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
                merge_from_ak!(view(acc.moments, l, :, :), view(acc.totals, l), view(acc._moments, l, :, :), view(acc._totals, l))
            end
            # @inbounds acc.totals[update_ls] .+= acc._totals[update_ls]
        end
    end
end

# There's the potential for shards to be larger than the slice they're correlated with

# First pass in two pass approach
function centered_sum_update_pass_1!(acc::UniVarMomentsAccVecLabel{Tt, Tl, Tarray, LD}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    @boundscheck begin
        checkbounds(acc._sums, LD, acc.nl, size(traces, 2))
        checkbounds(acc._moments, LD, acc.nl, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), LD)
    end

    label_wise_sum_ak_transposed!(traces, labels, acc._sums, acc._totals)

    return
end

# Second pass in two pass approach
function centered_sum_update_pass_2!(acc::UniVarMomentsAccVecLabel{Tt, Tl, Tarray, LD}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    @boundscheck begin
        checkbounds(acc._sums, LD, acc.nl, size(traces, 2))
        checkbounds(acc._moments, LD, acc.nl, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), LD)
    end

    @. acc._moments[:, :, 1, :] = acc._sums / acc._totals

    centered_sum_kern_ak!(acc._moments, traces, labels)

    acc.moments .= acc._moments
    acc.totals .= acc._totals

    return
end

function centered_sum_update!(acc::UniVarMomentsAccVecLabel{Tt, Tl, Tarray, LD}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray, LD}
    centered_sum_update_pass_1!(acc, traces, labels)
    centered_sum_update_pass_2!(acc, traces, labels)
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
    @boundscheck begin
        checkbounds(M_new, size(M_old)...)
        checkbounds(total_new, size(total_old)...)
    end
    
    order = size(M_old, 1)
    @inbounds AK.foraxes(M_old, 2) do i
        δ = M_new[1, i] - M_old[1, i]
        total_result = total_old[1] + total_new[1]

        for p in order:-1:2
            (as_input1, to_update1) = view(M_old, 1:p-1, :), view(M_old, p, :)
            (as_input2, to_update2) = view(M_new, 1:p-1, :), view(M_new, p, :)

            to_update1[i] += to_update2[i] 

            for k in 1:p-2
                cst = binomial(Int32(k), Int32(p))  # explicity Int32 cast avoids unnecessary use of arbitrary precision arithmetic 
                tmp1 = as_input1[p-k, i] * ((-total_new[1]/total_result[1])^k)
                tmp2 = as_input2[p-k, i] * ((total_old[1]/total_result[1])^k)
                tmp3 = tmp1 + tmp2
                # to_update1[i] += (δ_pows[k, i] * cst) * tmp3
                to_update1[i] += (δ^k * cst) * tmp3
            end
            tmp = (1/(total_new[1]^(p-1))) - ((-1/total_old[1])^(p-1))  # about 20% of runtime
            tmp *= ((total_old[1] * total_new[1])/total_result[1])^p  # another 20% of the runtime, mostly the exponent (so thats fine)

            to_update1[i] += δ^p * tmp
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