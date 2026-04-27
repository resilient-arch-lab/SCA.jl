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
using Dagger

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

struct UniVarMomentsAccDagger{Tt<:AbstractFloat, Tl<:Integer}
    workers::Array{Int, 1}
    totals::Array
    moments::Array
    order::UInt
    ns::UInt
    nl::UInt
    chunksize::NTuple{2, Int}
    _totals::Array
    _moments::Array
    _sums::Array
    _totals_shard::Dagger.Shard
    _moments_shard::Dagger.Shard
    _sums_shard::Dagger.Shard
    worker_mapping::Dict{Int, Int}


    function UniVarMomentsAccDagger{Tt, Tl}(workers, order, ns, nl, chunksize) where {Tt<:AbstractFloat, Tl<:Integer}
        nworkers = size(workers, 1)
        totals = zeros(UInt32, nl)
        moments = zeros(Tt, nl, order, ns)

        _totals = similar(totals)
        _moments = similar(moments)
        _sums = zeros(Tt, nl, ns)
        _totals_shard = Dagger.@shard workers=workers similar(totals)
        _moments_shard = Dagger.@shard workers=workers similar(moments, size(moments)[1:2]..., chunksize[2])
        _sums_shard = Dagger.@shard workers=workers zeros(Tt, nl, chunksize[2])

        worker_mapping = Dict(s => w for (s, w) ∈ zip(1:nworkers, workers))

        new(workers, totals, moments, order, ns, nl, chunksize, _totals, _moments, _sums, _totals_shard, _moments_shard, _sums_shard, worker_mapping)
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

function label_wise_sum_ak!(traces::AbstractMatrix{Tt}, labels::AbstractMatrix{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
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
    to_update = @view moments[:, :, 2:end, :]

    @inbounds AK.foraxes(traces, 2) do j
        for i in axes(traces, 1)
            t_i = traces[i, j]
            for l in axes(labels, 2)
                l_i = convert(Int32, labels[i, l]+1)
                t_update = t_i - moments[l, l_i, 1, j]
                pow = t_update
                for d in axes(to_update, 3)
                    pow *= t_update
                    to_update[l, l_i, d, j] += pow
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

    @time "label_wise_sum_ak!" label_wise_sum_ak!(traces, labels, acc._sums, acc._totals)

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
function centered_sum_update!(acc::UniVarMomentsAccDagger{Tt, Tl}, traces::DArray{Tt, 2}, labels::DArray{Tl, 1}) where {Tt<:AbstractFloat, Tl<:Integer}
    # @boundscheck begin
    #     checkbounds(acc._sums, acc.nl, size(traces, 2))
    #     checkbounds(acc._moments, acc.nl, acc.order, size(traces, 2))
    #     checkbounds(labels, size(traces, 1))
    # end
    
    workers_scope = Dagger.scope(workers=acc.workers, thread=1)  # I should be using theread=1 to not mess up the internal multithreading on each worker
    lead_scope = Dagger.scope(worker=acc.workers[1], thread=1)
    lead_proc = OSProc(acc.workers[1])
    
    Dagger.spawn_datadeps() do
        for w ∈ acc.workers
            Dagger.@spawn scope=Dagger.scope(worker=w) fill!(InOut(acc._sums_shard), 0)
            # Dagger.@spawn scope=Dagger.scope(worker=w) fill!(InOut(acc._moments_shard), 0)
            Dagger.@spawn scope=Dagger.scope(worker=w) fill!(InOut(acc._totals_shard), 0)
            fill!(acc._moments, 0)
        end
        println("prepared shards")
        
        for slice ∈ axes(traces.chunks, 2)  # I don't think this parallelizes such that multiple workers can process the same slice
            slice_idxs = Dagger.indexes(traces.subdomains[1, slice])[2]
            for batch ∈ axes(traces.chunks, 1)
                Dagger.@spawn scope=workers_scope label_wise_sum_ak_transposed!(In(traces.chunks[slice, batch]), In(labels.chunks[batch]), InOut(acc._sums_shard), InOut(acc._totals_shard))
            end
            Dagger.@spawn scope=Dagger.scope(worker=acc.worker_mapping[slice]) copyto!(InOut(view(acc._sums, :, slice_idxs)), In(acc._sums_shard))
        end
        Dagger.@spawn scope=Dagger.scope(worker=acc.worker_mapping[1]) copyto!(InOut(acc._totals), In(acc._totals_shard))
        println("finished 1st pass")

        @. acc._moments[:, 1, :] = acc._sums / acc._totals
        for slice ∈ axes(traces.chunks, 2)
            slice_idxs = Dagger.indexes(traces.subdomains[1, slice])[2]
            Dagger.@spawn scope=Dagger.scope(worker=acc.worker_mapping[slice]) copyto!(Out(acc._moments_shard), In(acc._moments[:, :, slice_idxs]))
        end
        println("calculated mean and distributed result")

        for slice ∈ axes(traces.chunks, 2)
            slice_idxs = Dagger.indexes(traces.subdomains[1, slice])[2]
            for batch ∈ axes(traces.chunks, 1)
                Dagger.@spawn scope=workers_scope centered_sum_kern_ak_transposed!(InOut(acc._moments_shard), In(traces.chunks[slice, batch]), In(labels.chunks[batch]))
            end
            _moments_to_update = view(acc._moments, :, 2:size(acc._moments, 2), slice_idxs)
            _moments_update = Dagger.@spawn scope=Dagger.scope(worker=acc.worker_mapping[slice]) view(acc._moments_shard)  # this needs an In(), but that makes it error
            Dagger.@spawn scope=Dagger.scope(worker=acc.worker_mapping[slice]) copyto!(InOut(_moments_to_update), In(_moments_update))
        end
        println("finished 2nd pass")
        
        init_ls = acc.totals .== 0
        update_ls = acc.totals .!= 0
        if any(init_ls)
            @inbounds acc.moments[init_ls, :, :] .= acc._moments[init_ls, :, :]
            @inbounds acc.totals[init_ls] .= acc._totals[init_ls]
        end
        if any(update_ls)
            throw(ErrorException("Not supposed to happen!"))
        end
        println("finished merge")
    end
    return nothing
end

# Assume that the dataset has been split over the time axis between nodes already. Any distributed
# parallelism added in this function should consider that.
# State:
#   Currently its accurate. However, every time this is called theres a lot of overhead. It seems to mostly
#   be coming from allocation of distributed arrays. Maybe I could create a new UniVarMomentsAcc struct that
#   holds a `workers` array and pre-allocated distributed arrays.
#   Using shards (correctly this time) actually reduced overhead a lot
function centered_sum_update_dagger!(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}, ctx::Context, workers::Array{Int, 1}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    workers_scope = Dagger.scope(workers=workers, thread=1)
    
    @boundscheck begin
        checkbounds(acc._sums, acc.nl, size(traces, 2))
        checkbounds(acc._moments, acc.nl, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1))
    end

    # For each Proccessor (worker) in ctx.procs, return the set of ThreadProc / GPUDeviceProc it contains
    # sub_processors = Dagger.get_processors.(ctx.procs)

    _sums = Dagger.@shard workers=workers fill!(similar(acc._sums), 0)
    _totals = Dagger.@shard workers=workers fill!(similar(acc._totals), 0)
    
    # _sums = zeros(Blocks(1, size(acc._sums)...), eltype(acc._sums), size(workers, 1), size(acc._sums)...)
    # _totals = zeros(Blocks(1, size(acc._totals)...), eltype(acc._totals), size(workers, 1), size(acc._totals)...)
    # _moments = zeros(Blocks(1, size(acc._moments)...), eltype(acc._moments), size(workers, 1), size(acc._moments)...)
    fill!(acc._moments, 0)

    Dtraces = distribute(traces, Blocks(Int(ceil(size(traces, 1) / size(workers, 1))), size(traces, 2)), reshape(workers, size(workers, 1), 1))
    Dlabels = distribute(labels, Blocks(Int(ceil(size(labels, 1) / size(workers, 1)))), workers)

    # first pass
    @sync for c in axes(workers, 1)
        # Dagger.@spawn scope=workers_scope label_wise_sum_ak_transposed!(Dtraces.chunks[c], Dlabels.chunks[c], Dagger.@spawn(dropdims(_sums.chunks[c], dims=1)), Dagger.@spawn(dropdims(_totals.chunks[c], dims=1)))
        Dagger.@spawn scope=workers_scope label_wise_sum_ak_transposed!(Dtraces.chunks[c], Dlabels.chunks[c], _sums, _totals)
    end
    acc._sums .= .+(map(shard->fetch(Dagger.@spawn copy(shard)), _sums)...)
    acc._totals .= .+(map(shard->fetch(Dagger.@spawn copy(shard)), _totals)...)
    # acc._sums .= dropdims(.+(fetch.(_sums.chunks)...), dims=1)  # same for each run
    # acc._totals .= dropdims(.+(fetch.(_totals.chunks)...), dims=1)  # same for each run
    println("finished 1st pass")

    # find means
    @. acc._moments[:, 1, :] = acc._sums / acc._totals
    _moments = Dagger.@shard workers=workers copy(acc._moments)
    # @sync for c in axes(workers, 1)        
    #     # Sometimes throws a concurrency violation due to concurrent resizing of vector?
    #     @assert size(_moments[c, :, 1, :]) == size(view(acc._moments, :, 1, :))
    #     Dagger.@spawn _moments[c, :, 1, :] = view(acc._moments, :, 1, :)
    # end
    println("finished copying _moments")

    # second pass
    @sync for c in axes(workers, 1)
        # Dagger.@spawn scope=workers_scope centered_sum_kern_ak_transposed!(Dagger.@spawn(dropdims(_moments.chunks[c], dims=1)), Dtraces.chunks[c], Dlabels.chunks[c])
        Dagger.@spawn scope=workers_scope centered_sum_kern_ak_transposed!(_moments, Dtraces.chunks[c], Dlabels.chunks[c])
    end
    acc._moments[:, 2:end, :] .= .+(map(shard->fetch(Dagger.@spawn copy(shard)), _moments)...)[:, 2:end, :]  # not the same each run
    # acc._moments[:, 2:end, :] .= dropdims(.+(fetch.(_moments.chunks)...), dims=1)[:, 2:end, :]
    println("finished 2nd pass")

    # merge centered sum estimations
    init_ls = acc.totals .== 0
    update_ls = acc.totals .!= 0
    if any(init_ls)
        @inbounds acc.moments[init_ls, :, :] .= acc._moments[init_ls, :, :]
        @inbounds acc.totals[init_ls] .= acc._totals[init_ls]
    end
    if any(update_ls)
        throw(ErrorException("Not supposed to happen!"))
    end
    println("finished merge")
    return nothing
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