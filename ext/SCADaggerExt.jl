module SCADaggerExt
using SCA
using Dagger

# get node where each chunk of a resides
(chunk_processors(a::DArray{T, N})::Array{Dagger.Processor, N}) where {T, N} = map(chunk -> fetch(chunk; raw=true).processor, a.chunks)

# map input chunks to dependent output chunks 
a_chunks(ctrd_sums_chunk::NTuple{4, Union{Int, UnitRange, Colon}})::NTuple{2, Union{Int, UnitRange, Colon}} = (Colon(), ctrd_sums_chunk[4])
labels_chunks(ctrd_sums_chunk::NTuple{4, Union{Int, UnitRange, Colon}})::NTuple{2, Union{Int, UnitRange, Colon}} = (Colon(), ctrd_sums_chunk[1])

# TODO: finish when I finalize the strategy in `centered_sum_update`
function Moments.UniVarMomentsAcc{Tt, Tl, DArray}(order, a::DArray, labels::DArray) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    @assert typeof(a) <: AbstractVecOrMat "a expected to be a Vector or Matrix, got $(typeof(a))"
    @assert typeof(labels) <: AbstractVecOrMat "labels expected to be a Vector or Matrix, got $(typeof(labels))"
    
    ns = size(a, 2)
    ldim = size(labels, 2)
    lrange = length(unique(labels))

    a_parts = a.partitioning
    l_parts = labels.partitioning
    @assert a_parts.blocksize[1] == l_parts.blocksize[1] "a and labels must be chunked equally over first dimension"

    # totals = fill!(Ta{UInt32, 2}(undef, ldim, lrange), 0)
    # ctrd_sums = fill!(Ta{Tt, 4}(undef, ldim, lrange, order, ns), 0)
    # sums = fill!(Ta{Tt, 3}(undef, ldim, lrange, ns), 0)

    sums = fill!(Ta{Tt, 3}(undef, ldim, lrange, ns), 0)
    totals = fill!(Ta{UInt32, 2}(undef, ldim, lrange), 0)
    ctrd_sums = zeros(Blocks(l_parts.blocksize[2], lrange, order, a_parts.blocksize[2]), Tt, size(labels, 2), lrange, order, size(a, 2))

    UniVarMomentsAcc{Tt, Tl, DArray}(order, ns, lrange, ldim)
end


function mean_helper(moments::AbstractArray{Tt, 4}, sums::AbstractArray{Tt, 3}, totals::AbstractArray) where{Tt<:AbstractFloat}
    moments[:, :, 1, :] .= sums[:, :, :] ./ totals[:, :]
end

function sum_reduction_helper!(a1::AbstractArray{T1, N}, a2::AbstractArray{T2, N})::AbstractArray{T1, N} where {T1, T2, N}
    a1 .+= a2
end

function cs_sum_reduction_helper!(a1::AbstractArray{T1, N}, a2::AbstractArray{T2, N})::AbstractArray{T1, N} where {T1, T2, N}
    @views a1[:, :, 2:end, :] .+= a2[:, :, 2:end, :]
end

# gives some kind of aliasing error from dagger
function datadeps_bin_tree_reduce(op::Base.Callable, As::Vector{<:Dagger.DArray})
    to_reduce = Vector[]
    push!(to_reduce, As)
    while !isempty(to_reduce)
        As = pop!(to_reduce)
        n = length(As)
        if n == 2
            Dagger.@spawn Base.mapreducedim!(identity, op, InOut(As[1]), In(As[2]))
        elseif n > 2
            push!(to_reduce, [As[1], As[div(n,2)+1]])
            push!(to_reduce, As[1:div(n,2)])
            push!(to_reduce, As[div(n,2)+1:end])
        end
    end
    return As[1]
end

"""
Distributed moment estimation with intermediate value communication.

No constraints on how `a` and `labels` are distributed between nodes, but this causes 
a lot of communication of the intermediate values between nodes and is not very efficient
"""
function Moments.centered_sum_update(a::DMatrix{Tt}, labels::DMatrix{Tl}, lrange::Int, order::Int, ::Val{:naive})::DArray{Tt, 4} where {Tt<:AbstractFloat, Tl<:Integer}
    @boundscheck begin
        checkbounds(labels, size(a, 1), 1)
        checkbounds(a, size(labels, 1), 1)
    end

    # check distribution constraints
    a_parts = a.partitioning
    l_parts = labels.partitioning
    @assert a_parts.blocksize[1] == l_parts.blocksize[1] "traces and labels must be chunked equally over first dimension"

    # allocate intermediate values
    raw_sums = zeros(Blocks(l_parts.blocksize[2], lrange, a_parts.blocksize[2]), Tt, size(labels, 2), lrange, size(a, 2))
    totals = zeros(Blocks(l_parts.blocksize[2], lrange, 1), UInt32, size(labels, 2), lrange, size(a.subdomains, 2))  # add redundant dimension so that increments from nodes with j>1 can be discarded
    ctrd_sums = zeros(Blocks(l_parts.blocksize[2], lrange, order, a_parts.blocksize[2]), Tt, size(labels, 2), lrange, order, size(a, 2))
    
    Dagger.spawn_datadeps() do 
        # distributed pass 1
        for i in axes(labels.chunks, 1)
            for k in axes(labels.chunks, 2)
                for j in axes(a.chunks, 2)
                    Dagger.@spawn Moments.centered_sum_update_pass_1!(InOut(raw_sums.chunks[k, 1, j]), InOut(totals.chunks[k, 1, j]), In(a.chunks[i, j]), In(labels.chunks[i, k]))
                end
            end
        end
        
        # calculate elementwise mean
        for j in axes(a.chunks, 2)
            for k in axes(labels.chunks, 2)
                Dagger.@spawn mean_helper(Out(ctrd_sums.chunks[k, 1, 1, j]), In(raw_sums.chunks[k, 1, j]), In(totals.chunks[k, 1, 1]))  # discard totals where j != 1
            end
        end

        # distributed pass 2
        for i in axes(labels.chunks, 1)
            for k in axes(labels.chunks, 2)
                for j in axes(a.chunks, 2)
                    Dagger.@spawn Moments.centered_sum_update_pass_2!(InOut(ctrd_sums.chunks[k, 1, 1, j]), In(a.chunks[i, j]), In(labels.chunks[i, k]))
                end
            end
        end
    end

    return ctrd_sums
end

"""
Distributed moment estimation with intermediate value reduction between workers.


"""
function Moments.centered_sum_update(a::DMatrix{Tt}, labels::DMatrix{Tl}, lrange::Int, order::Int, ::Val{:reduction})::DArray{Tt, 4} where {Tt<:AbstractFloat, Tl<:Integer}
    # TODO: do tree-reduction between DArray intermedaites instaed of naive reduction
    @boundscheck begin
        checkbounds(labels, size(a, 1), 1)
        checkbounds(a, size(labels, 1), 1)
    end

    # check distribution constraints
    a_parts = a.partitioning; a_procs = chunk_processors(a)
    l_parts = labels.partitioning; l_procs = chunk_processors(labels)
    @assert a_parts.blocksize[1] == l_parts.blocksize[1] "traces and labels must be chunked equally over first dimension"
    @assert size(labels.chunks, 2) == 1 "labels may not be partitioned over dimension 2"
    @assert size(a.chunks, 2) == 1 "a may not be partitioned over dimension 2"
    @assert all(a_procs[n, 1] == l_procs[n, 1] for n in axes(a_procs, 1)) "a and labels chunks on same dim 1 index must reside on same processor"


    # allocate intermediate values for each chunk on i axis
    @debug "allocating intermediates"
    raw_sums = [zeros(Blocks(size(labels, 2), lrange, size(a, 2)), Tt, size(labels, 2), lrange, size(a, 2); assignment=reshape([node, ], 1, 1, 1)) for node in vec(a_procs)]
    totals = [zeros(Blocks(size(labels, 2), lrange), UInt32, size(labels, 2), lrange; assignment=reshape([node, ], 1, 1)) for node in vec(a_procs)]
    ctrd_sums = [zeros(Blocks(size(labels, 2), lrange, order, size(a, 2)), Tt, size(labels, 2), lrange, order, size(a, 2); assignment=reshape([node, ], 1, 1, 1, 1)) for node in vec(a_procs)]
    
    @debug "starting datadeps routine"
    Dagger.spawn_datadeps() do 
        # distributed pass 1
        for i in axes(labels.chunks, 1)
            Dagger.@spawn Moments.centered_sum_update_pass_1!(InOut(raw_sums[i].chunks[1, 1, 1]), InOut(totals[i].chunks[1, 1]), In(a.chunks[i, 1]), In(labels.chunks[i, 1]))
        end
        
        # calculate mean
        for i in 2:size(labels.chunks, 1)
            Dagger.@spawn sum_reduction_helper!(InOut(raw_sums[1].chunks[1, 1, 1]), In(raw_sums[i].chunks[1, 1, 1]))
            Dagger.@spawn sum_reduction_helper!(InOut(totals[1].chunks[1, 1]), In(totals[i].chunks[1, 1]))
        end
        # datadeps_bin_tree_reduce(sum_reduction_helper!, raw_sums)
        # datadeps_bin_tree_reduce(sum_reduction_helper!, totals)
        Dagger.@spawn mean_helper(Out(ctrd_sums[1].chunks[1, 1, 1, 1]), In(raw_sums[1].chunks[1, 1, 1]), In(totals[1].chunks[1, 1]))
        for i in 2:size(labels.chunks, 1)
            Dagger.@spawn copyto!(Out(ctrd_sums[i].chunks[1, 1, 1, 1]), In(ctrd_sums[1].chunks[1, 1, 1, 1]))
        end

        # means are correct, other orders are not...
        # distributed pass 2
        for i in axes(labels.chunks, 1)
            Dagger.@spawn Moments.centered_sum_update_pass_2!(InOut(ctrd_sums[i].chunks[1, 1, 1, 1]), In(a.chunks[i, 1]), In(labels.chunks[i, 1]))
        end

        # reduce ctrd_sums
        for i in 2:size(labels.chunks, 1)
            Dagger.@spawn cs_sum_reduction_helper!(InOut(ctrd_sums[1].chunks[1, 1, 1, 1]), In(ctrd_sums[i].chunks[1, 1, 1, 1]))
        end
    end

    return ctrd_sums[1]
end

end