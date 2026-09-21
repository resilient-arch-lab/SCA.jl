module SCADaggerExt
using SCA
using Dagger

# get node where each chunk of a resides
(chunk_nodes(a::DArray{T, N})::Array{Int, N}) where {T, N} = map(chunk -> chunk.thunk_ref.owner, a.chunks)

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

"""
Distributed moment estimation
accepts 
"""
function Moments.centered_sum_update(a::DMatrix{Tt}, labels::DMatrix{Tl}, lrange::Int, order::Int)::DArray{Tt, 4} where {Tt<:AbstractFloat, Tl<:Integer}
    @boundscheck begin
        checkbounds(labels, size(a, 1), 1)
        checkbounds(a, size(labels, 1), 1)
    end

    # check distribution constraints
    a_parts = a.partitioning
    l_parts = labels.partitioning
    @assert a_parts.blocksize[1] == l_parts.blocksize[1] "traces and labels must be chunked equally over first dimension"

    # allocate intermediate values
    # TODO: make this so that there's an intermediate value for each input chunk which are then reduced 
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

end