module SCADaggerExt
using SCA
using Dagger

# get node where each chunk of a resides
(chunk_processors(a::DArray{T, N})::Array{Dagger.Processor, N}) where {T, N} = map(chunk -> fetch(chunk; raw=true).processor, a.chunks)

# map input chunks to dependent output chunks 
a_chunks(ctrd_sums_chunk::NTuple{4, Union{Int, UnitRange, Colon}})::NTuple{2, Union{Int, UnitRange, Colon}} = (Colon(), ctrd_sums_chunk[4])
labels_chunks(ctrd_sums_chunk::NTuple{4, Union{Int, UnitRange, Colon}})::NTuple{2, Union{Int, UnitRange, Colon}} = (Colon(), ctrd_sums_chunk[1])


function Moments.UniVarMomentsAcc{Tt, Tl, DArray}(order, a::DArray, labels::DArray) where {Tt<:AbstractFloat, Tl<:Integer}
    @assert typeof(a) <: AbstractVecOrMat "a expected to be a Vector or Matrix, got $(typeof(a))"
    @assert typeof(labels) <: AbstractVecOrMat "labels expected to be a Vector or Matrix, got $(typeof(labels))"

    a_parts = a.partitioning; a_procs = chunk_processors(a)  # seems to be pretty slow with large distributed arrays
    l_parts = labels.partitioning; l_procs = chunk_processors(labels)
    @assert a_parts.blocksize[1] == l_parts.blocksize[1] "traces and labels must be chunked equally over first dimension"
    @assert size(labels.chunks, 2) == 1 "labels may not be partitioned over dimension 2"
    # @assert size(a.chunks, 2) == 1 "a may not be partitioned over dimension 2"
    @assert all(map(row_procs -> all(==(row_procs[1]), row_procs), eachrow(a_procs)))
    @assert all(a_procs[n, 1] == l_procs[n, 1] for n in axes(a_procs, 1)) "a and labels chunks on same dim 1 index must reside on same processor"

    @debug "getting dataset parameters"
    ns = size(a, 2)
    ldim = size(labels, 2)
    uniques = map(x -> Dagger.@spawn(unique(x)), labels.chunks)
    uniques = fetch.(uniques)
    lrange = length(unique(cat(uniques...; dims=1)))

    @debug "allocating intermediate vals"
    sums = zeros(Blocks(1, size(labels, 2), lrange, a.partitioning.blocksize[2]), Tt, size(a.chunks, 1), size(labels, 2), lrange, size(a, 2); assignment=a_procs[:, :, :, :])  # extra dim so each worker has chunk
    # totals = zeros(Blocks(1, size(labels, 2), lrange), Int32, size(a.chunks, 1), size(labels, 2), lrange; assignment=a_procs[:, :, :])  # extra dim so each worker has chunk
    totals = zeros(Blocks(1, size(labels, 2), lrange, 1), Int32, size(a.chunks, 1), size(labels, 2), lrange, size(a.chunks, 2); assignment=a_procs[:, :, :, :])  # extra dim so each worker has chunk
    ctrd_sums = zeros(Blocks(1, size(labels, 2), lrange, order, a.partitioning.blocksize[2]), Tt, size(a.chunks, 1), size(labels, 2), lrange, order, size(a, 2); assignment=a_procs[:, :, :, :, :])  # extra dim so each worker has chunk

    @debug "initializing struct"
    out = Moments.UniVarMomentsAcc{Tt, Tl, DArray}(totals, ctrd_sums, order, ns, lrange, ldim, sums)
    @debug "finished initializing struct"
    out
end


function Moments.centered_sums(acc::Moments.AbstractUnivariateMomentsAcc{Tt, Tl, DArray}) where {Tt, Tl}
    # assumes chunking scheme with no chunking over sample axis
    return dropdims(cat(fetch.(acc.ctrd_sums.chunks[1, 1, 1, 1, :])..., dims=5), dims=1)
    @debug "fetched centered sums"
end

# function Moments.centered_sums(acc::Moments.AbstractMultivariateMomentsAcc{DArray, Tt, Tl}) where {Tt, Tl}
#     return acc.SCPs
# end

function Moments.totals(acc::Moments.AbstractMomentsAcc{Tt, Tl, DArray}) where {Tt, Tl}
    # assumes chunking scheme with no chunking over sample axis
    @debug "fetched totals"
    return dropdims(fetch(acc.totals.chunks[1, 1, 1, 1]), dims=1)
end



# function mean_helper(moments::AbstractArray{Tt, 4}, sums::AbstractArray{Tt, 3}, totals::AbstractArray) where {Tt<:AbstractFloat}
#     moments[:, :, 1, :] .= sums[:, :, :] ./ totals[:, :]
#     @debug "ran mean helper at $(time())"
# end

function mean_helper(moments::AbstractArray{Tt, 5}, sums::AbstractArray{Tt, 4}, totals::AbstractArray) where {Tt<:AbstractFloat}
    @debug "ran mean helper at $(time())"
    moments[:, :, :, 1, :] .= sums[:, :, :, :] ./ dropdims(view(totals, :, :, :, :), dims=4)
end

function sum_reduction_helper!(a1::AbstractArray{T1, N}, a2::AbstractArray{T2, N})::AbstractArray{T1, N} where {T1, T2, N}
    @debug "ran sum reduction helper at $(time())"
    a1 .+= a2
end

function cs_sum_reduction_helper!(a1::AbstractArray{T1, 4}, a2::AbstractArray{T2, 4})::AbstractArray{T1, 4} where {T1, T2}
    @debug "ran centered sum reduction helper at $(time())"
    @views a1[:, :, 2:end, :] .+= a2[:, :, 2:end, :]
end

function cs_sum_reduction_helper!(a1::AbstractArray{T1, 5}, a2::AbstractArray{T2, 5})::AbstractArray{T1, 5} where {T1, T2}
    @debug "ran centered sum reduction helper at $(time())"
    @views a1[:, :, :, 2:end, :] .+= a2[:, :, :, 2:end, :]
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
These methods are necessary since the reduction-style distributed estimation method adds another dimension to the intermediate
arrays which cannot be removed before calling the appropriate method.
"""

function dagger_centered_sum_update_pass_1!(sums::AbstractArray{Tt, 4}, totals::AbstractArray{Int32, 4}, traces::AbstractVecOrMat{<:Real}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    Moments.centered_sum_update_pass_1!(dropdims(sums, dims=1), dropdims(totals, dims=(1, 4)), traces, labels)
    @debug "ran dagger centered sum update pass 1 at $(time())"
    return
end

function dagger_centered_sum_update_pass_2!(ctrd_sums::AbstractArray{Tt, 5}, traces::AbstractVecOrMat{<:Real}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    Moments.centered_sum_update_pass_2!(dropdims(ctrd_sums, dims=1), traces, labels)
    @debug "ran dagger centered sum update pass 2 at $(time())"
    return
end


function Moments.fit_moments!(acc::Moments.UniVarMomentsAcc{Tt, Tl, DArray}, traces::DArray, labels::DArray) where {Tt, Tl}
    Dagger.spawn_datadeps() do
        # distributed pass 1
        for j in axes(traces.chunks, 2)
            for i in axes(labels.chunks, 1)
                Dagger.@spawn dagger_centered_sum_update_pass_1!(InOut(acc.sums.chunks[i, 1, 1, j]), InOut(acc.totals.chunks[i, 1, 1, j]), In(traces.chunks[i, j]), In(labels.chunks[i, 1]))
            end
        end
        
        # calculate mean
        for i in 2:size(labels.chunks, 1)
            for j in axes(traces.chunks, 2)
                Dagger.@spawn sum_reduction_helper!(InOut(acc.sums.chunks[1, 1, 1, j]), In(acc.sums.chunks[i, 1, 1, j]))
            end
            Dagger.@spawn sum_reduction_helper!(InOut(acc.totals.chunks[1, 1, 1, 1]), In(acc.totals.chunks[i, 1, 1, 1]))
        end

        for j in axes(traces.chunks, 2)
            Dagger.@spawn mean_helper(Out(acc.ctrd_sums.chunks[1, 1, 1, 1, j]), In(acc.sums.chunks[1, 1, 1, j]), In(acc.totals.chunks[1, 1, 1, 1]))
            for i in 2:size(labels.chunks, 1)
                Dagger.@spawn copyto!(Out(acc.ctrd_sums.chunks[i, 1, 1, 1, j]), In(acc.ctrd_sums.chunks[1, 1, 1, 1, j]))
            end
        end

        # distributed pass 2
        for j in axes(traces.chunks, 2)
            for i in axes(labels.chunks, 1)
                Dagger.@spawn dagger_centered_sum_update_pass_2!(InOut(acc.ctrd_sums.chunks[i, 1, 1, 1, j]), In(traces.chunks[i, j]), In(labels.chunks[i, 1]))
            end

            # reduce ctrd_sums
            for i in 2:size(labels.chunks, 1)
                Dagger.@spawn cs_sum_reduction_helper!(InOut(acc.ctrd_sums.chunks[1, 1, 1, 1, j]), In(acc.ctrd_sums.chunks[i, 1, 1, 1, j]))
            end
        end
    end

    @debug "finished fit moments datadeps region at $(time())"
end



"""
Distributed moment estimation with intermediate value communication.

No constraints on how `a` and `labels` are distributed between nodes, but this causes 
a lot of communication of the intermediate values between nodes and is not very efficient
"""
function Moments.get_moments(a::DMatrix{Tt}, labels::DMatrix{Tl}, lrange::Int, order::Int, ::Val{:naive})::Array{Tt, 4} where {Tt<:AbstractFloat, Tl<:Integer}
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
    totals = zeros(Blocks(l_parts.blocksize[2], lrange, 1), Int32, size(labels, 2), lrange, size(a.subdomains, 2))  # add redundant dimension so that increments from nodes with j>1 can be discarded
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

    return collect(ctrd_sums)
end

"""
Distributed moment estimation with intermediate value reduction between workers.

# TODO: document input shape / chunking constraints (see assert statements)
"""
function Moments.get_moments(a::DMatrix{Tt}, labels::DMatrix{Tl}, lrange::Int, order::Int, ::Val{:reduction})::Array{Tt, 4} where {Tt<:AbstractFloat, Tl<:Integer}
    # TODO: do tree-reduction between DArray intermedaites instaed of naive reduction
    @boundscheck begin
        checkbounds(labels, size(a, 1), 1)
        checkbounds(a, size(labels, 1), 1)
    end

    # check distribution constraints
    a_parts = a.partitioning; a_procs = chunk_processors(a)  # should be a (? x 1) matrix of processors
    l_parts = labels.partitioning; l_procs = chunk_processors(labels)  # also (? x 1) matrix of processors
    @assert a_parts.blocksize[1] == l_parts.blocksize[1] "traces and labels must be chunked equally over first dimension"
    @assert size(labels.chunks, 2) == 1 "labels may not be partitioned over dimension 2"
    @assert size(a.chunks, 2) == 1 "a may not be partitioned over dimension 2"
    @assert all(a_procs[n, 1] == l_procs[n, 1] for n in axes(a_procs, 1)) "a and labels chunks on same dim 1 index must reside on same processor"


    # allocate intermediate values for each chunk on i axis
    @debug "allocating intermediates"
    raw_sums = zeros(Blocks(1, size(labels, 2), lrange, size(a, 2)), Tt, size(a.chunks, 1), size(labels, 2), lrange, size(a, 2); assignment=a_procs[:, :, :, :])  # extra dim so each worker has chunk
    totals = zeros(Blocks(1, size(labels, 2), lrange), Int32, size(a.chunks, 1), size(labels, 2), lrange; assignment=assignment=a_procs[:, :, :])  # extra dim so each worker has chunk
    ctrd_sums = zeros(Blocks(1, size(labels, 2), lrange, order, size(a, 2)), Tt, size(a.chunks, 1), size(labels, 2), lrange, order, size(a, 2); assignment=a_procs[:, :, :, :, :])  # extra dim so each worker has chunk
    
    @debug "starting datadeps routine"
    Dagger.spawn_datadeps() do 
        # distributed pass 1
        for i in axes(labels.chunks, 1)
            Dagger.@spawn dagger_centered_sum_update_pass_1!(InOut(raw_sums.chunks[i, 1, 1, 1]), InOut(totals.chunks[i, 1, 1]), In(a.chunks[i, 1]), In(labels.chunks[i, 1]))
        end
        
        # calculate mean
        for i in 2:size(labels.chunks, 1)
            Dagger.@spawn sum_reduction_helper!(InOut(raw_sums.chunks[1, 1, 1, 1]), In(raw_sums.chunks[i, 1, 1, 1]))
            Dagger.@spawn sum_reduction_helper!(InOut(totals.chunks[1, 1, 1]), In(totals.chunks[i, 1, 1]))
        end
        Dagger.@spawn mean_helper(Out(ctrd_sums.chunks[1, 1, 1, 1, 1]), In(raw_sums.chunks[1, 1, 1, 1]), In(totals.chunks[1, 1, 1]))
        for i in 2:size(labels.chunks, 1)
            Dagger.@spawn copyto!(Out(ctrd_sums.chunks[i, 1, 1, 1, 1]), In(ctrd_sums.chunks[1, 1, 1, 1, 1]))
        end

        # distributed pass 2
        for i in axes(labels.chunks, 1)
            Dagger.@spawn dagger_centered_sum_update_pass_2!(InOut(ctrd_sums.chunks[i, 1, 1, 1, 1]), In(a.chunks[i, 1]), In(labels.chunks[i, 1]))
        end

        # reduce ctrd_sums
        for i in 2:size(labels.chunks, 1)
            Dagger.@spawn cs_sum_reduction_helper!(InOut(ctrd_sums.chunks[1, 1, 1, 1, 1]), In(ctrd_sums.chunks[i, 1, 1, 1, 1]))
        end
    end

    return dropdims(fetch(ctrd_sums.chunks[1, 1, 1, 1, 1]), dims=1)
end


end