module SCADaggerExt
using SCA
using Dagger


function mean_helper(moments::AbstractArray{Tt, 4}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix) where{Tt<:AbstractFloat}
    moments[:, :, 1, :] .= sums[:, :, :] ./ totals[:, :]
end

# distributed moment estimation
# NOTE: right now, this function yields incorrect results when there is more than 1 chunk in the second trace dimension.
function Moments.centered_sum_update(traces::DMatrix{Tt}, labels::DMatrix{Tl}, nl::Int, order::Int)::DArray{Tt, 4} where {Tt<:AbstractFloat, Tl<:Integer}
    @boundscheck begin
        checkbounds(labels, size(traces, 1), 1)
        checkbounds(traces, size(labels, 1), 1)
    end

    # check distribution constraints
    t_parts = traces.partitioning
    l_parts = labels.partitioning
    @assert t_parts.blocksize[1] == l_parts.blocksize[1] "traces and labels must be blocked equally over first dimension"

    # allocate intermediate values
    sums = zeros(Blocks(l_parts.blocksize[2], nl, t_parts.blocksize[2]), Tt, size(labels, 2), nl, size(traces, 2))
    totals = zeros(Blocks(l_parts.blocksize[2], nl), UInt32, size(labels, 2), nl)
    moments = zeros(Blocks(l_parts.blocksize[2], nl, order, t_parts.blocksize[2]), Tt, size(labels, 2), nl, order, size(traces, 2))

    Dagger.spawn_datadeps() do 
        # distributed pass 1
        for i in axes(labels.chunks, 1)
            for k in axes(labels.chunks, 2)
                for j in axes(traces.chunks, 2)
                    # TODO: totals can only be incremented if j=1. 
                    Dagger.@spawn Moments.centered_sum_update_pass_1!(InOut(sums.chunks[k, 1, j]), InOut(totals.chunks[k, 1]), In(traces.chunks[i, j]), In(labels.chunks[i, k]))
                    # Dagger.@spawn label_wise_sum_ak_transposed!(In(traces.chunks[i, j]), In(labels.chunks[i, k]), InOut(sums.chunks[k, 1, j]), InOut(totals.chunks[k, 1]))
                end
            end 
        end
        
        # calculate elementwise mean
        for j in axes(traces.chunks, 2)
            for k in axes(labels.chunks, 2)
                Dagger.@spawn mean_helper(Out(moments.chunks[k, 1, 1, j]), In(sums.chunks[k, 1, j]), In(totals.chunks[k, 1]))
            end
        end

        # distributed pass 2
        for i in axes(labels.chunks, 1)
            for k in axes(labels.chunks, 2)
                for j in axes(traces.chunks, 2)
                    Dagger.@spawn Moments.centered_sum_update_pass_2!(InOut(moments.chunks[k, 1, 1, j]), In(traces.chunks[i, j]), In(labels.chunks[i, k]))
                end
            end
        end
    end

    return moments
end



end