"""
Parallel estimation of statistical moments. based on the implementation from
[SCALib](https://github.com/simple-crypto/SCALib).
"""

"""
TODO:
- Get rid of non-VecLabel acc and make the VecLabel methods handle label vectors (just rehsape them as n X 1 matrices)
- Make a single function `fit!` to apply to incremental and non incremental MomentsAccs
- Make functions `raw_moments`, `central_moments`, and `standardized_moments` for `AbstractMomentsAcc`
- Move multivariate stuff here from testing branch
"""


module Moments
export centered_sum_update!, merge_from!, get_mean_and_var, UniVarMomentsAccIncremental, centered_sum_update_pass_1!, centered_sum_update_pass_2!

include("Utils.jl")
using .Utils

using Random
using KernelAbstractions, Atomix
import AcceleratedKernels as AK
using Base: convert

abstract type AbstractMomentsAcc end
abstract type AbstractUnivariateMomentsAcc <: AbstractMomentsAcc end

struct UniVarMomentsAccIncremental{Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray} <: AbstractUnivariateMomentsAcc
    totals::Tarray
    ctrd_sums::Tarray
    order::UInt
    ns::UInt
    lrange::UInt
    ldim::UInt
    _totals::Tarray
    _ctrd_sums::Tarray
    _sums::Tarray
end

function UniVarMomentsAccIncremental{Tt, Tl, Tarray}(order, ns, lrange, ldim) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray} 
    totals = fill!(Tarray{UInt32, 2}(undef, ldim, lrange), 0)
    ctrd_sums = fill!(Tarray{Tt, 4}(undef, ldim, lrange, order, ns), 0)
    _totals = fill!(similar(totals), 0)
    _ctrd_sums = fill!(similar(ctrd_sums), 0)
    _sums = fill!(Tarray{Tt, 3}(undef, ldim, lrange, ns), 0)
    UniVarMomentsAccIncremental{Tt, Tl, Tarray}(totals, ctrd_sums, order, ns, lrange, ldim, _totals, _ctrd_sums, _sums)
end

# initialize from dataset shape and labels
function UniVarMomentsAccIncremental{Tt, Tl, Tarray}(order, a::Tarray, labels::Tarray) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    ns = size(a, 2)
    ldim = size(l, 2)
    lrange = length(unique(labels))

    totals = fill!(Tarray{UInt32, 2}(undef, ldim, lrange), 0)
    ctrd_sums = fill!(Tarray{Tt, 4}(undef, ldim, lrange, order, ns), 0)
    _totals = fill!(similar(totals), 0)
    _ctrd_sums = fill!(similar(ctrd_sums), 0)
    _sums = fill!(Tarray{Tt, 3}(undef, ldim, lrange, ns), 0)
    UniVarMomentsAccIncremental{Tt, Tl, Tarray}(totals, ctrd_sums, order, ns, lrange, ldim, _totals, _ctrd_sums, _sums)
end

function label_wise_sum_ak!(traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}, sums::AbstractMatrix{Tt}, totals::AbstractVector{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
    @inbounds AK.foraxes(traces, 1) do i
        l_i = convert(Int32, labels[i]+1)
        Atomix.@atomic totals[l_i] += 1
        for j in axes(traces, 2)
            Atomix.@atomic sums[l_i, j] += traces[i, j]
        end
    end
end

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

function label_wise_sum_ak_transposed!(traces::AbstractMatrix{Tt}, labels::AbstractVecOrMat{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
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
function label_wise_sum_ak!(traces::AbstractMatrix{Tt}, labels::AbstractVecOrMat{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
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
@inline function label_wise_sum!(traces::AbstractMatrix{Tt}, labels::AbstractVecOrMat{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractMatrix) where {Tt<:AbstractFloat, Tl<:Integer}
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

function centered_sum_kern_ak!(moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
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

function centered_sum_kern_ak_atomic!(moments::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
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
@inline function centered_sum!(ctrd_sums::AbstractArray{Tt, 4}, traces::AbstractMatrix{Tt}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    order = size(ctrd_sums, 3)
    
    for i in axes(traces, 1)
        for l in axes(labels, 2)
            l_i = convert(Int, labels[i, l])+1
            for j in axes(traces, 2)
                t_update = traces[i, j] - ctrd_sums[l, l_i, 1, j]
                pow = t_update
                for d in 2:order
                    pow *= t_update
                    ctrd_sums[l, l_i, d, j] += pow
                end
            end
        end
    end
end

# First pass in two pass approach
function centered_sum_update_pass_1!(acc::UniVarMomentsAccIncremental{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    @boundscheck begin
        checkbounds(acc._sums, acc.ldim, acc.lrange, size(traces, 2))
        checkbounds(acc._ctrd_sums, acc.ldim, acc.lrange, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), acc.ldim)
    end
    
    fill!(acc._ctrd_sums, 0)
    fill!(acc._sums, 0)
    fill!(acc._totals, 0)

    label_wise_sum_ak_transposed!(traces, labels, acc._sums, acc._totals)

    return
end

function centered_sum_update_pass_1!(sums::AbstractArray{Tt}, totals::AbstractArray{UInt32}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    label_wise_sum_ak_transposed!(traces, labels, sums, totals)
    return
end

# Second pass in two pass approach
function centered_sum_update_pass_2!(acc::UniVarMomentsAccIncremental{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    @boundscheck begin
        checkbounds(acc._sums, acc.ldim, acc.lrange, size(traces, 2))
        checkbounds(acc._ctrd_sums, acc.ldim, acc.lrange, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), acc.ldim)
    end

    @. acc._ctrd_sums[:, :, 1, :] = acc._sums / acc._totals

    centered_sum_kern_ak!(acc._ctrd_sums, traces, labels)

    # merge centered sum estimations
    init_ls = acc.totals .== 0
    update_ls = acc.totals .!= 0
    if any(init_ls)
        @inbounds @views acc.ctrd_sums[init_ls, :, :] .= acc._ctrd_sums[init_ls, :, :]
        @inbounds @views acc.totals[init_ls] .= acc._totals[init_ls]
    end
    if any(update_ls)
        for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
            @inbounds merge_from_ak!(view(acc.ctrd_sums, l, :, :), view(acc.totals, l), view(acc._ctrd_sums, l, :, :), view(acc._totals, l))
        end
        @inbounds @views acc.totals[update_ls] .+= acc._totals[update_ls]
    end

    return
end

function centered_sum_update_pass_2!(ctrd_sums::AbstractArray{Tt}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    centered_sum_kern_ak!(ctrd_sums, traces, labels)
    return
end

function centered_sum_update!(acc::UniVarMomentsAccIncremental{Tt, Tl, Tarray}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
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

# TODO: This seems to be consistently innaccurate, not due to floating point precision issues. I should
# figure out why that is.
function merge_from_ak!(CS_old::AbstractArray{Tt, 2}, total_old::AbstractArray{UInt32, 0}, CS_new::AbstractArray{Tt, 2}, total_new::AbstractArray{UInt32, 0}) where { Tt<:AbstractFloat }
    @boundscheck begin
        checkbounds(CS_new, size(CS_old)...)
        checkbounds(total_new, size(total_old)...)
    end
    
    order = size(CS_old, 1)

    @inbounds AK.foraxes(CS_old, 2) do j  # most allocations here 
        δ = CS_new[1, j] - CS_old[1, j]
        total_result = total_old[1] + total_new[1]

        for p in order:-1:2
            CS_old[p, j] += CS_new[p, j]

            # This loop seems to be where the error is coming from. orders 1 and 2 are accurate but 3 is where extreme error starts happening
            # Error also seems to be worst at orders 3, 5, 7, ...
            # At orders 3, 5, 7, ..., the error appears to be more data dependent than the subtle error at even orders
            # Error seems to decrease on average as order rises beyond 3.
            M_tmp = 0
            # for k in p-2:-1:1
            for k in 1:p-2
                k_choose_p = binomial(Int32(p), Int32(k))  # explicit Int32 cast avoids unnecessary use of arbitrary precision arithmetic 
                tmp1 = CS_old[p-k, j] * ((-total_new[1]/total_result[1])^k)
                tmp2 = CS_new[p-k, j] * ((total_old[1]/total_result[1])^k)
                tmp3 = tmp1 + tmp2
                M_tmp += ((δ^k) * k_choose_p) * tmp3
            end
            CS_old[p, j] += M_tmp

            # with batches of size 10000, this section is stable with float64 up to at least order 16 within 5 decimal places
            tmp = ((1/total_new[1])^(p-1)) - ((-1/total_old[1])^(p-1))  # this is not how its shown in the paper, but is how scalib implements it.
            # ^ This improves numerical stability at orders > 4 by avoiding division of 1 by total_new[1]^(p-1), which is quite large at p>4
            tmp *= (((total_old[1] * total_new[1])/total_result[1]) * δ)^p
            CS_old[p, j] += tmp
        end

        CS_old[1, j] += (δ * (total_new[1]/total_result[1]))  # update mean seperately
    end
    
    return nothing
end

function centered_moment(m::UniVarMomentsAccIncremental, d::Int)
    if d == 1
        @views CM_d = m.ctrd_sums[:, :, 1, :]
    else 
        @views CM_d = m.ctrd_sums[:, :, d, :] ./ m.totals
    end
    return CM_d
end

function get_mean_and_var(m::UniVarMomentsAccIncremental, d::Int)
    if d == 1
        @inbounds μ = @view m.ctrd_sums[:, :, 1, :]
        @inbounds σ2 = m.ctrd_sums[:, :, 2, :] ./ m.totals
        return μ, σ2
    elseif d == 2
        @inbounds μ = m.ctrd_sums[:, :, 2, :] ./ m.totals
        @inbounds σ2 = m.ctrd_sums[:, :, 4, :] ./ m.totals
        return μ, σ2
    elseif d > 2
        @inbounds μ = (m.ctrd_sums[:, :, d, :] ./ m.totals) ./ ((m.ctrd_sums[:, :, 2, :] ./ m.totals).^(d/2))
        @inbounds σ2 = ((m.ctrd_sums[:, :, 2*d, :] ./ m.totals) .- ((m.ctrd_sums[:, :, d, :] ./ m.totals).^2)) ./ ((m.ctrd_sums[:, :, 2, :] ./ m.totals).^d)
        return μ, σ2
    end
end

end  # module Moments