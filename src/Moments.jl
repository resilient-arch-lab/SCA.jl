"""
Parallel estimation of statistical moments
"""

"""
TODO:
- Get rid of non-VecLabel acc and make the VecLabel methods handle label vectors (just rehsape them as n X 1 matrices)
- Make a single function `fit!` to apply to incremental and non incremental MomentsAccs
- Make functions `raw_moments`, `central_moments`, and `standardized_moments` for `AbstractMomentsAcc`
- Move multivariate stuff here from testing branch
"""


module Moments
export fit_moments!, merge_from!, get_mean_and_var, UniVarMomentsAccIncremental, centered_sum_update_pass_1!, centered_sum_update_pass_2!

include("Utils.jl")
using .Utils

using Random
using KernelAbstractions, Atomix
import AcceleratedKernels as AK
using Base: convert
using StaticArrays

abstract type AbstractMomentsAcc{Tt, Tl, Ta} end
abstract type AbstractUnivariateMomentsAcc{Tt, Tl, Ta} <: AbstractMomentsAcc{Tt, Tl, Ta} end
abstract type AbstractMultivariateMomentsAcc{Tt, Tl, Ta} <: AbstractMomentsAcc{Tt, Tl, Ta} end


struct UniVarMomentsAcc{Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray} <: AbstractUnivariateMomentsAcc{Tt, Tl, Ta}
    totals::Ta
    ctrd_sums::Ta
    order::UInt
    ns::UInt
    lrange::UInt
    ldim::UInt
    sums::Ta
end

function UniVarMomentsAcc{Tt, Tl, Ta}(order, ns, lrange, ldim) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    totals = fill!(Ta{UInt32, 2}(undef, ldim, lrange), 0)
    ctrd_sums = fill!(Ta{Tt, 4}(undef, ldim, lrange, order, ns), 0)
    sums = fill!(Ta{Tt, 3}(undef, ldim, lrange, ns), 0)
    UniVarMomentsAcc{Tt, Tl, Ta}(totals, ctrd_sums, order, ns, lrange, ldim, sums)
end

function UniVarMomentsAcc{Tt, Tl, Ta}(order, a::Ta, labels::Ta) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    @assert typeof(a) <: AbstractVecOrMat "a expected to be a Vector or Matrix, got $(typeof(a))"
    @assert typeof(labels) <: AbstractVecOrMat "labels expected to be a Vector or Matrix, got $(typeof(labels))"
    
    ns = size(a, 2)
    ldim = size(labels, 2)
    lrange = length(unique(labels))
    UniVarMomentsAcc{Tt, Tl, Ta}(order, ns, lrange, ldim)
end


struct UniVarMomentsAccIncremental{Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray} <: AbstractUnivariateMomentsAcc{Tt, Tl, Ta}
    totals::Ta
    ctrd_sums::Ta
    order::UInt
    ns::UInt
    lrange::UInt
    ldim::UInt
    _totals::Ta
    _ctrd_sums::Ta
    _sums::Ta
end

function UniVarMomentsAccIncremental{Tt, Tl, Ta}(order, ns, lrange, ldim) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray} 
    totals = fill!(Ta{UInt32, 2}(undef, ldim, lrange), 0)
    ctrd_sums = fill!(Ta{Tt, 4}(undef, ldim, lrange, order, ns), 0)
    _totals = fill!(similar(totals), 0)
    _ctrd_sums = fill!(similar(ctrd_sums), 0)
    _sums = fill!(Ta{Tt, 3}(undef, ldim, lrange, ns), 0)
    UniVarMomentsAccIncremental{Tt, Tl, Ta}(totals, ctrd_sums, order, ns, lrange, ldim, _totals, _ctrd_sums, _sums)
end

# initialize from dataset shape and labels
function UniVarMomentsAccIncremental{Tt, Tl, Ta}(order, a::Ta, labels::Ta) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    @assert typeof(a) <: AbstractVecOrMat "a expected to be a Vector or Matrix, got $(typeof(a))"
    @assert typeof(labels) <: AbstractVecOrMat "labels expected to be a Vector or Matrix, got $(typeof(labels))"
    
    ns = size(a, 2)
    ldim = size(labels, 2)
    lrange = length(unique(labels))

    totals = fill!(Ta{UInt32, 2}(undef, ldim, lrange), 0)
    ctrd_sums = fill!(Ta{Tt, 4}(undef, ldim, lrange, order, ns), 0)
    _totals = fill!(similar(totals), 0)
    _ctrd_sums = fill!(similar(ctrd_sums), 0)
    _sums = fill!(Ta{Tt, 3}(undef, ldim, lrange, ns), 0)
    UniVarMomentsAccIncremental{Tt, Tl, Ta}(totals, ctrd_sums, order, ns, lrange, ldim, _totals, _ctrd_sums, _sums)
end

# initialize from non-incremental struct
function UniVarMomentsAccIncremental{Tt, Tl, Ta}(acc::UniVarMomentsAcc) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    ns = acc.ns
    ldim = acc.ldim
    lrange = acc.lrange

    totals = acc.totals
    ctrd_sums = acc.ctrd_sums
    _totals = fill!(similar(totals), 0)
    _ctrd_sums = fill!(similar(ctrd_sums), 0)
    _sums = fill!(Ta{Tt, 3}(undef, ldim, lrange, ns), 0)
    UniVarMomentsAccIncremental{Tt, Tl, Ta}(totals, ctrd_sums, order, ns, lrange, ldim, _totals, _ctrd_sums, _sums)
end


struct MultiVarMomentsAccIncremental{Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray} <: AbstractMultivariateMomentsAcc{Tt, Tl, Ta}
    totals::Ta
    SCPs::Ta  # sums of centered products
    α::Matrix{Int}  # order vectors (vector rows)
    ns::UInt  # number of samples per trace (and therefore the variateness of sums of centered prods)
    lrange::UInt
    ldim::UInt
    _totals::Ta
    _SCPs::Ta
    _sums::Ta
end

function MultiVarMomentsAccIncremental{Tt, Tl, Ta}(order::Union{Int, AbstractVector{Int}, AbstractMatrix{Int}}, ns::Integer, lrange::Integer, ldim::Integer) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    if typeof(order) == Int
        α = fill!(Ta{Int, 2}(undef, 1, ns), order)  # the same order is calculated for each sample position 
    elseif typeof(order) <: AbstractVector{Int}
        α = reshape(order, 1, ns)
    else typeof(order) <: AbstractMatrix{Int}
        checkbounds(order, 1, ns)
        α = order
    end
    
    totals = fill!(Ta{UInt32, 2}(undef, ldim, lrange), 0)
    SCPs = fill!(Ta{Tt, 4}(undef, ldim, lrange, size(α, 1), 1), 0)

    _totals = similar(totals)
    _SCPs = similar(SCPs)
    _sums = Ta{Tt, 3}(undef, ldim, lrange, ns)
    MultiVarMomentsAccIncremental{Tt, Tl, Ta}(totals, SCPs, α, ns, lrange, ldim, _totals, _SCPs, _sums)
end


# simple, sequential label-wise sum op for cpu
@inline function label_wise_sum!(traces::AbstractVecOrMat{Tt}, labels::AbstractVecOrMat{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractArray) where {Tt<:AbstractFloat, Tl<:Integer}
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

# no method matching label_wise_sum_ak_transposed!(::Matrix{Float64}, ::Matrix{UInt8}, ::Array{Float64, 3}, ::Array{UInt32, 3})
function label_wise_sum_ak_transposed!(traces::AbstractVecOrMat{Tt}, labels::AbstractVecOrMat{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractArray{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
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

function label_wise_sum_ak!(traces::AbstractVecOrMat{Tt}, labels::AbstractVecOrMat{Tl}, sums::AbstractArray{Tt, 3}, totals::AbstractArray{UInt32}) where {Tt<:AbstractFloat, Tl<:Integer}
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


# simple, sequential centered sum update op for cpu
@inline function centered_sum!(ctrd_sums::AbstractArray{Tt, 4}, traces::AbstractVecOrMat{Tt}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
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

function centered_sum_kern_ak!(moments::AbstractArray{Tt, 4}, traces::AbstractVecOrMat{Tt}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
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

function centered_sum_kern_ak_atomic!(moments::AbstractArray{Tt, 4}, traces::AbstractVecOrMat{Tt}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
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


# First pass in two pass approach
function centered_sum_update_pass_1!(sums::AbstractArray{Tt}, totals::AbstractArray{UInt32}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    label_wise_sum_ak_transposed!(traces, labels, sums, totals)
    return
end

function centered_sum_update_pass_1!(acc::UniVarMomentsAccIncremental{Tt, Tl, Ta}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    @boundscheck begin
        checkbounds(acc._sums, acc.ldim, acc.lrange, size(traces, 2))
        checkbounds(acc._ctrd_sums, acc.ldim, acc.lrange, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), acc.ldim)
    end
    
    fill!.([acc._ctrd_sums, acc._sums, acc._totals], 0)

    centered_sum_update_pass_1!(acc._sums, acc._totals, traces, labels)

    return
end

function centered_sum_update_pass_1!(acc::UniVarMomentsAcc{Tt, Tl, Ta}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    @boundscheck begin
        checkbounds(acc.sums, acc.ldim, acc.lrange, size(traces, 2))
        checkbounds(acc.ctrd_sums, acc.ldim, acc.lrange, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), acc.ldim)
    end
    
    centered_sum_update_pass_1!(acc.sums, acc.totals, traces, labels)

    return
end

# Second pass in two pass approach
function centered_sum_update_pass_2!(ctrd_sums::AbstractArray{Tt}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    centered_sum_kern_ak!(ctrd_sums, traces, labels)
    return
end

function centered_sum_update_pass_2!(acc::UniVarMomentsAccIncremental{Tt, Tl, Ta}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
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
        @warn "Centered sum estimation merging is an experimental feature and introduces significant error (up to 500% in some tests). Do not use if accuracy is important"
        for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
            @inbounds merge_from_ak!(view(acc.ctrd_sums, l, :, :), view(acc.totals, l), view(acc._ctrd_sums, l, :, :), view(acc._totals, l))
        end
        @inbounds @views acc.totals[update_ls] .+= acc._totals[update_ls]
    end

    return
end

function centered_sum_update_pass_2!(acc::UniVarMomentsAcc{Tt, Tl, Ta}, traces::AbstractArray{Tt}, labels::AbstractArray{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    @boundscheck begin
        checkbounds(acc.sums, acc.ldim, acc.lrange, size(traces, 2))
        checkbounds(acc.ctrd_sums, acc.ldim, acc.lrange, acc.order, size(traces, 2))
        checkbounds(labels, size(traces, 1), acc.ldim)
    end

    @. acc.ctrd_sums[:, :, 1, :] = acc.sums / acc.totals

    centered_sum_kern_ak!(acc.ctrd_sums, traces, labels)

    return
end


function fit_moments!(acc::AbstractUnivariateMomentsAcc, traces, labels)
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
function merge_from_ak!(CS1::AbstractArray{Tt, 2}, n1::AbstractArray{UInt32, 0}, CS2::AbstractArray{Tt, 2}, n2::AbstractArray{UInt32, 0}) where { Tt<:AbstractFloat }
    @boundscheck begin
        checkbounds(CS2, size(CS1)...)
        checkbounds(n2, size(n1)...)
    end
    
    order = size(CS1, 1)
    n1 = n1[1]
    n2 = n2[1]
    n = n1 + n2

    @inbounds AK.foraxes(CS1, 2) do j  # most allocations here 
        δ_21 = CS2[1, j] - CS1[1, j]
        
        for p in order:-1:2  # p = order to update
            CS1[p, j] += CS2[p, j]

            # V still fucked up
            M_tmp = 0.0
            for k in 1:p-2
                pck = binomial(Int(p), Int(k))  # explicit Int32 cast avoids unnecessary use of arbitrary precision arithmetic
                tmp1 = CS1[p-k, j] * ((-n2/n)^k)
                tmp2 = CS2[p-k, j] * ((n1/n)^k)
                tmp3 = tmp1 + tmp2
                M_tmp += ((δ_21^k) * pck) * tmp3
            end
            CS1[p, j] += M_tmp

            # if (M_tmp >= 100) && (j == 1)
            #     @error "Error 1: M_tmp = $(M_tmp)\tδ_21=$(δ_21)\tp=$(p)"
            # end

            # with batches of size 10000, this section is stable with float64 up to at least order 16 within 5 decimal places
            tmp = (((1/n2)^(p-1)) - ((-1/n1)^(p-1))) * ((((n1 * n2)/n) * δ_21)^p)  # this is not how its shown in the paper, but is how scalib implements it.
            # ^ This improves numerical stability at orders > 4 by avoiding division of 1 by n2^(p-1), which is quite large at p>4

            # tmp = (n1*((-n2/n)*δ_21)^p) + (n2*((n1/n)*δ_21)^p)
            # if !(tmp ≈ (((1/n2)^(p-1)) - ((-1/n1)^(p-1))) * (((n1 * n2)/n) * δ_21)^p) && (j == 1)
            #     @error "Error 1: $((((1/n2)^(p-1)) - ((-1/n1)^(p-1))) * (((n1 * n2)/n) * δ_21)^p) vs $(tmp)\tδ_21=$(δ_21)\tp=$(p)"
            # end
            
            CS1[p, j] += tmp
        end

        CS1[1, j] += (δ_21 * (n2/n))  # update mean seperately
    end
    
    return nothing
end

function raw_moments(m::AbstractUnivariateMomentsAcc, d::Int)
    @assert d >= 1 "cannot compute raw moment of order < 1 from MomentsAcc struct"

    if d == 1
        M_d = m.ctrd_sums[:, :, 1, :]
    else
        @error "cannot compute raw moment of order > 1 from MomentsAcc struct"
    end

    return M_d
end

function centeral_moments(m::AbstractUnivariateMomentsAcc, d::Int)
    @assert d >= 2 "cannot compute centered moment of order < 2 from MomentsAcc struct"

    CM_d = @view(m.ctrd_sums[:, :, d, :]) ./ m.totals
    return CM_d
end

function standardized_moments(m::AbstractUnivariateMomentsAcc, d::Int)
    @assert d >= 2 "cannot comute standardized moment of order < 2 from MomentsAcc struct"

    SM_d = centeral_moments(m, d) ./ (centeral_moments(m, 2) .^ (d / 2))

    return SM_d
end

function get_mean_and_var(m::AbstractUnivariateMomentsAcc, d::Int)
    if d == 1
        @inbounds μ = raw_moments(m, 1)
        @inbounds σ2 = centeral_moments(m, 2)
        return μ, σ2
    elseif d == 2
        @inbounds μ = centeral_moments(m, 2)
        @inbounds σ2 = centeral_moments(m, 4) .- (μ.^2)
        return μ, σ2
    elseif d > 2
        @inbounds μ = standardized_moments(m, d)
        @inbounds σ2 = (centeral_moments(m, 2*d) .- (centeral_moments(m, d).^2)) ./ (centeral_moments(m, 2).^d)
        return μ, σ2
    end
end


# Multivariate methods

function centered_sum_kern_ak!(SCPs::AbstractArray{Tt, 4}, traces::AbstractVecOrMat{Tt}, labels::AbstractVecOrMat{Tl}, order::AbstractMatrix{Int}, means::AbstractArray{Tt, 3}) where {Tt<:AbstractFloat, Tl<:Integer}
    @boundscheck begin
        # TODO
    end

    itr_view = @view SCPs[:, 1, :, 1]

    # parallelize over orders (for now)
    AK.foreachindex(itr_view) do idx
        (l, o) = CartesianIndices(itr_view)[idx].I
        for i in axes(traces, 1)
            l_i = convert(Int32, labels[i, l]+1)
            ctrd_prod = 1
            for j in axes(traces, 2)
                ctrd_prod *= (traces[i, j] - means[l, l_i, j]) ^ order[o, j]
            end
            SCPs[l, l_i, o, 1] += ctrd_prod
        end
    end
end

function centered_sum_update!(acc::MultiVarMomentsAccIncremental{Tt, Tl, Ta}, traces::AbstractVecOrMat{Tt}, labels::AbstractVecOrMat{Tl}) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    fill!(acc._sums, 0)
    fill!(acc._totals, 0)
    fill!(acc._SCPs, 0)
    
    # Pass 1, calculate labels wise sums
    label_wise_sum_ak!(traces, labels, acc._sums, acc._totals)

    # Pass 2: find means and calculate sums of centered prods
    means = acc._sums ./ acc._totals
    centered_sum_kern_ak!(acc._SCPs, traces, labels, acc.α, means)

    acc.SCPs .= acc._SCPs
    acc.totals .= acc._totals
end

end  # module Moments