module Utils
export tiled_view

using StaticArrays

"""
Divide `A` into `N` dimensional tile views of size `tile_size`, Where the tiling origin is the
element at position 1 on each dimension of `A`. Tiles at the end of a dimension may not be the 
expected `tile_size`.

views of Base.Arrays have a member variable of the indices of the tile in the original array, 
so typically return_indices is not needed. 
"""
function tiled_view(A::AbstractArray{T, N}, tile_size::NTuple{N}; return_indices::Bool = false) where {T, N}
    slices = [[s for s=Iterators.partition(axes(A, d), tile_size[d])] for d=1:ndims(A)]
    tile_indices = collect.(collect(Iterators.product(slices...))) # NDArray of N-element vectors{UnitRange}
    @views out = [A[idx...] for idx in tile_indices]
    if return_indices
        return out, tile_indices
    else
        return out
    end
end

function tile_static(A::AbstractArray{T, N}, ::Val{tile_size})::Array{MArray{Tuple{tile_size...}, T}} where {T, N, tile_size}
    @assert (N == size(tile_size, 1)) & all(size(A) .% tile_size .== 0) "bad tile size"
    out_size = size(A) .÷ tile_size
    out = Array{MArray{Tuple{tile_size...}, T}}(undef, out_size...)
    for tile in eachindex(IndexCartesian(), out)
        tile_start = 1 .+ (tile_size .* (tile.I .- 1))
        tile_end = tile_size .* tile.I
        out[tile] = MArray{Tuple{tile_size...}, T}(A[(tile_start[i]:tile_end[i] for i in 1:N)...])
    end
    out
end


end  # module Utils