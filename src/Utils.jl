module Utils
export tiled_view

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



end  # module Utils