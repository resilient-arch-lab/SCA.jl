module SCADaggerZarrDiskArraysExt

using SCA
using DiskArrays
using Zarr
using Dagger

# Base.getindex(arr::Zarr.ZArray, d::ArrayDomain) = arr[d.indexes...]

# Cant load from disk in a truly lazy way, since DArrays are made of chunks, and delayed tasks are not chunks

function Dagger.distribute(Za::Zarr.ZArray{T, N}) where {T, N}
    Za_chunks = DiskArrays.eachchunk(Za)

    starts = Tuple(map(x -> x.offset + 1, Za_chunks.chunks))
    cumsums = Tuple(map(grid_chunk -> grid_chunk[n].stop, Za_chunks[((1 for _ in 1:(n - 1))..., Colon(), (1 for _ in 1:(N - n))...)...]) for n in 1:N)
    subdomains = Dagger.DomainBlocks{N}(starts, cumsums)
    
    chunks = [Dagger.@spawn(map(c -> Za[c...], chunk)) for chunk in Za_chunks]

    partitioning = Blocks(map(c -> c.chunksize, Za_chunks.chunks))
    
    DArray(T, Dagger.domain(Za), subdomains, chunks, partitioning)
end

Dagger.distribute(Za::Zarr.ZArray{T, N}, ::AutoBlocks) where {T, N} = Dagger.distribute(Za)

function Dagger.distribute(Za::Zarr.ZArray{T, N}, dist::Blocks{N}) where {T, N}
    starts = Tuple(1 for _ in 1:N)
    cumsums = Tuple(map(slice -> slice.stop, Utils.tiled_view(axes(Za, n), (dist.blocksize[n], ))) for n in 1:N)
    subdomains = Dagger.DomainBlocks{N}(starts, cumsums)
    
    chunks = [Dagger.@spawn(Za[chunk.indexes...]) for chunk in subdomains]
    
    DArray(T, Dagger.domain(Za), subdomains, chunks, dist)
end

function Dagger.distribute(Za::Zarr.ZArray{T, N}, dist::Blocks{N}, assignment::Union{AbstractArray{<:Int64, N}, AbstractArray{<:Dagger.Processor, N}}) where {T, N}
    procgrid = Dagger.build_procgrid(assignment, size(Za), dist.blocksize, Dagger.current_acceleration())
    
    subdomains = Dagger.partition(dist, Dagger.domain(Za))
    
    chunks = Dagger.emit_chunk_tasks!(subdomains, procgrid, T,
            (scope, I, i) -> begin
            c = subdomains[I]
            Dagger.@spawn compute_scope=scope Za[c.indexes...]
    end)
    
    DArray(T, Dagger.domain(Za), subdomains, chunks, dist)
end

end