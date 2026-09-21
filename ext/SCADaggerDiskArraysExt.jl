module SCADaggerDiskArraysExt
using DiskArrays
using Dagger

Base.getindex(arr::DiskArrays.AbstractDiskArray, d::ArrayDomain) = arr[d.indexes...]


end