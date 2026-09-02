module TestUtils
export permute_dataset_rows

using Random


function permute_dataset_rows(a::AbstractArray, l::AbstractArray)
    perm = randperm(size(a, 1))
    a = a[perm, :]
    l = l[perm, :]

    return a, l
end

function permute_dataset_rows(a::AbstractArray, l::AbstractArray, perm::AbstractVector)
    # perm = randperm(size(a, 1))
    a = a[perm, :]
    l = l[perm, :]

    return a, l
end


end