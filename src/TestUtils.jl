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

function generate_test_dataset(nt::Int, ns::Int, ldim::Int, ttype::DataType, ltype::DataType)
    a = rand(ttype, nt, ns)
    l = rand(ltype, nt, ldim)
    return a, l
end

function generate_test_dataset(nt::Int, ns::Int, ldim::Int, lrange::UnitRange, ttype::DataType, ltype::DataType)
    a = rand(ttype, nt, ns)
    l = rand(ltype.(lrange), nt, ldim)
    return a, l
end




end