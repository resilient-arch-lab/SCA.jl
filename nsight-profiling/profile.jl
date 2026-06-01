using SCA
using CUDA
using KernelAbstractions
using Random

function test1(ntraces::Int, tsize::Int, order::Int, ::Val{lsize}, ldomain::UnitRange{UInt8} = 0x0:0xff) where {lsize}
    Random.seed!(12)
    t_cpu = rand(Float32, ntraces, tsize)
    l_cpu = rand(UInt8, ntraces, lsize)

    # m = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, lsize}(order, tsize, size(ldomain, 1))
    # Moments.centered_sum_update!(m, t_cpu, l_cpu)
    # Moments.centered_sum_KA_wrapper!(m._moments, t_cpu, l_cpu, 
        # Val((64, 4, 2)), Val((2, 2, 8)), Val(4), Val(lsize), Val(256), Val(4))
    # KernelAbstractions.synchronize(get_backend(t_cpu))

    t_dev = CuArray(t_cpu)
    l_dev = CuArray(l_cpu)

    m = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, lsize}(order, tsize, size(ldomain, 1))
    # CUDA.@profile Moments.centered_sum_update!(m, t_dev, l_dev)  # error on thread (1, 40), block (6, 1)
    res = CUDA.@profile Moments.centered_sum_KA_wrapper!(m._moments, t_dev, l_dev, 
       Val((64, 4, 1)), Val((2, 16, 16)), Val(8), Val(lsize), Val(256), Val(4))
    # KernelAbstractions.synchronize(get_backend(t_dev))
    show(res)
end

# 3050ti (AK kernel): 
# 3050ti (KA kernel): 59.13ms pass 2
test1(100000, 1000, 8, Val(16))