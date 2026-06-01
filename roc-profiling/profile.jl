using SCA
using AMDGPU
using KernelAbstractions

function test1(ntraces::Int, tsize::Int, lsize::Int, order::Int, ldomain::UnitRange{UInt8} = 0x0:0xff)
    t_cpu = rand(Float32, ntraces, tsize)
    l_cpu = rand(UInt8, ntraces, lsize)

    # m = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, lsize}(order, tsize, size(ldomain, 1))
    # Moments.centered_sum_update!(m, t_cpu, l_cpu)
    # Moments.centered_sum_KA_wrapper!(m._moments, t_cpu, l_cpu, 
        # Val((64, 4, 2)), Val((2, 2, 8)), Val(4), Val(lsize), Val(256), Val(4))
    # KernelAbstractions.synchronize(get_backend(t_cpu))

    t_dev = ROCArray(t_cpu)
    l_dev = ROCArray(l_cpu)

    m = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, ROCArray, lsize}(order, tsize, size(ldomain, 1))
    # Moments.centered_sum_update!(m, t_dev, l_dev)
    Moments.centered_sum_KA_wrapper!(m._moments, t_dev, l_dev, 
        Val((64, 4, 4)), Val((1, 1, 1)), Val(4), Val(lsize), Val(256), Val(4))
    KernelAbstractions.synchronize(get_backend(t_dev))
end

# RX9070XT: 458ms pass 1 + 3.28s pass 2
# RX9070XT (KA kernel): 35.512ms pass 2
test1(100000, 1000, 16, 4)