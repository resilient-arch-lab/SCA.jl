using SCA 
using CUDA

function test_1()
    t = CuArray(rand(Float32, 100000, 1000))
    l = CuArray(rand(UInt8, 100000, 16))
    # m2 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, 8}(8, 1000, 256)
    # m3 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, 16}(8, 1000, 256)
    m4 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, CuArray, 4}(8, 1000, 256)

    CUDA.@profile Moments.centered_sum_update!(m4, t, l[:, 1:4])
    m4
end

function main()
    test_1()
end

main()