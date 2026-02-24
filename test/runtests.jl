using SCA
using Test
using Statistics

@testset "Test SNR chunked implementation" begin
    t = rand(Float64, 2000, 1000)
    l = rand(UInt8, 2000)

    snr1 = SNR.SNRMoments{Float64, UInt8}(1000, 256)
    snr2 = SNR.SNRMomentsChunked{Float64, UInt8}(1000, 256, 200)
    snr3 = SNR.SNRMomentsChunkedMulti{Float64, UInt8}(1000, 256, (1000, 200))

    SNR.SNR_fit!(snr1, t, l)
    SNR.SNR_fit!(snr2, t, l)
    SNR.SNR_fit!(snr3, t, l)

    res1 = SNR.SNR_finalize(snr1)
    res2 = SNR.SNR_finalize(snr2)
    res3 = SNR.SNR_finalize(snr3)

    @test all(res1 .≈ res2)
    @test all(res1 .≈ res3)
end

# Test precision / stability of centered sum merging formula [Prop. 2.1, 10.2172/1028931]
# with reference to a single centered sum calculation on the same data, requiring no merge
# operation. Initial results are inconsistent even with a relative tolarance of 1. 
@testset "Moment merging precision comparison" begin
    a = rand(10000, 20)
    l = rand(UInt8, 10000)
    m1 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)
    m2 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)

    a_tiles, l_tiles = Utils.tiled_view(a, (5000, 20)), Utils.tiled_view(l, (5000, ))

    Moments.centered_sum_update!(m1, a, l)
    for (a_tile, l_tile) in zip(a_tiles, l_tiles)
        Moments.centered_sum_update!(m2, a_tile, l_tile)
    end

    correct = all(isapprox.(m1.moments, m2.moments; rtol=1e-2))
    prcnt_err = abs.((m2.moments .- m1.moments) ./ m1.moments).*100

    println("Moment merging algorithm test case percent error per order $(1:m1.order)")
    display(vec(mean(prcnt_err, dims=(1, 3))))
end