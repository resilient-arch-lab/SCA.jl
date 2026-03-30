using SCA
using Test
using Statistics

# Test precision / stability of centered sum merging formula [Prop. 2.1, 10.2172/1028931]
# with reference to a single centered sum calculation on the same data, requiring no merge
# operation. Initial results are inconsistent even with a relative tolarance of 1. 
@testset "Moment merging precision comparison (Legacy reference function)" begin
    a = rand(10000, 20)
    l = rand(UInt8, 10000)
    m1 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)
    m2 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)

    a_tiles, l_tiles = Utils.tiled_view(a, (5000, 20)), Utils.tiled_view(l, (5000, ))

    Moments.centered_sum_update_old!(m1, a, l)
    for (a_tile, l_tile) in zip(a_tiles, l_tiles)
        Moments.centered_sum_update_old!(m2, a_tile, l_tile)
    end

    correct = all(isapprox.(m1.moments, m2.moments; rtol=1e-2))
    prcnt_err = abs.((m2.moments .- m1.moments) ./ m1.moments).*100

    println("Moment merging algorithm test case percent error per order $(1:m1.order)")
    display(vec(mean(prcnt_err, dims=(1, 3))))
end

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

@testset "Moment merging kernel comparison to legacy reference function" begin
    a = rand(20000, 5)
    l = rand(UInt8, 20000)
    m1 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 5, 256)
    m2 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 5, 256)

    Moments.centered_sum_update_old!(m1, a[1:10000, :], l[1:10000])
    Moments.centered_sum_update_old!(m1, a[10001:end, :], l[10001:end])
    Moments.centered_sum_update!(m2, a[1:10000, :], l[1:10000])
    Moments.centered_sum_update!(m2, a[10001:end, :], l[10001:end])

    @test all(isapprox.(m1.moments, m2.moments; rtol=1e-2))
    correct = all(isapprox.(m1.moments, m2.moments; rtol=1e-2))
    println("Correct: $correct")

    prcnt_err = abs.((m2.moments .- m1.moments) ./ m1.moments).*100
    println("Moment merging algorithm test case percent error per order $(1:m1.order)")
    display(vec(mean(prcnt_err, dims=(1, 3))))
end

@testset "UniVarMomentsAccNDLabel tests" begin
    a = rand(20000, 20)
    l = rand(UInt8, 20000, 4)
    m1 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)
    m2 = Moments.UniVarMomentsAccNDLabel{Float64, UInt8, Array, 1}(10, 20, 256, (4, ))

    # initialization update
    Moments.centered_sum_update!(m1, a[1:10000, :], l[1:10000, 1])
    Moments.centered_sum_update!(m2, a[1:10000, :], l[1:10000, :])
    @test all(isapprox.(m1.moments, m2.moments[1, :, :, :]))
    println("Update 1 percent error by order:")
    prcnt_err = abs.((m1.moments .- m2.moments[1, :, :, :]) ./ m2.moments[1, :, :, :]).*100
    display(vec(mean(prcnt_err, dims=(1, 3))))


    # merge update
    Moments.centered_sum_update!(m1, a[10001:end, :], l[10001:end, 1])
    Moments.centered_sum_update!(m2, a[10001:end, :], l[10001:end, :])
    @test all(isapprox.(m1.moments, m2.moments[1, :, :, :]))
    println("Update 2 percent error by order:")
    prcnt_err = abs.((m1.moments .- m2.moments[1, :, :, :]) ./ m2.moments[1, :, :, :]).*100
    display(vec(mean(prcnt_err, dims=(1, 3))))

end

@testset "Test that Chunked SNR is equivalent on dimension 2" begin
    t = rand(Float64, 2000, 1000)
    l = rand(UInt8, 2000)

    snr1 = SNR.SNRMoments{Float64, UInt8, Array}(1000, 256)
    snr2 = SNR.SNRMomentsChunked{Float64, UInt8, Array}(1000, 256, (2000, 200))

    SNR.SNR_fit!(snr1, t, l)
    SNR.SNR_fit!(snr2, t, l)

    res1 = SNR.SNR_finalize(snr1)
    res2 = SNR.SNR_finalize(snr2)

    @test all(res1 .≈ res2)
end

@testset "Test that Chunked TTest is equivalent on dimension 2" begin
    t = rand(Float64, 5000, 1000)
    l = UInt8.(rand([0, 1], 5000))

    ttest1 = TTest.TTestSingle{Float64, UInt8}(2, 1000)
    ttest2 = TTest.TTestChunked{Float64, UInt8}(2, 1000, (5000, 200))

    TTest.ttest_fit!(ttest1, t, l)
    TTest.ttest_fit!(ttest2, t, l)

    res1 = TTest.ttest_finalize(ttest1)
    res2 = TTest.ttest_finalize(ttest2)

    @test all(res1 .≈ res2)
end

@testset "centered_sum_update_combined" begin
    a1 = rand(10000, 20)
    a2 = rand(10000, 20)
    l1 = rand(UInt8, 10000)
    l2 = rand(UInt8, 10000)
    m1 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)
    m2 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)

    Moments.centered_sum_update_combined!(m1, a1, l1)
    Moments.centered_sum_update_combined!(m1, a2, l2)
    Moments.centered_sum_update!(m2, a1, l1)
    Moments.centered_sum_update!(m2, a2, l2)

    @test all(isapprox.(m1.moments, m2.moments))
end