using Test
using SCA
using Statistics
using Random
using StatsBase
Random.seed!(12)


@testset "Multivariate Moment Estimation" begin
    ns = 2  # must = 2 so that covariance can be calculated from the sums of centered products
    order = 1  # must = 1 for same reason
    mref = MultiVarMoments.MultiVarMomentsAccReference{Float64, UInt8, Array}(UInt(order), ns, 256)
    m = MultiVarMoments.MultiVarMomentsAcc{Float64, UInt8, Array}(order, ns, 256)
    a = rand(50000, ns)  # lots of measurements are required for `cov` and `centered_sum_update` to converge
                         # since covariance is calculated per label and there are 256 labels
    l = rand(UInt8, 50000)

    set_0 = a[l.==0, :]
    cov_0 = cov(set_0) 

    MultiVarMoments._centered_sum_update_reference!(mref, a, l)
    MultiVarMoments.centered_sum_update!(m, a, l)

    @test all(.≈((mref.SCPs[1, :, :] / size(a[l.==0, :], 1)), cov_0[1, 2], rtol=0.01))
    @test all(.≈(mref.SCPs, m.SCPs))
    @test all(mref.totals .== m.totals)
end

@testset "Multivariate Moment Estimation with VecLabels" begin
    ns = 2  # must = 2 so that covariance can be calculated from the sums of centered products
    order = 1  # must = 1 for same reason
    mref1 = MultiVarMoments.MultiVarMomentsAccReference{Float64, UInt8, Array}(UInt(order), ns, 256)
    mref2 = MultiVarMoments.MultiVarMomentsAccReference{Float64, UInt8, Array}(UInt(order), ns, 256)
    m = MultiVarMoments.MultiVarMomentsAccVecLabel{Float64, UInt8, Array}(order, ns, 256, 2)
    a = rand(50000, ns)  # lots of measurements are required for `cov` and `centered_sum_update` to converge
                         # since covariance is calculated per label and there are 256 labels
    l = rand(UInt8, 50000, 2)

    set_0_1 = a[l[:, 1].==0, :]
    cov_0_1 = cov(set_0_1)
    set_0_2 = a[l[:, 2].==0, :]
    cov_0_2 = cov(set_0_2)

    MultiVarMoments._centered_sum_update_reference!(mref1, a, vec(l[:, 1]))
    MultiVarMoments._centered_sum_update_reference!(mref2, a, vec(l[:, 2]))
    MultiVarMoments.centered_sum_update!(m, a, l)

    @test all(.≈((mref1.SCPs[1, :, :] / size(a[l[:, 1].==0, :], 1)), cov_0_1[1, 2], rtol=0.01))
    @test all(.≈((mref2.SCPs[1, :, :] / size(a[l[:, 2].==0, :], 1)), cov_0_2[1, 2], rtol=0.01))
    @test all(.≈(mref1.SCPs, m.SCPs[1, :, :, :]))
    @test all(.≈(mref2.SCPs, m.SCPs[2, :, :, :]))
    @test all(mref1.totals .== m.totals[1, :])
    @test all(mref2.totals .== m.totals[2, :])
end

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

@testset "Test that Chunked SNR is equivalent on dimension 2" begin
    t = rand(Float64, 5000, 1000)
    l = rand(UInt8, 5000)

    snr1 = SNR.SNRMoments{Float64, UInt8, Array}(1000, 256)
    snr2 = SNR.SNRMomentsChunked{Float64, UInt8, Array}(1000, 256, (5000, 200))

    SNR.SNR_fit!(snr1, t, l)
    SNR.SNR_fit!(snr2, t, l)

    res1 = SNR.SNR_finalize(snr1)
    res2 = SNR.SNR_finalize(snr2)

    @test all(res1 .≈ res2)

    snr3 = SNR.SNRMomentsChunked{Float64, UInt8, Array}(1000, 256, (5000, 200))
    for slice in keys(snr3.chunk_map)
        SNR.SNR_fit!(snr3, slice, t[:, slice], l[:])
    end
    res3 = SNR.SNR_finalize(snr3)
    @test all(res1 .≈ res3)
end

@testset "Test that Chunked TTest is equivalent on dimension 2" begin
    t = rand(Float64, 5000, 1000)
    l = UInt8.(rand([0, 1], 5000))

    ttest1 = TTest.TTestSingle{Float64, UInt8, Array}(2, 1000)
    ttest2 = TTest.TTestChunked{Float64, UInt8, Array}(2, 1000, (5000, 200))

    TTest.ttest_fit!(ttest1, t, l)
    TTest.ttest_fit!(ttest2, t, l)

    res1 = TTest.ttest_finalize(ttest1)
    res2 = TTest.ttest_finalize(ttest2)

    @test all(res1 .≈ res2)
end