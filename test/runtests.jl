using SCA
using Test
using Statistics
using Random


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

@testset "Centered products estimation satability test 1: re-ordering error vs. float length (no merging)" begin
    NL = 2
    a32 = rand(Float32, 20000, 20)
    a64 = rand(Float64, 20000, 20)
    l = rand(UInt8, 20000, NL)

    n = 10
    order = 16
    results32 = []
    results64 = []
    
    println("Dataset parameters: $(size(a32, 1))x$(size(a32, 2)) traces\t$(NL) label length")
    println("Statistic parameters: centered product order = $(order)")
    println("")
    # perform `n` iterations
    for i in 1:n
        # estimate moments
        m32 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, NL}(order, size(a32, 2), 256)
        m64 = Moments.UniVarMomentsAccVecLabel{Float64, UInt8, Array, NL}(order, size(a64, 2), 256)
        Moments.centered_sum_update!(m32, a32, l)
        Moments.centered_sum_update!(m64, a64, l)
        
        # store results
        if isempty(results32)
            results32 = m32.moments
        else
            results32 = cat(results32, m32.moments, dims=5)
        end
        if isempty(results64)
            results64 = m64.moments
        else
            results64 = cat(results64, m64.moments, dims=5)
        end

        # shuffle trace rows and labels 
        perm = randperm(size(a32, 1))
        a32, l = SCA.TestUtils.permute_dataset_rows(a32, l, perm)
        a64, _ = SCA.TestUtils.permute_dataset_rows(a64, l, perm)
    end

    mean32 = mean(results32, dims=5)
    mean64 = mean(results64, dims=5)
    std32 = sqrt.(var(results32, dims=5))
    std64 = sqrt.(var(results64, dims=5))

    println("32 bit report: ")
    println("Std. Dev. per order (averaged over sample positions and $(n) iterations):")
    display(vec(mean(std32, dims=(1, 2, 4)))')
    println("")

    println("64 bit report: ")
    println("Std. Dev. per order (averaged over sample positions and $(n) iterations):")
    display(vec(mean(std64, dims=(1, 2, 4)))')
    println("")
end

@testset "Centered products estimation satability test 2: RMS error vs. float length (no merging)" begin
    setprecision(BigFloat, 256)
    prec = precision(BigFloat)
    println("Ground truth float precision: $(prec)")
    
    NL = 2
    a32 = rand(Float32, 20000, 20)
    a64 = Float64.(a32)
    a256 = big.(a64)
    l = rand(UInt8, 20000, NL)

    n = 10
    order = 16
    results32 = []
    results64 = []

    println("Dataset parameters: $(size(a32, 1))x$(size(a32, 2)) traces\t$(NL) label length")
    println("Statistic parameters: centered product order = $(order)")
    println("")

    # calculate ground truth
    m256 = Moments.UniVarMomentsAccVecLabel{BigFloat, UInt8, Array, NL}(order, size(a64, 2), 256)
    Moments.centered_sum_update!(m256, a256, l)
    resultsref = m256.moments

    # perform `n` iterations
    for i in 1:n
        # estimate moments
        m32 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, NL}(order, size(a32, 2), 256)
        m64 = Moments.UniVarMomentsAccVecLabel{Float64, UInt8, Array, NL}(order, size(a64, 2), 256)
        Moments.centered_sum_update!(m32, a32, l)
        Moments.centered_sum_update!(m64, a64, l)
        
        # store results
        if isempty(results32)
            results32 = m32.moments
        else
            results32 = cat(results32, m32.moments, dims=5)
        end
        if isempty(results64)
            results64 = m64.moments
        else
            results64 = cat(results64, m64.moments, dims=5)
        end

        # shuffle trace rows and labels 
        perm = randperm(size(a32, 1))
        a32, l = SCA.TestUtils.permute_dataset_rows(a32, l, perm)
        a64, _ = SCA.TestUtils.permute_dataset_rows(a64, l, perm)
    end

    # calculate average RMS error
    rms32 = sqrt.(mean((resultsref .- results32).^2, dims=5))
    rms64 = sqrt.(mean((resultsref .- results64).^2, dims=5))

    println("32 bit report: ")
    println("RMS error vs reference (averaged over sample positions and $(n) iterations):")
    display(vec(mean(rms32, dims=(1, 2, 4)))')
    println("Maximum RMS error: ")
    display(vec(maximum(rms32, dims=(1, 2, 4)))')
    println("")
    
    println("64 bit report: ")
    println("RMS error vs reference (averaged over sample positions and $(n) iterations):")
    display(vec(mean(rms64, dims=(1, 2, 4)))')
    println("Maximum RMS error: ")
    display(vec(maximum(rms64, dims=(1, 2, 4)))')
    println("")
end

@testset "Centered products estimation satability test 3: RMS error vs. float length and number of merges" begin
    setprecision(BigFloat, 256)
    prec = precision(BigFloat)
    println("Ground truth float precision: $(prec)")
    
    NL = 2
    a32 = rand(Float32, 20000, 20)
    a64 = Float64.(a32)
    a256 = big.(a64)
    l = rand(UInt8, 20000, NL)

    batch_settings = [1, 2, 4, 10]
    order = 16
    
    ref_results = Dict{Int, AbstractArray}()

    println("Dataset parameters: $(size(a32, 1))x$(size(a32, 2)) traces\t$(NL) label length")
    println("Statistic parameters: centered product order = $(order)")
    println("")

    for batches in batch_settings
        # initialize accumulator structs
        m256 = Moments.UniVarMomentsAccVecLabel{BigFloat, UInt8, Array, NL}(order, size(a256, 2), 256)
        m64 = Moments.UniVarMomentsAccVecLabel{Float64, UInt8, Array, NL}(order, size(a64, 2), 256)
        m32 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, Array, NL}(order, size(a32, 2), 256)
        
        batch_size = (Int(ceil(size(a32, 1) / batches)), size(a32, 2))

        a32_batches = Utils.tiled_view(a32, batch_size)
        a64_batches = Utils.tiled_view(a64, batch_size)
        a256_batches = Utils.tiled_view(a256, batch_size)
        l_batches = Utils.tiled_view(l, batch_size)
        
        # run batch workloads
        for batch in 1:batches
            Moments.centered_sum_update!(m256, a256_batches[batch, 1], l_batches[batch, 1])
            Moments.centered_sum_update!(m64, a64_batches[batch, 1], l_batches[batch, 1])
            Moments.centered_sum_update!(m32, a32_batches[batch, 1], l_batches[batch, 1])
        end
        
        # get final results
        results32 = m32.moments
        results64 = m64.moments
        results256 = m256.moments

        # analysis
        if batches == 1
            ref_results[32] = results32
            ref_results[64] = results64
            ref_results[256] = results256
        else
            rms32 = sqrt.((ref_results[32] .- results32).^2)
            rms64 = sqrt.((ref_results[64] .- results64).^2)
            rms256 = sqrt.((ref_results[256] .- results256).^2)

            mean32 = vec(mean(rms32, dims=(1, 2, 4)))'
            max32 = vec(maximum(rms32, dims=(1, 2, 4)))'
            mean64 = vec(mean(rms64, dims=(1, 2, 4)))'
            max64 = vec(maximum(rms64, dims=(1, 2, 4)))'
            mean256 = vec(mean(rms256, dims=(1, 2, 4)))'
            max256 = vec(maximum(rms256, dims=(1, 2, 4)))'

            println("$(batches) Batches:")
            println("\t32 bit mean and max errors vs reference:")
            display(mean32)
            display(max32)
            println("\t64 bit mean and max errors vs reference:")
            display(mean64)
            display(max64)
            println("\t256 bit mean and max errors vs reference:")
            display(mean256)
            display(max256)
            println("")
        end
    end




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