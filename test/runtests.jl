using SCA
using Statistics
using Random
using Test

using StatsBase
Random.seed!(12)


@testset "Multivariate Moment Estimation" begin
    # NOTE: this test doesn't always pass, even though the RNG has a fixed seed. Why?
    ns = 2  # must = 2 so that covariance can be calculated from the sums of centered products
    order = 1  # must = 1 for same reason
    m = Moments.MultiVarMomentsAcc{Float64, UInt8, Array}(order, ns, 256, 1)
    a = rand(50000, ns)  # lots of measurements are required for `cov` and `centered_sum_update` to converge
                         # since covariance is calculated per label and there are 256 labels
    l = rand(UInt8, 50000)

    set_0 = a[l.==0, :]
    cov_0 = cov(set_0) 

    Moments.fit_moments!(m, a, l)

    @test all(.≈((m.SCPs[1, 1, :, :] / size(a[l.==0, :], 1)), cov_0[1, 2], rtol=0.01))
end

@testset "Moment estimation dataset shape compatibility test" begin
    amat = rand(Float32, 20000, 20)
    avec = vec(amat[:, 1])
    lmat = rand(UInt8, 20000, 4)
    lvec = vec(lmat[:, 1])

    m_avec_lvec = Moments.UniVarMomentsAccIncremental{Float32, UInt8, Array}(2, avec, lvec)
    m_avec_lmat = Moments.UniVarMomentsAccIncremental{Float32, UInt8, Array}(2, avec, lmat)
    m_amat_lvec = Moments.UniVarMomentsAccIncremental{Float32, UInt8, Array}(2, amat, lvec)
    m_amat_lmat = Moments.UniVarMomentsAccIncremental{Float32, UInt8, Array}(2, amat, lmat)

    Moments.fit_moments!(m_avec_lvec, avec, lvec)
    Moments.fit_moments!(m_avec_lmat, avec, lmat)
    Moments.fit_moments!(m_amat_lvec, amat, lvec)
    Moments.fit_moments!(m_amat_lmat, amat, lmat)

    @test all(m_avec_lvec.ctrd_sums[1, :, :, :] .≈ m_amat_lvec.ctrd_sums[1, :, :, 1])
    @test all(m_avec_lvec.ctrd_sums[1, :, :, :] .≈ m_avec_lmat.ctrd_sums[1, :, :, 1])
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
        m32 = Moments.UniVarMomentsAccIncremental{Float32, UInt8, Array}(order, size(a32, 2), 256, NL)
        m64 = Moments.UniVarMomentsAccIncremental{Float64, UInt8, Array}(order, size(a64, 2), 256, NL)
        Moments.fit_moments!(m32, a32, l)
        Moments.fit_moments!(m64, a64, l)
        
        # store results
        if isempty(results32)
            results32 = m32.ctrd_sums
        else
            results32 = cat(results32, m32.ctrd_sums, dims=5)
        end
        if isempty(results64)
            results64 = m64.ctrd_sums
        else
            results64 = cat(results64, m64.ctrd_sums, dims=5)
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

@testset "Centered products estimation satability test 2: % error vs. float length (no merging)" begin
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
    m256 = Moments.UniVarMomentsAccIncremental{BigFloat, UInt8, Array}(order, size(a64, 2), 256, NL)
    Moments.fit_moments!(m256, a256, l)
    resultsref = m256.ctrd_sums

    # perform `n` iterations
    for i in 1:n
        # estimate moments
        m32 = Moments.UniVarMomentsAccIncremental{Float32, UInt8, Array}(order, size(a32, 2), 256, NL)
        m64 = Moments.UniVarMomentsAccIncremental{Float64, UInt8, Array}(order, size(a64, 2), 256, NL)
        Moments.fit_moments!(m32, a32, l)
        Moments.fit_moments!(m64, a64, l)
        
        # store results
        if isempty(results32)
            results32 = m32.ctrd_sums
        else
            results32 = cat(results32, m32.ctrd_sums, dims=5)
        end
        if isempty(results64)
            results64 = m64.ctrd_sums
        else
            results64 = cat(results64, m64.ctrd_sums, dims=5)
        end

        # shuffle trace rows and labels
        perm = randperm(size(a32, 1))
        a32, l = SCA.TestUtils.permute_dataset_rows(a32, l, perm)
        a64, _ = SCA.TestUtils.permute_dataset_rows(a64, l, perm)
    end

    # calculate average RMS error
    # rms32 = sqrt.(mean((resultsref .- results32).^2, dims=5))
    # rms64 = sqrt.(mean((resultsref .- results64).^2, dims=5))

    rel32 = abs.((results32 .- resultsref)) .* 100
    rel64 = abs.((results64 .- resultsref)) .* 100

    mean32 = vec(mean(rel32, dims=(1, 2, 4, 5)))'
    max32 = vec(maximum(rel32, dims=(1, 2, 4, 5)))'
    mean64 = vec(mean(rel64, dims=(1, 2, 4, 5)))'
    max64 = vec(maximum(rel64, dims=(1, 2, 4, 5)))'


    println("32 bit report: ")
    println("% error vs reference (averaged over sample positions and $(n) iterations):")
    display(mean32)
    println("Maximum % error: ")
    display(max32)
    println("")
    
    println("64 bit report: ")
    println("% error vs reference (averaged over sample positions and $(n) iterations):")
    display(mean64)
    println("Maximum % error: ")
    display(max64)
    println("")
end

@testset "Centered products estimation satability test 3: % error vs. float length and number of merges" begin
    setprecision(BigFloat, 256)
    prec = precision(BigFloat)
    println("Ground truth float precision: $(prec)")
    
    NL = 1
    a32 = rand(Float32, 40000, 20)
    a64 = Float64.(a32)
    a256 = big.(a64)
    l = rand(UInt8, 40000, NL)

    batch_settings = [1, 2, 4, 8]
    order = 16
    
    ref_results = Dict{Int, AbstractArray}()

    println("Dataset parameters: $(size(a32, 1))x$(size(a32, 2)) traces\t$(NL) label length")
    println("Statistic parameters: centered product order = $(order)")
    println("")

    for batches in batch_settings
        # initialize accumulator structs
        m256 = Moments.UniVarMomentsAccIncremental{BigFloat, UInt8, Array}(order, size(a256, 2), 256, NL)
        m64 = Moments.UniVarMomentsAccIncremental{Float64, UInt8, Array}(order, size(a64, 2), 256, NL)
        m32 = Moments.UniVarMomentsAccIncremental{Float32, UInt8, Array}(order, size(a32, 2), 256, NL)
        
        batch_size = (Int(ceil(size(a32, 1) / batches)), size(a32, 2))

        a32_batches = Utils.tiled_view(a32, batch_size)
        a64_batches = Utils.tiled_view(a64, batch_size)
        a256_batches = Utils.tiled_view(a256, batch_size)
        l_batches = Utils.tiled_view(l, batch_size)
        
        # run batch workloads
        for batch in 1:batches
            Moments.fit_moments!(m256, a256_batches[batch, 1], l_batches[batch, 1])
            Moments.fit_moments!(m64, a64_batches[batch, 1], l_batches[batch, 1])
            Moments.fit_moments!(m32, a32_batches[batch, 1], l_batches[batch, 1])
        end
        
        # get final results
        results32 = m32.ctrd_sums
        results64 = m64.ctrd_sums
        results256 = m256.ctrd_sums

        # analysis
        if batches == 1
            ref_results[32] = results32
            ref_results[64] = results64
            ref_results[256] = results256
        else
            rms32 = sqrt.((ref_results[32] .- results32).^2)
            rms64 = sqrt.((ref_results[64] .- results64).^2)
            rms256 = sqrt.((ref_results[256] .- results256).^2)
            rel32 = abs.((results32 .- ref_results[32])) .* 100
            rel64 = abs.((results64 .- ref_results[64])) .* 100
            rel256 = abs.((results256 .- ref_results[256])) .* 100

            mean32 = vec(mean(rel32, dims=(1, 2, 4)))'
            max32 = vec(maximum(rel32, dims=(1, 2, 4)))'
            mean64 = vec(mean(rel64, dims=(1, 2, 4)))'
            max64 = vec(maximum(rel64, dims=(1, 2, 4)))'
            mean256 = vec(mean(rel256, dims=(1, 2, 4)))'
            max256 = vec(maximum(rel256, dims=(1, 2, 4)))'

            println("$(batches) Batches:")
            println("\t32 bit mean and max % error vs reference:")
            display(mean32)
            display(max32)
            println("\t64 bit mean and max % error vs reference:")
            display(mean64)
            display(max64)
            println("\t256 bit mean and max % error vs reference:")
            display(mean256)
            display(max256)
            println("")
        end
    end




end

@testset "Test SNR and MoMSNR equivalence" begin
    nt = 10000
    ns = 100
    lrange = 256
    ldim = 16

    a = rand(nt, ns)
    l = rand(UInt8, nt, ldim)

    snr1 = [SNR.SNRBasic{Float64, UInt8}(ns, lrange) for _ in 1:ldim]
    snr2 = SNR.SNRMoM{Float64, UInt8, Array}(ns, lrange, ldim)

    for i in 1:ldim
        SNR.SNR_fit!(snr1[i], a, @view(l[:, i]))
    end
    SNR.SNR_fit!(snr2, a, l)

    snr1_results = [SNR.SNR_finalize(snr1[i]) for i in 1:ldim]
    snr2_results = SNR.SNR_finalize(snr2)

    for i in 1:ldim
        @test all(snr2_results[i, :] .≈ snr1_results[i])
    end
end