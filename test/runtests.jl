using SCA
using Statistics
using Random
using CSV, DataFrames
using Test

using StatsBase
Random.seed!(12)



@testset "Univariate moment estimation dataset shape compatibility test" begin
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

    @test all(Moments.centered_sums(m_avec_lvec)[1, :, :, :] .≈ Moments.centered_sums(m_amat_lvec)[1, :, :, 1])
    @test all(Moments.centered_sums(m_avec_lvec)[1, :, :, :] .≈ Moments.centered_sums(m_avec_lmat)[1, :, :, 1])
end

@testset "Univariate moment estimation live datatype conversion test" begin
    aint = rand(Int8, 20000, 20)
    afloat = Float32.(aint)
    l=rand(UInt8, 20000, 1)

    m1 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(2, afloat, l)
    m2 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(2, afloat, l)

    Moments.fit_moments!(m1, aint, l)
    Moments.fit_moments!(m2, afloat, l)

    @test all(Moments.centered_sums(m1) .≈ Moments.centered_sums(m2))
end

@testset "Multivariate moment estimation rework test" begin
    ns = 2  # must = 2 so that covariance can be calculated from the sums of centered products
    order = 1  # must = 1 for same reason
    
    mtest = Moments.MultiVarMomentsAcc{Float64, UInt8, Array}(((1, order), (2, order)), ns, 2, 1)

    a = rand(50000, ns)
    l = UInt8.(rand([0, 1], 50000))

    Moments.fit_moments!(mtest, a, l)

    set_0 = a[l.==0, :]
    cov_0 = cov(set_0) 

    @test all(.≈((Moments.centered_sums(mtest)[1, 1, :] / size(set_0, 1)), cov_0[1, 2], rtol=0.01))
end

@testset "Moment estimation satability test 1: re-ordering error vs. float length (no merging)" begin
    NL = 1
    sizes = [20000, 50000, 100000]
    a32 = Dict(s => rand(Float32, s, 1) for s in sizes)
    a64 = Dict(s => rand(Float64, s, 1) for s in sizes)
    l = Dict(s => zeros(UInt8, s, NL) for s in sizes)


    n = 10
    order = 16
    results = Dict{Tuple{Int, Int}, AbstractArray}()
    
    # perform `n` iterations
    for i in 1:n
        for s in sizes
            # estimate moments
            m32 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(order, size(a32[s], 2), 1, NL)
            m64 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(order, size(a64[s], 2), 1, NL)
            Moments.fit_moments!(m32, a32[s], l[s])
            Moments.fit_moments!(m64, a64[s], l[s])
            
            # store results
            if !(haskey(results, (32, s)))
                results[(32, s)] = Float64.(Moments.centered_sums(m32))
            else
                results[(32, s)] = cat(results[(32, s)], Float64.(Moments.centered_sums(m32)), dims=5)

            end
            if !(haskey(results, (64, s)))
                results[(64, s)] = Float64.(Moments.centered_sums(m64))
            else
                results[(64, s)] = cat(results[(64, s)], Float64.(Moments.centered_sums(m64)), dims=5)
            end

            # shuffle trace rows and labels 
            perm = randperm(size(a32[s], 1))
            a32[s], l[s] = SCA.TestUtils.permute_dataset_rows(a32[s], l[s], perm)
            a64[s], _ = SCA.TestUtils.permute_dataset_rows(a64[s], l[s], perm)
        end
    end

    mean_32_20k = mean(results[(32, 20000)], dims=5)
    mean_64_20k = mean(results[(64, 20000)], dims=5)
    std_32_20k = sqrt.(var(results[(32, 20000)], dims=5))
    std_64_20k = sqrt.(var(results[(64, 20000)], dims=5))
    
    mean_32_50k = mean(results[(32, 50000)], dims=5)
    mean_64_50k = mean(results[(64, 50000)], dims=5)
    std_32_50k = sqrt.(var(results[(32, 50000)], dims=5))
    std_64_50k = sqrt.(var(results[(64, 50000)], dims=5))

    mean_32_100k = mean(results[(32, 100000)], dims=5)
    mean_64_100k = mean(results[(64, 100000)], dims=5)
    std_32_100k = sqrt.(var(results[(32, 100000)], dims=5))
    std_64_100k = sqrt.(var(results[(64, 100000)], dims=5))

    df = DataFrame(
        f32_20k_SD=(vec(mean(std_32_20k, dims=(1, 2, 4)))), 
        f64_20k_SD=(vec(mean(std_64_20k, dims=(1, 2, 4)))), 
        f32_50k_SD=(vec(mean(std_32_50k, dims=(1, 2, 4)))), 
        f64_50k_SD=(vec(mean(std_64_50k, dims=(1, 2, 4)))), 
        f32_100k_SD=(vec(mean(std_32_100k, dims=(1, 2, 4)))), 
        f64_100k_SD=(vec(mean(std_64_100k, dims=(1, 2, 4)))), 
        order=1:16
    )

    CSV.write(joinpath(@__DIR__, "test1_report.csv"), df)
end

@testset "Moment estimation satability test 2: % error vs. float length (no merging)" begin
    setprecision(BigFloat, 256)
    prec = precision(BigFloat)
    println("Ground truth float precision: $(prec)")

    NL = 1
    sizes = [20000, 50000, 100000]
    a32 = Dict(s => rand(Float32, s, 1) for s in sizes)
    a64 = Dict(s => Float64.(a32[s]) for s in sizes)
    a256 = Dict(s => big.(a64[s]) for s in sizes)
    l = Dict(s => zeros(UInt8, s, NL) for s in sizes)

    n = 10
    order = 16
    results = Dict{Tuple{Int, Int}, AbstractArray}()
    resultsref = Dict{Int, AbstractArray}()

    # perform `n` iterations
    for s in sizes
        # calculate ground truth
        m256 = Moments.UniVarMomentsAcc{BigFloat, UInt8, Array}(order, size(a64[s], 2), 1, NL)
        Moments.fit_moments!(m256, a256[s], l[s])
        resultsref[s] = Float64.(Moments.centered_sums(m256))
    
        for i in 1:n
            # estimate moments
            m32 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(order, size(a32[s], 2), 1, NL)
            m64 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(order, size(a64[s], 2), 1, NL)
            Moments.fit_moments!(m32, a32[s], l[s])
            Moments.fit_moments!(m64, a64[s], l[s])
            
            # store results
            if !(haskey(results, (32, s)))
                results[(32, s)] = big.(Moments.centered_sums(m32))
            else
                results[(32, s)] = cat(results[(32, s)], big.(Moments.centered_sums(m32)), dims=5)

            end
            if !(haskey(results, (64, s)))
                results[(64, s)] = big.(Moments.centered_sums(m64))
            else
                results[(64, s)] = cat(results[(64, s)], big.(Moments.centered_sums(m64)), dims=5)
            end

            # shuffle trace rows and labels 
            perm = randperm(size(a32[s], 1))
            a32[s], l[s] = SCA.TestUtils.permute_dataset_rows(a32[s], l[s], perm)
            a64[s], _ = SCA.TestUtils.permute_dataset_rows(a64[s], l[s], perm)
        end
    end

    relerr_mean(prec::Int, size::Int) = vec(mean(abs.(results[(prec, size)] .- resultsref[size]) .* 100, dims=(1, 2, 4, 5)))'
    relerr_max(prec::Int, size::Int) = vec(maximum(abs.(results[(prec, size)] .- resultsref[size]) .* 100, dims=(1, 2, 4, 5)))'

    mean_32_100k = relerr_mean(32, 100000)
    max_32_100k = relerr_max(32, 100000)
    mean_64_100k = relerr_mean(64, 100000)
    max_64_100k = relerr_max(64, 100000)

    mean_32_50k = relerr_mean(32, 50000)
    max_32_50k = relerr_max(32, 50000)
    mean_64_50k = relerr_mean(64, 50000)
    max_64_50k = relerr_max(64, 50000)

    mean_32_20k = relerr_mean(32, 20000)
    max_32_20k = relerr_max(32, 20000)
    mean_64_20k = relerr_mean(64, 20000)
    max_64_20k = relerr_max(64, 20000)

    df = DataFrame(
        f32_relerr_avg_100k=mean_32_100k', 
        f32_relerr_max_100k=max_32_100k',
        f64_relerr_avg_100k=mean_64_100k', 
        f64_relerr_max_100k=max_64_100k',
        f32_relerr_avg_50k=mean_32_50k', 
        f32_relerr_max_50k=max_32_50k',
        f64_relerr_avg_50k=mean_64_50k', 
        f64_relerr_max_50k=max_64_50k', 
        f32_relerr_avg_20k=mean_32_20k', 
        f32_relerr_max_20k=max_32_20k',
        f64_relerr_avg_20k=mean_64_20k', 
        f64_relerr_max_20k=max_64_20k', 
        order=1:16
    )
    CSV.write(joinpath(@__DIR__, "test2_report.csv"), df)
end

@testset "Moment estimation satability test 3: % error vs. float length and number of merges" begin
    setprecision(BigFloat, 256)
    prec = precision(BigFloat)
    println("Ground truth float precision: $(prec)")
    
    NL = 1
    a32 = rand(Float32, 100000, 10)
    a64 = Float64.(a32)
    a256 = big.(a64)
    l = zeros(UInt8, 100000, NL)

    batch_settings = [1, 2, 4, 8, 16]
    order = 16
    
    ref_results = Dict{Int, AbstractArray}()
    inc_results = Dict{Tuple{Int, Int, String}, AbstractArray}()

    println("Dataset parameters: $(size(a32, 1))x$(size(a32, 2)) traces\t$(NL) label length")
    println("Statistic parameters: centered product order = $(order)")
    println("")

    for batches in batch_settings
        # initialize accumulator structs
        m256 = Moments.UniVarMomentsAccIncremental{BigFloat, UInt8, Array}(order, size(a256, 2), 1, NL)
        m64 = Moments.UniVarMomentsAccIncremental{Float64, UInt8, Array}(order, size(a64, 2), 1, NL)
        m32 = Moments.UniVarMomentsAccIncremental{Float32, UInt8, Array}(order, size(a32, 2), 1, NL)
        
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
        results32 = Moments.centered_sums(m32)
        results64 = Moments.centered_sums(m64)
        results256 = Moments.centered_sums(m256)

        # analysis
        if batches == 1
            ref_results[32] = results32
            ref_results[64] = results64
            ref_results[256] = results256
        else
            rel32 = Float64.(abs.((results32 .- ref_results[256])) .* 100)
            rel64 = Float64.(abs.((results64 .- ref_results[256])) .* 100)
            rel256 = Float64.(abs.((results256 .- ref_results[256])) .* 100)

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

            inc_results[(32, batches, "mean")] = mean32'
            inc_results[(32, batches, "max")] = max32'
            inc_results[(64, batches, "mean")] = mean64'
            inc_results[(64, batches, "max")] = max64'
            inc_results[(256, batches, "mean")] = mean256'
            inc_results[(256, batches, "max")] = max256'
        end
    end

    df = DataFrame(
        f32_relerr_avg_2batch = inc_results[(32, 2, "mean")],
        f32_relerr_max_2batch = inc_results[(32, 2, "max")],
        f64_relerr_avg_2batch = inc_results[(64, 2, "mean")],
        f64_relerr_max_2batch = inc_results[(64, 2, "max")],
        f256_relerr_avg_2batch = inc_results[(256, 2, "mean")],
        f256_relerr_max_2batch = inc_results[(256, 2, "max")],
        f32_relerr_avg_4batch = inc_results[(32, 4, "mean")],
        f32_relerr_max_4batch = inc_results[(32, 4, "max")],
        f64_relerr_avg_4batch = inc_results[(64, 4, "mean")],
        f64_relerr_max_4batch = inc_results[(64, 4, "max")],
        f256_relerr_avg_4batch = inc_results[(256, 4, "mean")],
        f256_relerr_max_4batch = inc_results[(256, 4, "max")],
        f32_relerr_avg_8batch = inc_results[(32, 8, "mean")],
        f32_relerr_max_8batch = inc_results[(32, 8, "max")],
        f64_relerr_avg_8batch = inc_results[(64, 8, "mean")],
        f64_relerr_max_8batch = inc_results[(64, 8, "max")],
        f256_relerr_avg_8batch = inc_results[(256, 8, "mean")],
        f256_relerr_max_8batch = inc_results[(256, 8, "max")],
        f32_relerr_avg_16batch = inc_results[(32, 16, "mean")],
        f32_relerr_max_16batch = inc_results[(32, 16, "max")],
        f64_relerr_avg_16batch = inc_results[(64, 16, "mean")],
        f64_relerr_max_16batch = inc_results[(64, 16, "max")],
        f256_relerr_avg_16batch = inc_results[(256, 16, "mean")],
        f256_relerr_max_16batch = inc_results[(256, 16, "max")],
        order=1:16
    )
    CSV.write(joinpath(@__DIR__, "test3_report.csv"), df)
    


end

@testset "Test Naive SNR and MoMSNR equivalence" begin
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

@testset "Test MoMSNR intitialization" begin
    nt = 10000
    ns = 100
    lrange = 256
    ldim = 16

    a = rand(nt, ns)
    l = rand(UInt8, nt, ldim)

    snrmoments = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(2, ns, lrange, ldim)
    snrmomentsinc = Moments.UniVarMomentsAccIncremental{Float64, UInt8, Array}(2, ns, lrange, ldim)
    snr = SNR.SNRMoM{Float64, UInt8, Array}(ns, lrange, ldim)
    snr_from_moments = SNR.SNRMoM{Float64, UInt8, Array}(snrmoments)
    snr_from_moments_inc = SNR.SNRMoM{Float64, UInt8, Array}(snrmomentsinc)

    SNR.SNR_fit!(snr, a, l)
    SNR.SNR_fit!(snr_from_moments, a, l)
    SNR.SNR_fit!(snr_from_moments_inc, a, l)

    snr_results = SNR.SNR_finalize(snr)
    snr_from_moments_results = SNR.SNR_finalize(snr_from_moments)
    snr_from_moments_inc_results = SNR.SNR_finalize(snr_from_moments_inc)

    @test all(snr_results .≈ snr_from_moments_results)
    @test all(snr_results .≈ snr_from_moments_inc_results)
end