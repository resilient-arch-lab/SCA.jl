module Benchmarks
export bench_SNRMoments

using SCA
using BenchmarkTools
using KernelAbstractions
using CSV, DataFrames
using StatsBase

# The BenchmarkGroup seems to be required to be declared globally
bench_suite = BenchmarkGroup()

function bench_SNRMoments(TArray::Type = Array)
    t = TArray(rand(Float32, 300000, 1000))
    l = TArray(rand(UInt8, 300000))

    snr1 = SNR.SNRMoments{Float32, UInt8, TArray}(size(t, 2), 256)
    snr2 = SNR.SNRMomentsChunked{Float32, UInt8, TArray}(size(t, 2), 256, (size(t, 1), 500))
    snr3 = SNR.SNRMomentsChunked{Float32, UInt8, TArray}(size(t, 2), 256, (size(t, 1), 200))
    snr4 = SNR.SNRMomentsChunked{Float32, UInt8, TArray}(size(t, 2), 256, (100000, 500))
    snr5 = SNR.SNRMomentsChunked{Float32, UInt8, TArray}(size(t, 2), 256, (100000, 1000))
    snr6 = SNR.SNRMomentsChunked{Float32, UInt8, TArray}(size(t, 2), 256, (50000, 250))

    bench_suite["SNR"]["Fit"]["Single"] = @benchmarkable SNR.SNR_fit!($snr1, $t, $l)
    bench_suite["SNR"]["Fit"]["Chunked[all, 500]"] = @benchmarkable SNR.SNR_fit!($snr2, $t, $l)
    bench_suite["SNR"]["Fit"]["Chunked[all, 200]"] = @benchmarkable SNR.SNR_fit!($snr3, $t, $l)
    bench_suite["SNR"]["Fit"]["Chunked[100k, 500]"] = @benchmarkable SNR.SNR_fit!($snr4, $t, $l)
    bench_suite["SNR"]["Fit"]["Chunked[100k, all]"] = @benchmarkable SNR.SNR_fit!($snr5, $t, $l)
    bench_suite["SNR"]["Fit"]["Chunked[50k, 250]"] = @benchmarkable SNR.SNR_fit!($snr6, $t, $l)

    bench_suite["SNR"]["Finalize"]["Single"] = @benchmarkable SNR.SNR_finalize($snr1)
    bench_suite["SNR"]["Finalize"]["Chunked[all, 500]"] = @benchmarkable SNR.SNR_finalize($snr2)
    bench_suite["SNR"]["Finalize"]["Chunked[all, 200]"] = @benchmarkable SNR.SNR_finalize($snr3)
    bench_suite["SNR"]["Finalize"]["Chunked[100k, 500]"] = @benchmarkable SNR.SNR_finalize($snr4)
    bench_suite["SNR"]["Finalize"]["Chunked[100k, all]"] = @benchmarkable SNR.SNR_finalize($snr5)
    bench_suite["SNR"]["Finalize"]["Chunked[50k, 250]"] = @benchmarkable SNR.SNR_finalize($snr6)

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 2)
end

# AcceleratedKernels code is faster on CPU and GPU
function bench_Moments_label_wise_sum(TArray::Type = Array)
    t = TArray(rand(Float32, 500000, 1000))
    l = TArray(rand(UInt8, 500000))
    sums = TArray(zeros(Float32, 256, 1000))
    totals = TArray(zeros(UInt32, 256))

    _label_wise_sum_kern = Moments.label_wise_sum_shared!(get_backend(t), (4, 64))
    bench_suite["Label Wise Sum"]["Kernal Abstractions"] = @benchmarkable $_label_wise_sum_kern($t, $l, $sums, $totals, ndrange=size($t))
    bench_suite["Label Wise Sum"]["Accelerated Kernels"] = @benchmarkable Moments.label_wise_sum_ak!($t, $l, $sums, $totals)

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 10)
end

# AcceleratedKernels code is faster on CPU and GPU
function bench_Moments_centered_sum_update(TArray::Type = Array)
    t = TArray(rand(Float32, 300000, 1000))
    l = TArray(rand(UInt8, 300000))
    m1 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(10, 1000, 256)
    m2 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(10, 1000, 256)
    
    Moments.centered_sum_update_old!(m1, t, l)
    Moments.centered_sum_update!(m2, t, l)
    bench_suite["Centered Sum Update (legacy)"] = @benchmarkable Moments.centered_sum_update_old!($m1, $t, $l)
    bench_suite["Centered Sum Update"] = @benchmarkable Moments.centered_sum_update!($m2, $t, $l)

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 5)
end

function bench_Moments_centered_sum_update_vs_cpu(TArray::Type = Array)
    t = TArray(rand(Float32, 300000, 1000))
    l = TArray(rand(UInt8, 300000))
    m1 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(10, 1000, 256)
    m2 = Moments.UniVarMomentsAcc{Float32, UInt8, TArray}(10, 1000, 256)
    
    Moments.centered_sum_update!(m2, t, l)
    Moments.centered_sum_update!(m1, t, l)
    Moments.centered_sum_update!(m2, t, l)
    Moments.centered_sum_update!(m1, t, l)
    bench_suite["Centered Sum Update, TArray ($(TArray)) Device"] = @benchmarkable Moments.centered_sum_update!($m2, $t, $l)
    bench_suite["Centered Sum Update, Array Device"] = @benchmarkable Moments.centered_sum_update!($m1, $t, $l)
    # bench_suite["Centered Sum Update 2"] = @benchmarkable Moments.centered_sum_update2!($m2, $t, $l)

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 5)
end

function bench_Moments_centered_sum_update_vs_combined(TArray::Type = Array)
    t = TArray(rand(Float32, 300000, 1000))
    l = TArray(rand(UInt8, 300000))
    m1 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(10, 1000, 256)
    m2 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(10, 1000, 256)
    
    Moments.centered_sum_update_combined!(m2, t, l)
    Moments.centered_sum_update!(m1, t, l)
    Moments.centered_sum_update_combined!(m2, t, l)
    Moments.centered_sum_update!(m1, t, l)
    bench_suite["Combined Centered Sum Update"] = @benchmarkable Moments.centered_sum_update!($m2, $t, $l)
    bench_suite["Centered Sum Update"] = @benchmarkable Moments.centered_sum_update!($m1, $t, $l)
    # bench_suite["Centered Sum Update 2"] = @benchmarkable Moments.centered_sum_update2!($m2, $t, $l)

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 5)
end

function bench_Moments_VecLabel(TArray::Type = Array, order::Int = 2, ns::Int = 1000, nt::Int = 100000)
    t = TArray(rand(Float32, nt, ns))
    l = TArray(rand(UInt8, nt, 16))
    m1 = Moments.UniVarMomentsAcc{Float32, UInt8, TArray}(order, ns, 256)
    # m2 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, TArray, 1}(order, ns, 256)  # not working
    m3 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, TArray, 4}(order, ns, 256)
    m4 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, TArray, 8}(order, ns, 256)
    m5 = Moments.UniVarMomentsAccVecLabel{Float32, UInt8, TArray, 16}(order, ns, 256)
    mlist = [Moments.UniVarMomentsAcc{Float32, UInt8, TArray}(order, ns, 256) for _ in 1:8]

    bench_suite["Centered Sum Update, 1 scalar label"] = @benchmarkable Moments.centered_sum_update!($m1, $t, $l[:, 1])
    # bench_suite["Centered Sum Update, vec 1 label"] = @benchmarkable Moments.centered_sum_update!($m2, $t, $(l[:, 1]))
    bench_suite["Centered Sum Update, vec 4 label"] = @benchmarkable Moments.centered_sum_update!($m3, $t, $(l[:, 1:4]))  # [106μs on RX9070XT]
    bench_suite["Centered Sum Update, vec 8 label"] = @benchmarkable Moments.centered_sum_update!($m4, $t, $(l[:, 1:8]))  
    bench_suite["Centered Sum Update, vec 16 label"] = @benchmarkable Moments.centered_sum_update!($m5, $t, $l) 
    bench_suite["Centered Sum Update, 8 scalar label"] = @benchmarkable Moments.centered_sum_update!.($mlist, $[view(t, :, :) for _ in 1:8], $[l[:, i] for i in 1:8])

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite; verbose = true, seconds = 5)
end

function bench_Moments_VecLabel_scaling(orders::Vector{Int}, Ns::Vector{Int}, Nt::Vector{Int}, TArray::Type = Array)
    results = DataFrame(
        "device" => "CPU", 
        "order" => 2, 
        "# samples" => 0, 
        "# traces" => 0,
        "type" => "...",
        "time" => 0.0, 
        "memory" => 0
    )

    if TArray == Array
        device = "CPU"
    end

    for (order, ns, nt) in Iterators.product(orders, Ns, Nt)
        println("Benching order=$(order), ns=$(ns), nt=$(nt)")
        bench_results = bench_Moments_VecLabel(TArray, order, ns, nt)
        
        push!(results, [device, order, ns, nt, "1 scalar label", StatsBase.mean(bench_results["Centered Sum Update, 1 scalar label"].times)*1e-9, bench_results["Centered Sum Update, 1 scalar label"].memory]; promote=true)
        push!(results, [device, order, ns, nt, "8 scalar label", StatsBase.mean(bench_results["Centered Sum Update, 8 scalar label"].times)*1e-9, bench_results["Centered Sum Update, 8 scalar label"].memory]; promote=true)
        # push!(results, [device, order, ns, nt, "vec 1 label", StatsBase.mean(bench_results["Centered Sum Update, vec 1 label"].times)*1e-9, bench_results["Centered Sum Update, vec 1 label"].memory]; promote=true)
        push!(results, [device, order, ns, nt, "vec 4 label", StatsBase.mean(bench_results["Centered Sum Update, vec 4 label"].times)*1e-9, bench_results["Centered Sum Update, vec 4 label"].memory]; promote=true)
        push!(results, [device, order, ns, nt, "vec 8 label", StatsBase.mean(bench_results["Centered Sum Update, vec 8 label"].times)*1e-9, bench_results["Centered Sum Update, vec 8 label"].memory]; promote=true)
        push!(results, [device, order, ns, nt, "vec 16 label", StatsBase.mean(bench_results["Centered Sum Update, vec 16 label"].times)*1e-9, bench_results["Centered Sum Update, vec 16 label"].memory]; promote=true)
    end

    CSV.write("bench/results/result-VecLabel.csv", results)
end

# minimal memory overhead
function bench_chunked_memory_use()
    t = rand(Float32, 500000, 1000)
    l = rand(UInt8, 500000)
    dset_size = (sizeof(eltype(t)) * prod(size(t))) + (sizeof(eltype(l)) * prod(size(l)))
    println("dataset size: $(dset_size * 1e-6) Mb")
    
    snr1 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (50000, 1000))
    snr2 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (50000, 500))
    snr3 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (50000, 250))
    snr4 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (25000, 1000))
    snr5 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (25000, 500))
    snr6 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (25000, 250))

    for (i, snr) in enumerate((snr1, snr2, snr3, snr4, snr5, snr6))
        allocated = @allocated SNR.SNR_fit!(snr1, t, l)
        println("SNR$i (chunksize: $(snr.chunksize)) allocated: $(allocated * 1e-6) Mb")
    end
end

end  # module Benchmarks