module Benchmarks
export bench_SNRMoments

using SCA
using BenchmarkTools
using KernelAbstractions

# The BenchmarkGroup seems to be required to be declared globally
bench_suite = BenchmarkGroup()

function bench_SNRMoments(TArray::Type = Array)
    t = TArray(rand(Float32, 300000, 1000))
    l = TArray(rand(UInt8, 300000))

    snr1 = SNR.SNRMoments{Float32, UInt8}(size(t, 2), 256)
    snr2 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (size(t, 2), 500))
    snr3 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (size(t, 2), 200))
    snr4 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (100000, 500))
    snr5 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (100000, 1000))
    snr6 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, (50000, 250))

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
    run(bench_suite, verbose = true, seconds = 5)
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
    
    Moments.centered_sum_update!(m1, t, l)
    Moments.centered_sum_update2!(m2, t, l)
    bench_suite["Centered Sum Update"] = @benchmarkable Moments.centered_sum_update!($m1, $t, $l)
    bench_suite["Centered Sum Update 2"] = @benchmarkable Moments.centered_sum_update2!($m2, $t, $l)

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 5)
end

function bench_Moments_centered_sum_update_vs_cpu(TArray::Type = Array)
    t = TArray(rand(Float32, 300000, 1000))
    l = TArray(rand(UInt8, 300000))
    m1 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(10, 1000, 256)
    m2 = Moments.UniVarMomentsAcc{Float32, UInt8, TArray}(10, 1000, 256)
    
    Moments.centered_sum_update!(m1, t, l)
    Moments.centered_sum_update2!(m2, t, l)
    bench_suite["Centered Sum Update, Array Device"] = @benchmarkable Moments.centered_sum_update!($m1, $t, $l)
    bench_suite["Centered Sum Update, TArray ($(TArray)) Device"] = @benchmarkable Moments.centered_sum_update!($m2, $t, $l)
    # bench_suite["Centered Sum Update 2"] = @benchmarkable Moments.centered_sum_update2!($m2, $t, $l)

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 5)
end

function bench_Moments_NDLabel(TArray::Type = Array)
    t = TArray(rand(Float32, 100000, 1000))
    l = TArray(rand(UInt8, 100000, 16))
    m1 = Moments.UniVarMomentsAcc{Float32, UInt8, Array}(10, 1000, 256)
    m2 = Moments.UniVarMomentsAccNDLabel{Float32, UInt8, Array, 1}(10, 1000, 256, (4, ))
    mlist = [Moments.UniVarMomentsAcc{Float32, UInt8, Array}(10, 1000, 256) for _ in 1:4]

    bench_suite["Centered Sum Update, scalar label"] = @benchmarkable Moments.centered_sum_update!($m1, $t, $l[:, 1])
    bench_suite["Centered Sum Update, vec 4 label"] = @benchmarkable Moments.centered_sum_update!($m2, $t, $l[:, 1:4])
    bench_suite["Centered Sum Update, multiple scalar label"] = @benchmarkable Moments.centered_sum_update!.($mlist, $[view(t, :, :) for _ in 1:4], $[l[:, i] for i in 1:4])

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 5)
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