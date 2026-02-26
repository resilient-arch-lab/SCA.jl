module Benchmarks
export bench_SNRMoments

using SCA
using BenchmarkTools

# The BenchmarkGroup seems to be required to be declared globally
bench_suite = BenchmarkGroup(["SNR", ])

function bench_SNRMoments(TArray::Type = Array)
    t = TArray(rand(Float32, 500000, 1000))
    l = TArray(rand(UInt8, 500000))

    bench_suite["SNR"]["CPU"]["Fit"] = BenchmarkGroup(["Single", "Chunked[all, 500]", "Chunked[all, 200]", "Chunked[100k, all]", "Chunked[100k, 500]"])
    bench_suite["SNR"]["CPU"]["Finalize"] = BenchmarkGroup(["Single", "Chunked[all, 500]", "Chunked[all, 200]", "Chunked[100k, all]", "Chunked[100k, 500]"])

    snr1 = SNR.SNRMoments{Float32, UInt8}(size(t, 2), 256)
    snr2 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, 500)
    snr3 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, 200)
    snr4 = SNR.SNRMomentsChunkedMulti{Float32, UInt8}(size(t, 2), 256, (100000, 500))
    snr5 = SNR.SNRMomentsChunkedMulti{Float32, UInt8}(size(t, 2), 256, (100000, 1000))
    snr6 = SNR.SNRMomentsChunkedMulti{Float32, UInt8}(size(t, 2), 256, (50000, 250))

    bench_suite["SNR"]["CPU"]["Fit"]["Single"] = @benchmarkable SNR.SNR_fit!($snr1, $t, $l)
    bench_suite["SNR"]["CPU"]["Fit"]["Chunked[all, 500]"] = @benchmarkable SNR.SNR_fit!($snr2, $t, $l)
    bench_suite["SNR"]["CPU"]["Fit"]["Chunked[all, 200]"] = @benchmarkable SNR.SNR_fit!($snr3, $t, $l)
    bench_suite["SNR"]["CPU"]["Fit"]["Chunked[100k, 500]"] = @benchmarkable SNR.SNR_fit!($snr4, $t, $l)
    bench_suite["SNR"]["CPU"]["Fit"]["Chunked[100k, all]"] = @benchmarkable SNR.SNR_fit!($snr5, $t, $l)
    bench_suite["SNR"]["CPU"]["Fit"]["Chunked[50k, 250]"] = @benchmarkable SNR.SNR_fit!($snr6, $t, $l)

    bench_suite["SNR"]["CPU"]["Finalize"]["Single"] = @benchmarkable SNR.SNR_finalize($snr1)
    bench_suite["SNR"]["CPU"]["Finalize"]["Chunked[all, 500]"] = @benchmarkable SNR.SNR_finalize($snr2)
    bench_suite["SNR"]["CPU"]["Finalize"]["Chunked[all, 200]"] = @benchmarkable SNR.SNR_finalize($snr3)
    bench_suite["SNR"]["CPU"]["Finalize"]["Chunked[100k, 500]"] = @benchmarkable SNR.SNR_finalize($snr4)
    bench_suite["SNR"]["CPU"]["Finalize"]["Chunked[100k, all]"] = @benchmarkable SNR.SNR_finalize($snr5)
    bench_suite["SNR"]["CPU"]["Finalize"]["Chunked[50k, 250]"] = @benchmarkable SNR.SNR_finalize($snr6)

    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 10)
end

end  # module Benchmarks