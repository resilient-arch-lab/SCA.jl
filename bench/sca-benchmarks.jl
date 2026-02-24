module SCABenchmarks
export test_SNRMoments_CPU

using SCA
using BenchmarkTools


bench_suite = BenchmarkGroup(["SNR", ])

# TODO: benchmark SNR and Chunked SNR on CPU and GPU
t = rand(Float32, 500000, 1000)
l = rand(UInt8, 500000)

bench_suite["SNR"]["CPU"]["Fit"] = BenchmarkGroup(["Single", "Chunked_10k", "Chunked_50k"])
bench_suite["SNR"]["CPU"]["Finalize"] = BenchmarkGroup(["Single", "Chunked_10k", "Chunked_50k"])

snr1 = SNR.SNRMoments{Float32, UInt8}(size(t, 2), 256)
snr2 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, 10000)
snr3 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, 50000)

bench_suite["SNR"]["CPU"]["Fit"]["Single"] = @benchmarkable SNR.SNR_fit!(snr1, t, l)
bench_suite["SNR"]["CPU"]["Fit"]["Chunked_10k"] = @benchmarkable SNR.SNR_fit!(snr2, t, l)
bench_suite["SNR"]["CPU"]["Fit"]["Chunked_50k"] = @benchmarkable SNR.SNR_fit!(snr3, t, l)

bench_suite["SNR"]["CPU"]["Finalize"]["Single"] = @benchmarkable SNR.SNR_finalize(snr1)
bench_suite["SNR"]["CPU"]["Finalize"]["Chunked_10k"] = @benchmarkable SNR.SNR_finalize(snr2)
bench_suite["SNR"]["CPU"]["Finalize"]["Chunked_50k"] = @benchmarkable SNR.SNR_finalize(snr3)

function test_SNRMoments_CPU()
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 10)
end

end  # module SCABenchmarks