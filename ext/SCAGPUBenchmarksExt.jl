module SCAGPUBenchmarksExt
export SNRMoments_GPU

using SCA: SCAGPUArraysExt
using BenchmarkTools
using KernelAbstractions

bench_suite = BenchmarkGroup(["SNR", ])

# Get backend
GPUArrayType = nothing
try
    using CUDA
    if CUDA.functional()
        GPUArrayType = CuArray
    end
catch e
end
try
    using AMDGPU
    if AMDGPU.functional()
        GPUArrayType = ROCArray
    end
catch e
end
if isnothing(GPUArrayType)
    throw(ErrorException("No supported GPU backend found!"))
end



t = GPUArrayType(rand(Float32, 500000, 1000))
l = GPUArrayType(rand(UInt8, 500000))

bench_suite["SNR"]["GPU"]["Fit"] = BenchmarkGroup(["Single", "Chunked[all, 500]", "Chunked[all, 200]", "Chunked[100k, all]", "Chunked[100k, 500]"])
bench_suite["SNR"]["GPU"]["Finalize"] = BenchmarkGroup(["Single", "Chunked[all, 500]", "Chunked[all, 200]", "Chunked[100k, all]", "Chunked[100k, 500]"])

snr1 = SNR.SNRMoments{Float32, UInt8}(size(t, 2), 256)
snr2 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, 500)
snr3 = SNR.SNRMomentsChunked{Float32, UInt8}(size(t, 2), 256, 200)
snr4 = SNR.SNRMomentsChunkedMulti{Float32, UInt8}(size(t, 2), 256, (100000, 500))
snr5 = SNR.SNRMomentsChunkedMulti{Float32, UInt8}(size(t, 2), 256, (100000, 1000))

bench_suite["SNR"]["GPU"]["Fit"]["Single"] = @benchmarkable SNR.SNR_fit!(snr1, t, l)
bench_suite["SNR"]["GPU"]["Fit"]["Chunked[all, 500]"] = @benchmarkable SNR.SNR_fit!(snr2, t, l)
bench_suite["SNR"]["GPU"]["Fit"]["Chunked[all, 200]"] = @benchmarkable SNR.SNR_fit!(snr3, t, l)
bench_suite["SNR"]["GPU"]["Fit"]["Chunked[100k, 500]"] = @benchmarkable SNR.SNR_fit!(snr4, t, l)
bench_suite["SNR"]["GPU"]["Fit"]["Chunked[100k, all]"] = @benchmarkable SNR.SNR_fit!(snr5, t, l)

bench_suite["SNR"]["GPU"]["Finalize"]["Single"] = @benchmarkable SNR.SNR_finalize(snr1)
bench_suite["SNR"]["GPU"]["Finalize"]["Chunked[all, 500]"] = @benchmarkable SNR.SNR_finalize(snr2)
bench_suite["SNR"]["GPU"]["Finalize"]["Chunked[all, 200]"] = @benchmarkable SNR.SNR_finalize(snr3)
bench_suite["SNR"]["GPU"]["Finalize"]["Chunked[100k, 500]"] = @benchmarkable SNR.SNR_finalize(snr4)
bench_suite["SNR"]["GPU"]["Finalize"]["Chunked[100k, all]"] = @benchmarkable SNR.SNR_finalize(snr5)

function SNRMoments_GPU()
    println("Tuning benchmark parameters")
    tune!(bench_suite)
    run(bench_suite, verbose = true, seconds = 10)
end

end