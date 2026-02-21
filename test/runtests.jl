using SCA
using Test

@testset "SCA.jl" begin
    # Write your tests here.
end

@testset "Test SNR chunked implementation" begin

    t = rand(Float64, 2000, 1000)
    l = rand(UInt8, 2000)

    snr1 = SNR.SNRMoments{Float64, UInt8}(1000, 256)
    snr2 = SNR.SNRMomentsChunked{Float64, UInt8}(1000, 256, 200)

    SNR.SNR_fit!(snr1, t, l)
    SNR.SNR_fit!(snr2, t, l)

    res1 = SNR.SNR_finalize(snr1)
    res2 = SNR.SNR_finalize(snr2)

    @test all(res1 .≈ res2)

end