using SCA
using Statistics

function test_moment_merging_kernel()
    a = rand(10000, 20)
    l = rand(UInt8, 10000)
    m1 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)
    m2 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)

    # First update does not use merging algorithm
    Moments.centered_sum_update!(m1, a[1:5000, :], l[1:5000])
    # Moments.centered_sum_update!(m1, a[5001:end, :], l[5001:end])
    Moments.centered_sum_update2!(m2, a[1:5000, :], l[1:5000])
    # Moments.centered_sum_update2!(m2, a[5001:end, :], l[5001:end])

    sums1 = fill!(similar(a, eltype(a), m2.nl, m2.ns), 0)
    moments1 = fill!(similar(a, eltype(a), size(m2.moments)), 0)
    totals1 = fill!(similar(a, UInt32, size(m2.totals)), 0)
    sums2 = fill!(similar(a, eltype(a), m2.nl, m2.ns), 0)
    moments2 = fill!(similar(a, eltype(a), size(m2.moments)), 0)
    totals2 = fill!(similar(a, UInt32, size(m2.totals)), 0)

    Moments.label_wise_sum_ak!(a[5001:end, :], l[5001:end], sums1, totals1)
    Moments.label_wise_sum_ak!(a[5001:end, :], l[5001:end], sums2, totals2)
    if !(all(isapprox.(sums1, sums2; rtol=1)) && all(totals1 .== totals2))
        println("label wise sum error")
    end

    # find means
    @. moments1[:, 1, :] = sums1 / totals1
    @. moments2[:, 1, :] = sums2 / totals2

    # compute centered sums
    Moments.centered_sum_kern_ak!(moments1, a[5001:end, :], l[5001:end])
    Moments.centered_sum_kern_ak!(moments2, a[5001:end, :], l[5001:end])

    # merge centered sum estimations
    Moments.merge_from!(m1, moments1, totals1)
    for l in axes(moments2, 1)
        Moments.merge_from_kern!(view(m2.moments, l, :, :), view(m2.totals, l), view(moments2, l, :, :), view(totals2, l))
    end

    correct = all(isapprox.(m1.moments, m2.moments; rtol=1e-2))
    println("Correct: $correct")
    prcnt_err = abs.((m2.moments .- m1.moments) ./ m1.moments).*100

    println("Moment merging algorithm test case percent error per order $(1:m1.order)")
    display(vec(mean(prcnt_err, dims=(1, 3))))
    # prcnt_err
end

test_moment_merging_kernel()