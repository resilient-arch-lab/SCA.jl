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

function test_NDLabel_moments()
    a = rand(20000, 20)
    l = rand(UInt8, 20000, 4)
    m1 = Moments.UniVarMomentsAcc{Float64, UInt8, Array}(10, 20, 256)
    m2 = Moments.UniVarMomentsAccNDLabel{Float64, UInt8, Array, 1}(10, 20, 256, (4, ))
    test_l = 3

    sums1 = fill!(similar(a, eltype(a), m1.nl, m1.ns), 0)
    moments1 = fill!(similar(a, eltype(a), size(m1.moments)), 0)
    totals1 = fill!(similar(a, UInt32, size(m1.totals)), 0)
    sums2 = fill!(similar(a, eltype(a), size(l, 2), m2.nl, m2.ns), 0)
    moments2 = fill!(similar(a, eltype(a), size(m2.moments)), 0)
    moments3 = fill!(similar(a, eltype(a), size(m2.moments)), 0)
    totals2 = fill!(similar(a, UInt32, size(m2.totals)), 0)

    Moments.label_wise_sum_ak!(a[10001:end, :], l[10001:end, test_l], sums1, totals1)
    Moments.label_wise_sum_ak!(a[10001:end, :], l[10001:end, :], sums2, totals2)
    if !(all(isapprox.(sums1, sums2[test_l, :, :]; rtol=1)) && all(totals1 .== totals2[test_l, :]))
        println("label wise sum error")
    end

    # find means
    @. moments1[:, 1, :] = sums1 / totals1
    @. moments2[:, :, 1, :] = sums2 / totals2
    moments3[:, :, 1, :] .= moments2[:, :, 1, :] 
    if !all(moments1[:, 1, :] .≈ moments2[test_l, :, 1, :])
        println("means error")
    end

    # compute centered sums
    Moments.centered_sum_kern_ak!(moments1, a[10001:end, :], l[10001:end, test_l])
    Moments.centered_sum_kern_ak!(moments2, a[10001:end, :], l[10001:end, :])
    @sync for li in axes(l, 2)
        @async begin
            Moments.centered_sum_kern_ak!(view(moments3, li, :, :, :), a[10001:end, :], l[10001:end, li])
        end
    end
    if !all(moments1[:, :, :] .≈ moments2[test_l, :, :, :])
        println("centered sum error 1")
    end
    if !all(moments1[:, :, :] .≈ moments3[test_l, :, :, :])
        println("centered sum error 2")
    end
    if !all(moments2[:, :, :, :] .≈ moments3[:, :, :, :])
        println("centered sum error 3")
    end
    
    # merge centered sum estimations
    Moments.merge_from!(m1, moments1, totals1)
    init_ls = m2.totals .== 0
    update_ls = m2.totals .!= 0
    if any(init_ls)
        m2.moments[init_ls, :, :] .= moments2[init_ls, :, :]  # scalar indexing
        m2.totals[init_ls] .= totals2[init_ls]
    end
    if any(update_ls)
        for l in Array(findall(update_ls))  # cast labels-to-update to CPU mem for kernel execution loop
            merge_from_ak_gpu!(view(m2.moments, l, :, :), view(m2.totals, l), view(moments2, l, :, :), view(totals2, l))
        end
        m2.totals[update_ls] .+= totals2[update_ls]
    end

    correct = all(isapprox.(m1.moments, m2.moments[test_l, :, :, :]; rtol=1e-2))
    println("Correct: $correct")
    prcnt_err = abs.((m2.moments[test_l, :, :, :] .- m1.moments) ./ m1.moments).*100

    println("Moment merging algorithm test case percent error per order $(1:m1.order)")
    display(vec(mean(prcnt_err, dims=(1, 3))))
    # prcnt_err
end

function test_GPU_Template_Attack(TArray::Type = Array)
    t = TArray(rand(20000, 20))
    l = TArray(rand(UInt8, 20000, 4))
    PCA_dims = 3

    model_d0 = Attack.PCAGaussianModel{Float64, UInt8}(256, size(t, 2), PCA_dims)
    Attack.fit_model(model_d0, t, l[:, 1])
end