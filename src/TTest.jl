module TTest
export ttest_fit!, ttest_finalize, TTestMoM, TTestChunked

include("Utils.jl")
using .Utils
include("Moments.jl")
using .Moments

abstract type AbstractTTest end
abstract type AbstractMoMTTest <: AbstractTTest end

struct TTestMoM{Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray} <: AbstractMoMTTest
    moments::Moments.UniVarMomentsAcc{Tt, Tl, Tarray}
    order::Int
    ns::Int

    function TTestMoM{Tt, Tl, Tarray}(order::Int, ns::Int, ldim::Int) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
        moments = Moments.UniVarMomentsAcc{Tt, Tl, Tarray}(2*order, ns, 2, ldim)
        new(moments, order, ns)
    end
end

function ttest_fit!(ttest::AbstractMoMTTest, traces, labels)
    fit_moments!(ttest.moments, traces, labels)
end

# DOES NOT WORK ON GPU (because of scalar indexing in the last line)
function ttest_finalize(ttest::AbstractMoMTTest)::AbstractArray
    μ, σ = get_mean_and_var(ttest.moments, ttest.order)
    μ1, μ2 = view(μ, 1, 1, :), view(μ, 1, 2, :) 
    σ1, σ2 = view(σ, 1, 1, :), view(σ, 1, 2, :)
    t = (μ1 - μ2) ./ sqrt.((σ1 ./ ttest.moments.totals[1]) .+ (σ2 ./ ttest.moments.totals[2]))
end

end  # module TTest