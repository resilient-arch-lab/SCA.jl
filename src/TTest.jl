module TTest
export TTest_fit!, TTest_finalize, TTestMoM, TTestChunked

import ..Utils as Utils
import ..Moments as Moments

abstract type AbstractTTest end
abstract type AbstractMoMTTest <: AbstractTTest end

struct TTestMoM{Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray} <: AbstractMoMTTest
    moments::Moments.AbstractUnivariateMomentsAcc{Tt, Tl, Ta}
    order::Int
end

function TTestMoM{Tt, Tl, Ta}(order::Int, ns::Int, ldim::Int) where {Tt<:AbstractFloat, Tl<:Integer, Ta<:AbstractArray}
    moments = Moments.UniVarMomentsAcc{Tt, Tl, Ta}(2*order, ns, 2, ldim)
    TTestMoM{Tt, Tl, Ta}(moments, order)
end


function TTest_fit!(ttest::AbstractMoMTTest, traces, labels)
    Moments.fit_moments!(ttest.moments, traces, labels)
end

# DOES NOT WORK ON GPU (because of scalar indexing in the last line)
function TTest_finalize(ttest::AbstractMoMTTest)::AbstractArray
    μ, σ = Moments.get_mean_and_var(ttest.moments, ttest.order)
    μ1, μ2 = view(μ, 1, 1, :), view(μ, 1, 2, :)
    σ1, σ2 = view(σ, 1, 1, :), view(σ, 1, 2, :)
    t = (μ1 - μ2) ./ sqrt.((σ1 ./ Moments.totals(ttest.moments)[1]) .+ (σ2 ./ Moments.totals(ttest.moments)[2]))
    return t
end

end  # module TTest