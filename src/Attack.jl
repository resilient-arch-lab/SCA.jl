module Attack
export GaussianModel, PCAGaussianModel, fit_model, apply_model

include("Utils.jl")
using .Utils
include("Moments.jl")
using .Moments

using Statistics 
using MultivariateStats
using Distributions

# Classic Gaussian Template Attack stuff

mutable struct GaussianLabelTemplate{Tt<:Real, Tl<:Integer}
    label::Tl  # Template label
    d::MvNormal{Tt}  # Multivariate normal distribution, fit to data with label

    function GaussianLabelTemplate{Tt, Tl}(label::Tl, traces::AbstractMatrix{Tt}) where {Tt<:Real, Tl<:Integer}
        means = vec(mean(traces, dims=1))
        c = cov(traces, dims=1)
        d = MvNormal(means, c)
        new(label, d)
    end
end

mutable struct GaussianModel{Tt<:Real, Tl<:Integer} 
    labels::UnitRange{Tl}
    op_templates::Dict{Tl, GaussianLabelTemplate{Tt, Tl}}

    function GaussianModel{Tt, Tl}(nl::Int) where {Tt<:Real, Tl<:Integer}
        labels = 0:nl-1
        op_templates = Dict{Tl, GaussianLabelTemplate{Tt, Tl}}()
        new(labels, op_templates)
    end
end

# Fit gaussian profile around preprocessed traces
function fit_model(model::GaussianModel{Tt, Tl}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:Real, Tl<:Integer}
    for label in model.labels
        subset = traces[labels.==(label), :]
        model.op_templates[label] = GaussianLabelTemplate{Tt, Tl}(label, subset)
    end
end

function apply_model(model::GaussianModel{Tt, Tl}, trace::AbstractVector{Tt})::Vector{Float64} where {Tt<:Real, Tl<:Integer}
    pdfs::Vector{Float64} = zeros(size(model.labels, 1))
    Threads.@threads for l in model.labels
        pdfs[l+1] = logpdf(model.op_templates[l].d, trace)
    end
    pdfs
end

function apply_model(model::GaussianModel{Tt, Tl}, traces::AbstractMatrix{Tt})::Vector{Float64} where {Tt<:Real, Tl<:Integer}
    pdfs::Vector{Float64} = zeros(size(model.labels, 1))
    Threads.@threads for l in model.labels
        for i in axes(traces, 1)
            pdfs[l+1] += logpdf(model.op_templates[l].d, traces[i, :])
        end
        # pdfs[l+1] ./ size(traces, 1)
    end
    pdfs
end

# Gaussian model in PCA subspace
mutable struct PCAGaussianModel{Tt<:Real, Tl<:Integer}
    gaussian_model::GaussianModel{Tt, Tl}
    labels::UnitRange{Tl}
    moments::UniVarMomentsAcc{Tt, Tl, Array}
    n_PCA_dims::Int
    PCA::Union{PCA, Nothing}

    function PCAGaussianModel{Tt, Tl}(nl::Int, ns::Int, n_PCA_dims::Int=3) where {Tt<:Real, Tl<:Integer}
        gaussian_model = GaussianModel{Tt, Tl}(nl)
        m = UniVarMomentsAcc{Tt, Tl, Array}(2, ns, nl)
        new(gaussian_model, gaussian_model.labels, m, n_PCA_dims, nothing)
    end
end

function fit_model(model::PCAGaussianModel{Tt, Tl}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:Real, Tl<:Integer}
    centered_sum_update!(model.moments, traces, labels)
    means = model.moments.moments[:, 1, :]
    model.PCA = fit(PCA, means', maxoutdim=model.n_PCA_dims)
    t_fit = predict(model.PCA, traces')'
    fit_model(model.gaussian_model, t_fit, labels)
end

function apply_model(model::PCAGaussianModel{Tt, Tl}, traces::AbstractVecOrMat{Tt})::Vector{Float64} where {Tt<:Real, Tl<:Integer}
    t = predict(model.PCA, traces')'
    apply_model(model.gaussian_model, t)
end

end