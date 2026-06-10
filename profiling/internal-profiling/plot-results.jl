using DataFrames, CSV
using StatsPlots
using ColorSchemes, Colors
using CairoMakie
using Format

function plot_threadripper_results()
    results_df = CSV.read("profiling/internal-profiling/threadripper-results.csv", DataFrame)

    #Plot1:
    #   y = time
    #   x = NTraces * NSamples
    #   log10 scale axes

    @df results_df scatter(
        :NTraces .* :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)), yticks=10.0.^(collect(-3:2)),
        xlims=(1e5, 1e10), ylims=(1e-3, 1e3),
        xlabel="Samples in dataset", ylabel="Time (S)",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: 64-t Threadripper",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/threadripper-time-scatter.png")

    results_df = CSV.read("profiling/internal-profiling/threadripper-results-2.csv", DataFrame)
    @df results_df scatter(
        :NTraces .* :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)), yticks=10.0.^(collect(-3:2)),
        xlims=(1e5, 1e10), ylims=(1e-3, 1e3),
        xlabel="Samples in dataset", ylabel="Time (S)",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: 64-t Threadripper",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/threadripper-time-scatter-2.png")

end

function plot_A5000_results()
    results_df = CSV.read("profiling/internal-profiling/A5000-atomic-results.csv", DataFrame)
    
    #Plot1:
    #   y = time
    #   x = NTraces * NSamples
    #   log10 scale axes

    @df results_df scatter(
        :NTraces .* :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)), yticks=10.0.^(collect(-3:2)),
        xlims=(1e5, 1e10), ylims=(1e-3, 1e3),
        xlabel="Samples in dataset", ylabel="Time (S)",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-atomic-time-scatter.png")
    @df results_df scatter(
        :NTraces, :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, zscale=:log10,
        xlabel="# Traces", ylabel = "# Samples / Trace", zlabel="Time (S)",
        xlims=(1e4, 1e5), ylims=(100, 1000), zlims=(1e-3, 1e3),
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (atomic implementation)",
        dpi=300, size=(800, 600)
    )
    savefig("profiling/internal-profiling/A5000-atomic-time-scatter-3D.png")

    results_df = CSV.read("profiling/internal-profiling/A5000-non-atomic-results.csv", DataFrame)
    @df results_df scatter(
        :NTraces .* :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)), yticks=10.0.^(collect(-3:2)),
        xlims=(1e5, 1e10), ylims=(1e-3, 1e3),
        xlabel="Samples in dataset", ylabel="Time (S)",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (non-atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-non-atomic-time-scatter.png")
    
    results_df = CSV.read("profiling/internal-profiling/A5000-non-atomic-results-3.csv", DataFrame)
    @df results_df scatter(
        :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/32 .* :Order).^(1/4), 0, 1.0 .* (1/256 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10,
        xlims=(90, 2100), ylims=(1e-3, 1e3),
        xlabel="Samples per trace", ylabel="Time (S) [50k traces]",
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (non-atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-non-atomic-time-scatter-3.png")
    
    @df results_df scatter(
        :NTraces, :NSamples, :Time, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/256 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, zscale=:log10,
        xlabel="# Traces", ylabel = "# Samples / Trace", zlabel="Time (S)",
        xlims=(1e4, 1e5), ylims=(100, 1000), zlims=(1e-3, 1e3),
        legend=:outertopright,
        title="Moment Estimation Benchmarks: NVIDIA A5000 (non-atomic implementation)",
        dpi=300, size=(800, 600)
    )
    savefig("profiling/internal-profiling/A5000-non-atomic-time-scatter-3-3D.png")
end

function plot_intersection_speedup()
    threadripper_df = CSV.read("profiling/internal-profiling/threadripper-results-2.csv", DataFrame)
    a5k_df = CSV.read("profiling/internal-profiling/A5000-atomic-results.csv", DataFrame)

    a5k_intersection = innerjoin(a5k_df, threadripper_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])
    threadripper_intersection = innerjoin(threadripper_df, a5k_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])

    a5k_speedup = threadripper_intersection[:, :Time] ./ a5k_intersection[:, :Time]
    a5k_intersection[!, :Speedup] = a5k_speedup
    
    @df a5k_intersection scatter(
        :NTraces .* :NSamples, :Speedup, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)),
        xlims=(1e5, 1e10),
        xlabel="Samples in dataset", ylabel="Speedup",
        legend=:outertopright,
        title="NVIDIA A5000 speedup (atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-atomic-speedup-scatter.png")

    
    # Threadripper vs non-atomic results
    threadripper_df = CSV.read("profiling/internal-profiling/threadripper-results-2.csv", DataFrame)
    a5k_df = CSV.read("profiling/internal-profiling/A5000-non-atomic-results.csv", DataFrame)

    a5k_intersection = innerjoin(a5k_df, threadripper_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])
    threadripper_intersection = innerjoin(threadripper_df, a5k_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])

    a5k_speedup = threadripper_intersection[:, :Time] ./ a5k_intersection[:, :Time]
    a5k_intersection[!, :Speedup] = a5k_speedup
    
    @df a5k_intersection scatter(
        :NTraces .* :NSamples, :Speedup, group=(:Order, :NLabels),
        markercolor=RGB.(1.0 .* (1/64 .* :Order).^(1/4), 0, 1.0 .* (1/128 .* :NLabels).^(1/4)), 
        xscale=:log10, yscale=:log10, xticks=10.0.^(collect(6:10)),
        xlims=(1e5, 1e10),
        xlabel="Samples in dataset", ylabel="Speedup",
        legend=:outertopright,
        title="NVIDIA A5000 speedup (non-atomic implementation)",
        dpi=300, size=(1000, 600)
    )
    savefig("profiling/internal-profiling/A5000-non-atomic-speedup-scatter.png")

    a5k_intersection[a5k_intersection[:, :Speedup] .>= 1, :]
end


# Publication Plots Below...

function plot_threadripper_results_pub()
    results_df = CSV.read("profiling/internal-profiling/threadripper-results-3.csv", DataFrame)

    fig = Figure(size = (800, 500), fontsize = 14)
    axes = Axis[]
    legend_entries = []
    legend_labels = String[]

    orders = [2, 8, 16, 32]

    for (i, order) in enumerate(orders)
        row = div(i - 1, 2) + 1
        col = mod1(i, 2)

        ax = Axis(
            fig[row, col],
            title = "order=$(order)",
            xlabel = "# Samples / trace", ylabel = "Time(s)",
            xscale = log10, yscale = log10,
            limits = ((1e2, 1600), (1e-3, 1e2))
        )

        push!(axes, ax)

        try
            result_selection = results_df[results_df.Order .== order, :]
            result_selection = result_selection[result_selection.NTraces .== 20000, :]
            nlabels = sort(unique(result_selection.NLabels))

            for nlabel in nlabels
                group_df = result_selection[result_selection.NLabels .== nlabel, :]
                ln = lines!(ax, group_df.NSamples, group_df.Time, color = RGB((nlabel / 256)^(1/4), 0.0, 1-(nlabel / 256)^(1/4)))

                # collect legend entries only once
                if order == orders[1]
                    push!(legend_entries, ln)
                    push!(legend_labels, string(nlabel))
                end
            end
        catch e
            @warn "Failed to plot order=$order" exception=(e, catch_backtrace())
        end
    end

    # Figure title
    Label(fig[0, :], "Moment Estimation Benchmarks: AMD Threadripper Pro 5975W", fontsize = 16)
    Legend(fig[:, 3], legend_entries, legend_labels, "Label size", framevisible=false)
    save("profiling/internal-profiling/publication/threadripper-time-vs-order.png", fig, px_per_unit = 300 / 72)
end

function plot_A5000_results_pub()
    # Atomic routine:
    results_df = CSV.read("profiling/internal-profiling/A5000-atomic-results-3.csv", DataFrame)
    fig = Figure(size = (800, 500), fontsize = 14)
    axes = Axis[]
    legend_entries = []
    legend_labels = String[]

    orders = [2, 8, 16, 32]

    # subplots
    for (i, order) in enumerate(orders)
        row = div(i - 1, 2) + 1
        col = mod1(i, 2)

        ax = Axis(
            fig[row, col],
            title = "order=$(order)",
            xlabel = "# Samples / trace", ylabel = "Time(s)",
            xscale = log10, yscale = log10,
            limits = ((1e2, 1600), (1e-3, 1e2))
        )

        push!(axes, ax)

        try
            result_selection = results_df[results_df.Order .== order, :]
            result_selection = result_selection[result_selection.NTraces .== 20000, :]
            nlabels = sort(unique(result_selection.NLabels))

            for nlabel in nlabels
                group_df = result_selection[result_selection.NLabels .== nlabel, :]
                ln = lines!(ax, group_df.NSamples, group_df.Time, color = RGB((nlabel / 256)^(1/4), 0.0, 1-(nlabel / 256)^(1/4)))

                # collect legend entries only once
                if order == orders[1]
                    push!(legend_entries, ln)
                    push!(legend_labels, string(nlabel))
                end
            end
        catch e
            @warn "Failed to plot order=$order" exception=(e, catch_backtrace())
        end
    end

    Label(fig[0, :], "Moment Estimation Benchmarks: NVIDIA A5000 [atomic]", fontsize = 16)
    Legend(fig[:, 3], legend_entries, legend_labels, "Label size", framevisible=false)
    save("profiling/internal-profiling/publication/A5000-atomic-time-vs-order.png", fig, px_per_unit = 300 / 72)

    # Non-atomic routine
    results_df = CSV.read("profiling/internal-profiling/A5000-non-atomic-results-3.csv", DataFrame)
    fig = Figure(size = (800, 500), fontsize = 14)
    axes = Axis[]
    legend_entries = []
    legend_labels = String[]

    orders = [2, 8, 16, 32]
    
    # subplots
    for (i, order) in enumerate(orders)
        row = div(i - 1, 2) + 1
        col = mod1(i, 2)

        ax = Axis(
            fig[row, col],
            title = "order=$(order)",
            xlabel = "# Samples / trace", ylabel = "Time(s)",
            xscale = log10, yscale = log10,
            limits = ((1e2, 1600), (1e-3, 1e2))
        )

        push!(axes, ax)

        try
            result_selection = results_df[results_df.Order .== order, :]
            result_selection = result_selection[result_selection.NTraces .== 20000, :]
            nlabels = sort(unique(result_selection.NLabels))

            for nlabel in nlabels
                group_df = result_selection[result_selection.NLabels .== nlabel, :]
                ln = lines!(ax, group_df.NSamples, group_df.Time, color = RGB((nlabel / 256)^(1/4), 0.0, 1-(nlabel / 256)^(1/4)))

                # collect legend entries only once
                if order == orders[1]
                    push!(legend_entries, ln)
                    push!(legend_labels, string(nlabel))
                end
            end
        catch e
            @warn "Failed to plot order=$order" exception=(e, catch_backtrace())
        end
    end

    # Figure title
    Label(fig[0, :], "Moment Estimation Benchmarks: NVIDIA A5000 [non-atomic]", fontsize = 16)
    Legend(fig[:, 3], legend_entries, legend_labels, "Label size", framevisible=false)
    save("profiling/internal-profiling/publication/A5000-non-atomic-time-vs-order.png", fig, px_per_unit = 300 / 72)
end

function plot_A5000_speedup_pub()
    cpu_results_df = CSV.read("profiling/internal-profiling/threadripper-results-3.csv", DataFrame)
    gpu_results_df = CSV.read("profiling/internal-profiling/A5000-atomic-results-3.csv", DataFrame)

    gpu_intersection = innerjoin(gpu_results_df, cpu_results_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])
    cpu_intersection = innerjoin(cpu_results_df, gpu_results_df[!, [:Order, :NSamples, :NTraces, :NLabels]], on=[:Order, :NSamples, :NTraces, :NLabels])

    speedup = cpu_intersection[:, :Time] ./ gpu_intersection[:, :Time]
    gpu_intersection[!, :Speedup] = speedup

    fig = Figure(size = (800, 500), fontsize = 14)
    axes = Axis[]
    legend_entries = []
    legend_labels = String[]

    orders = [2, 8, 16, 32]

    # subplots
    for (i, order) in enumerate(orders)
        row = div(i - 1, 2) + 1
        col = mod1(i, 2)

        ax = Axis(
            fig[row, col],
            title = "order=$(order)",
            xlabel = "# Samples / trace", ylabel = "Speedup",
            xscale = log10, yscale = log10,
            limits = ((1e2, 1600), (0.5, 20))
        )

        push!(axes, ax)

        try
            result_selection = gpu_intersection[gpu_intersection.Order .== order, :]
            result_selection = result_selection[result_selection.NTraces .== 20000, :]
            nlabels = sort(unique(result_selection.NLabels))

            for nlabel in nlabels
                group_df = result_selection[result_selection.NLabels .== nlabel, :]
                ln = lines!(ax, group_df.NSamples, group_df.Speedup, color = RGB((nlabel / 256)^(1/4), 0.0, 1-(nlabel / 256)^(1/4)))

                # collect legend entries only once
                if order == orders[1]
                    push!(legend_entries, ln)
                    push!(legend_labels, string(nlabel))
                end
            end
        catch e
            @warn "Failed to plot order=$order" exception=(e, catch_backtrace())
        end
    end

    Label(fig[0, :], "Moment Estimation Benchmarks: GPU Speedup vs CPU [atomic GPU kernel]", fontsize = 16)
    Legend(fig[:, 3], legend_entries, legend_labels, "Label size", framevisible=false)
    save("profiling/internal-profiling/publication/speedup-vs-order.png", fig, px_per_unit = 300 / 72)
end

function plot_pub()
    plot_threadripper_results_pub()
    plot_A5000_results_pub()
    plot_A5000_speedup_pub()
end