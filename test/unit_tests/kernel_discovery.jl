using GaussianProcesses 
using Random 
using CairoMakie  

## ============================================ ## 
# generate data 

Random.seed!(1) 

# Generate synthetic data
function true_f(x)
    return sin(2π * x) + 0.5 * sin(4π * x)
end

n = 50
x = sort(rand(n)) * 4  # Input points between 0 and 4 
y = true_f.(x) + 0.1 * randn(n)  # Add some noise 
z = true_f.(x) + 0.2 * randn(n)  # Add some noise  

# Create prediction points
x_pred = range(0, 4, length=200) 

## ============================================ ## 

μ_best, σ²_best, best_gp = smooth_column_gp(x, z, x_pred) 

# Plot results
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Best Gaussian Process Kernel")

lines!(ax, x_pred, true_f.(x_pred), label="True function", linewidth=2)
scatter!(ax, x, y, label="Observations", markersize=4)
lines!(ax, x_pred, μ_best, label="GP prediction", linewidth=2)
band!(ax, x_pred, μ_best .- 2*sqrt.(σ²_best), μ_best .+ 2*sqrt.(σ²_best), color=(:blue, 0.3))

axislegend(ax)
fig

print_kernel(best_gp) 

## ============================================ ## 

y_data = [y z]  
μ_best, σ²_best, best_gp = smooth_array_gp(x, y_data, x_pred) 

# Plot results
fig = Figure()
ax = [Axis(fig[i, 1], xlabel="x", ylabel="y", title="Best Gaussian Process Kernel - Column $i") for i in 1:size(μ_best, 2)]

for i in 1:size(μ_best, 2)

    lines!(ax[i], x_pred, true_f.(x_pred), label="True function", linewidth=2)
    scatter!(ax[i], x, [y z][:, i], label="Observations", markersize=4)
    lines!(ax[i], x_pred, μ_best[:, i], label="GP prediction", linewidth=2)
    band!(ax[i], x_pred, μ_best[:, i] .- 2*sqrt.(σ²_best[:, i]), μ_best[:, i] .+ 2*sqrt.(σ²_best[:, i]), color=(:blue, 0.3))
    axislegend(ax[i])

    if ax == 1 
        axislegend(ax)
    end 
end

fig

print_kernel(best_gp) 

