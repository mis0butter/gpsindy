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

# Create prediction points
x_pred = range(0, 5, length=200)

## ============================================ ## 

# ... existing code ...

# Define a function to evaluate kernel performance
function evaluate_kernel(kernel, x, y)
    m = MeanZero()
    logNoise = log(0.1)
    gp = GP(x, y, m, kernel, logNoise)
    optimize!(gp)
    return gp.target  # Return log marginal likelihood
end

# Define a list of kernels to try
kernels = [
    Periodic(0.5, 1.0, 1.0) + SE(0.1, 0.1),
    Periodic(0.5, 1.0, 1.0) * SE(0.1, 0.1),
    SE(1.0, 1.0) + Periodic(0.5, 1.0, 1.0),
    RQ(1.0, 1.0, 1.0) + Periodic(0.5, 1.0, 1.0),
    Matern(3/2, 1.0, 1.0) + Periodic(0.5, 1.0, 1.0)
]

# Evaluate each kernel
results = []
for (i, kernel) in enumerate(kernels)
    score = evaluate_kernel(kernel, x, y)
    push!(results, (i, kernel, score))
    println("Kernel $i: Log marginal likelihood = $score")
end

# Find the best kernel
best_kernel = nothing
best_score = -Inf
for result in results
    if result[3] > best_score
        best_kernel = result
        best_score = result[3]
    end
end

if best_kernel === nothing
    error("No valid kernel found")
end
println("Best kernel: ", best_kernel[2], " with score ", best_kernel[3])

# Use the best kernel for final GP
best_gp = GP(x, y, MeanZero(), best_kernel[2], log(0.1))
optimize!(best_gp)

# Make predictions with the best kernel
μ_best, σ²_best = predict_y(best_gp, x_pred)

# Plot results
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Best Gaussian Process Kernel")

lines!(ax, x_pred, true_f.(x_pred), label="True function", linewidth=2)
scatter!(ax, x, y, label="Observations", markersize=4)
lines!(ax, x_pred, μ_best, label="GP prediction", linewidth=2)
band!(ax, x_pred, μ_best .- 2*sqrt.(σ²_best), μ_best .+ 2*sqrt.(σ²_best), color=(:blue, 0.3))

axislegend(ax)
fig

# ... existing code ...