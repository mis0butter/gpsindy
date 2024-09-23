## ============================================ ## 

# Plot results
using CairoMakie
using GaussianProcesses 
using Random 

Random.seed!(1) 

# Generate synthetic data
function true_f(x)
    return sin(2π * x) + 0.5 * sin(4π * x)
end

n = 50
x = sort(rand(n)) * 4  # Input points between 0 and 4
y = true_f.(x) + 0.1 * randn(n)  # Add some noise

# Define the periodic kernel with initial parameters
l = 0.5  # length scale (reduced from 1.0)
p = 1.0  # period 
σ = 1.0  # signal variance 
kernel = Periodic(l, p, σ) + SE(0.1, 0.1)

# Create and fit the Gaussian Process
m = MeanZero()  # zero mean function
logNoise = log(0.1)
gp = GP(x, y, m, kernel, logNoise)

# Optimize hyperparameters
optimize!(gp) 

# Make predictions
x_pred = range(0, 5, length=200)
μ, σ² = predict_y(gp, x_pred)

# Make predictions
x_pred = range(0, 5, length=200)
μ, σ² = predict_y(gp, x_pred)

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Gaussian Process with Periodic Kernel")

lines!(ax, x_pred, true_f.(x_pred), label="True function", linewidth=2)
scatter!(ax, x, y, label="Observations", markersize=4)
lines!(ax, x_pred, μ, label="GP prediction", linewidth=2)
band!(ax, x_pred, μ .- 2*sqrt.(σ²), μ .+ 2*sqrt.(σ²), color=(:blue, 0.3))

axislegend(ax)
fig




