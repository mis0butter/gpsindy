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

## ============================================ ## 

# Print GP kernel parameters
println("Optimized kernel parameters:")
println("Periodic kernel:")
println("  Length scale (l): ", gp.kernel.kernels[1].l)
println("  Period (p): ", gp.kernel.kernels[1].p)
println("  Signal variance (σ): ", exp(gp.kernel.kernels[1].lσ))
println("Squared Exponential kernel:")
println("  Length scale: ", gp.kernel.kernels[2].ℓ)
println("  Signal variance: ", exp(gp.kernel.kernels[2].lσ))
println("Noise variance: ", exp(2 * gp.logNoise))

# Get the length scale from the periodic kernel (first kernel in the sum)
periodic_kernel = gp.kernel.kernels[1]
length_scale = periodic_kernel.l

println("Length scale from gp.kernel.kernels[1].l: ", length_scale)

# If you want to access it through kleft (although it's not recommended):
# Note: This assumes the periodic kernel is the left kernel in the sum
kleft_length_scale = gp.kernel.kleft.l

println("Length scale from gp.kernel.kleft.l: ", kleft_length_scale)

# Verify that both methods give the same result
@assert length_scale == kleft_length_scale "Length scales should be equal"



