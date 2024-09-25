using GaussianProcesses 
using Random 
using CairoMakie  
using GaussianSINDy 
using LineSearches 
using Optim 

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

# define data 
x_data = x 
y_data = y 


## ============================================ ## 


y_data = y 
μ_best, σ²_best, best_gp = smooth_column_gp(x_data, y_data, x_pred) 

# Plot results
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Best Gaussian Process Kernel")

lines!(ax, x_pred, true_f.(x_pred), label="True function", linewidth=2)
scatter!(ax, x_data, y_data, label="Observations", markersize=4)
lines!(ax, x_pred, μ_best, label="GP prediction", linewidth=2)
band!(ax, x_pred, μ_best .- 2*sqrt.(σ²_best), μ_best .+ 2*sqrt.(σ²_best), color=(:blue, 0.3))

axislegend(ax)

print_kernel(best_gp) 
fig 


## ============================================ ## 

y_data = [y z]  
μ_best, σ²_best, best_gps = smooth_array_gp(x_data, y_data, x_pred) 

# Plot results 
fig = Figure()
ax  = [Axis(fig[i, 1], xlabel="x", ylabel="y", title="Best Gaussian Process Kernel - Column $i") for i in 1:size(μ_best, 2)]

for i in 1:size(μ_best, 2) 

    lines!(ax[i], x_pred, true_f.(x_pred), label="True function", linewidth=2)
    scatter!(ax[i], x_data, y_data[:, i], label="Observations", markersize=4)
    lines!(ax[i], x_pred, μ_best[:, i], label="GP prediction", linewidth=2)
    band!(ax[i], x_pred, μ_best[:, i] .- 2*sqrt.(σ²_best[:, i]), μ_best[:, i] .+ 2*sqrt.(σ²_best[:, i]), color=(:blue, 0.3))
    axislegend(ax[i])

    if ax == 1 
        axislegend(ax) 
    end 
end 

print_kernel(best_gp) 
fig


## ============================================ ## 
## ============================================ ## 


# Define a function to evaluate kernel performance
function test_evaluate_kernel(kernel, x_data, y_data)
    
    # Create the GP with a try-catch block
    try

        gp = GP(x_data', y_data, MeanZero(), kernel, log(0.1))
        
        try

            optimize!(gp, 
                method = LBFGS(linesearch = LineSearches.BackTracking()), 
                iterations = 100 
            )
            return (gp.target, gp)  # Return log marginal likelihood

        catch opt_error

            println("Optimization error: ", opt_error)
            return (-Inf, nothing)  # Return a very low score for failed optimizations

        end

    catch gp_error

        println("GP creation error: ", gp_error)
        return (-Inf, nothing)  # Return a very low score if GP creation fails

    end
end 

function test_evaluate_kernels(kernels, x_data, y_data)

    results = []
    for (i, kernel) in enumerate(kernels) 

        score, gp = test_evaluate_kernel(kernel, x_data, y_data)
        result    = (i = i, kernel = kernel, score = score, gp = gp) 

        push!(results, result)
        println("Kernel $i: Log marginal likelihood = $score")

    end

    return results
end 

function test_find_best_kernel(results)

    # Find the best kernel
    best_result = nothing
    best_score  = -Inf
    for result in results

        if result.score > best_score
            best_result = result
            best_score  = result.score 
        end

    end

    return best_result
end 

## ============================================ ## 


kernels = define_kernels(x_data, y_data)  
results = evaluate_kernels(kernels, x_data, y_data)  
best_result = find_best_kernel(results) 

best_gp = best_result.gp 

μ_best, σ²_best = predict_y(best_gp, x_pred')




