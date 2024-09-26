using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using Optim 
using LineSearches  
using GaussianProcesses 
using DataFrames 


## ============================================ ## 


csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz_noise_0.02/rollout_5.csv" 


## ============================================ ## 
## ============================================ ## 


function smooth_data_with_gp(data_train, i_u, i_x)

    x_data = data_train.u[:,i_u] 
    y_data = data_train.dx_noise[:,i_x]
    x_pred = data_train.u[:,i_u] 
    smoothed_dx_u, _, _ = smooth_column_gp(x_data, y_data, x_pred) 

    x_data = data_train.t 
    y_data = data_train.x_noise 
    x_pred = data_train.t 
    smoothed_x_t, _, _ = smooth_array_gp(x_data, y_data, x_pred)   

    x_data = smoothed_x_t 
    y_data = data_train.dx_noise[:,i_x]
    x_pred = smoothed_x_t 
    smoothed_dx_x_t, _, _ = smooth_column_gp(x_data, y_data, x_pred) 

    x_data = data_train.x_noise[:,i_x] 
    y_data = data_train.dx_noise[:,i_x]
    x_pred = data_train.x_noise[:,i_x] 
    smoothed_dx_x, _, _ = smooth_column_gp(x_data, y_data, x_pred)  

    x_data = data_train.t 
    y_data = data_train.dx_noise[:,i_x]
    x_pred = data_train.t 
    smoothed_dx_t, _, _ = smooth_column_gp(x_data, y_data, x_pred)  

    return smoothed_dx_u, smoothed_x_t, smoothed_dx_x_t, smoothed_dx_x, smoothed_dx_t
end 

smoothed_dx1_u, smoothed_x1_t, smoothed_dx1_x_t, smoothed_dx1_x, smoothed_dx1_t = smooth_data_with_gp(data_train, 1, 1)  

smoothed_dx2_u, smoothed_x2_t, smoothed_dx2_x_t, smoothed_dx2_x, smoothed_dx2_t = smooth_data_with_gp(data_train, 2, 2) 

smoothed_dx3_u, smoothed_x3_t, smoothed_dx3_x_t, smoothed_dx3_x, smoothed_dx3_t = smooth_data_with_gp(data_train, 1, 3) 

smoothed_dx4_u, smoothed_x4_t, smoothed_dx4_x_t, smoothed_dx4_x, smoothed_dx4_t = smooth_data_with_gp(data_train, 2, 4) 

## ============================================ ## 

# Load the data from the CSV file
data_train, data_test = make_data_structs(csv_path_file)

# Plot the control inputs
function plot_data!(ax, data_train, i_u, i_x)
    lines!(ax, data_train.t, data_train.u[:, i_u], color = :red, linewidth=2, label="u$i_u")
    lines!(ax, data_train.t, data_train.x_noise[:,i_x], color = :blue, linewidth=2, label="x$i_x")  
    lines!(ax, data_train.t, data_train.dx_noise[:,i_x], color = :blue, linestyle = :dash, linewidth=2, label="dx$i_x")  
end

function plot_smoothed_data(data_train, smoothed_dx_u, smoothed_dx_x, smoothed_dx_t, smoothed_dx_x_t, i_u, i_x)
    # Create a new figure
    fig = Figure(size=(600, 800))

    # Create an axis for the plot
    ax = Axis(fig[1, 1], 
        xlabel = "Time (s)", 
        title = "smoothed dx$i_x kernel: u")
        plot_data!(ax, data_train, i_u, i_x)
        lines!(ax, data_train.t, smoothed_dx_u, color = :cyan, label="gp dx3" )

        # Add a legend
        axislegend(ax)

    ax = Axis(fig[2, 1], 
        xlabel = "Time (s)", 
        title = "smoothed dx$i_x kernel: x")  
        plot_data!(ax, data_train, i_u, i_x)
        lines!(ax, data_train.t, smoothed_dx_x, color = :cyan, label="gp dx3" ) 

    ax = Axis(fig[3, 1],  
        xlabel = "Time (s)", 
        title = "smoothed dx$i_x kernel:t")  
        plot_data!(ax, data_train, i_u, i_x)
        lines!(ax, data_train.t, smoothed_dx_t, color = :cyan, label="gp dx3" )  

    ax = Axis(fig[4, 1],   
        xlabel = "Time (s)", 
        title = "smoothed dx$i_x kernel:x kernel:t")   
        plot_data!(ax, data_train, i_u, i_x)
        lines!(ax, data_train.t, smoothed_dx_x_t, color = :cyan, label="gp dx3" )   

    return fig
end

fig = plot_smoothed_data(data_train, smoothed_dx1_u, smoothed_dx1_x, smoothed_dx1_t, smoothed_dx1_x_t, 1, 1) 

fig = plot_smoothed_data(data_train, smoothed_dx2_u, smoothed_dx2_x, smoothed_dx2_t, smoothed_dx2_x_t, 2, 2) 

fig = plot_smoothed_data(data_train, smoothed_dx3_u, smoothed_dx3_x, smoothed_dx3_t, smoothed_dx3_x_t, 1, 3)

fig = plot_smoothed_data(data_train, smoothed_dx4_u, smoothed_dx4_x, smoothed_dx4_t, smoothed_dx4_x_t, 2, 4) 

# Display the figure
display(fig)


