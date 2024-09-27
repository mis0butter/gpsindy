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

csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz_noise_0/rollout_3.csv" 

df_best_sindy, df_best_gpsindy, fig_csv = cross_validate_csv( csv_path_file, 1 ) 
fig_csv 

## ============================================ ## 


# df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.01 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.04 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.05 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.06 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.07 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.08 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.09 ) 


## ============================================ ## 
## ============================================ ## 
# save results 


# go through all files in result folder 
results_path    = "test/results/jake_car_csvs_ctrlshift_no_trans/" 

somi_df = create_somi_df( results_path ) 

# save data frame 
CSV.write( string( results_path, "somi_df.csv" ), somi_df ) 


## ============================================ ## 


function boxplot_rmse_noise(somi_df, hz)

    # Extract rows from somi_df where hz matches the input
    somi_df_hz_sindy   = filter(row -> row.hz == string(hz) && row.method == "sindy", somi_df)
    somi_df_hz_gpsindy = filter(row -> row.hz == string(hz) && row.method == "gpsindy", somi_df)

    # Convert noise column to Float64
    somi_df_hz_sindy.noise   = parse.(Float64, somi_df_hz_sindy.noise)
    somi_df_hz_gpsindy.noise = parse.(Float64, somi_df_hz_gpsindy.noise) 

    # Convert rmse column to Float64
    somi_df_hz_sindy.rmse   = parse.(Float64, string.(somi_df_hz_sindy.rmse))
    somi_df_hz_gpsindy.rmse = parse.(Float64, string.(somi_df_hz_gpsindy.rmse)) 

    # Create a box plot with log scale for y-axis
    fig = Figure(size=(600, 400))

    ax = Axis(fig[1, 1], xlabel="Noise Level", ylabel="Log RMSE error", title="Log RMSE vs Noise Level for $(hz)Hz Data") 

    # Define colors for each method
    sindy_color = colorant"#1f77b4"  # A nice blue
    gpsindy_color = colorant"#ff7f0e"  # A complementary orange

    # Add some jittered points for better data representation
    scatter!(ax, somi_df_hz_sindy.noise .- 0.0005 .+ 0.001 .* randn(length(somi_df_hz_sindy.noise)),
             log.(somi_df_hz_sindy.rmse), color=(sindy_color, 0.3), markersize=7)
    
    scatter!(ax, somi_df_hz_gpsindy.noise .+ 0.0005 .+ 0.001 .* randn(length(somi_df_hz_gpsindy.noise)),
             log.(somi_df_hz_gpsindy.rmse), color=(gpsindy_color, 0.3), markersize=7)

    # Create boxplots with improved aesthetics, transparency, and outline
    boxplot!(ax, somi_df_hz_sindy.noise .- 0.0005, log.(somi_df_hz_sindy.rmse),
             width=0.008, label="SINDy", color=(sindy_color, 0.4), whiskerwidth=0.5, whiskercolor=sindy_color,
             mediancolor=sindy_color, show_outliers=false, strokecolor=sindy_color, strokewidth=1)
    
    boxplot!(ax, somi_df_hz_gpsindy.noise .+ 0.0005, log.(somi_df_hz_gpsindy.rmse),
             width=0.008, label="GPSINDy", color=(gpsindy_color, 0.4), whiskerwidth=0.5, whiskercolor=gpsindy_color, mediancolor=gpsindy_color, show_outliers=false, strokecolor=gpsindy_color, strokewidth=1)

    # Add legend
    Legend(fig[1,2], ax, framevisible = false)

    return fig 
end 

fig = boxplot_rmse_noise( somi_df, 5 )  

# Example usage:
# fig = plot_rmse_vs_noise(somi_df, 10)
# fig




