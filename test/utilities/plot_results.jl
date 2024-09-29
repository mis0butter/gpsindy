
using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 


## ============================================ ## 


# contains SSR_coeff and SSR_residual 
SSR_coeff_res_df = CSV.read( "test/results/smooth_rmse_results.csv", DataFrame )  

# contains sindy and gpsindy 
sindy_gpsindy_df = CSV.read( "test/results/jake_car_csvs_ctrlshift_no_trans/somi_df.csv", DataFrame )  

# combine with somi_df 
combined_df = vcat( SSR_coeff_res_df, sindy_gpsindy_df )  

# get headers 
headers = names(combined_df) 


## ============================================ ## 


function boxplot_rmse_noise(combined_df, hz)

    function create_boxplot!(ax, df, label, color, offset)
        boxplot!(ax, df.noise .+ offset, log.(df.rmse),
                 width=0.008, label=label, color=(color, 0.4), whiskerwidth=0.5, whiskercolor=color,
                 mediancolor=color, show_outliers=false, strokecolor=color, strokewidth=1)
    end

    # Extract rows from somi_df where hz matches the input
    sindy_df   = filter(row -> row.hz == hz && row.method == "sindy", combined_df)
    gpsindy_df = filter(row -> row.hz == hz && row.method == "gpsindy", combined_df) 
    ssr_coeff_df  = filter(row -> row.hz == hz && row.method == "SSR_coeff", combined_df) 
    ssr_res_df    = filter(row -> row.hz == hz && row.method == "SSR_residual", combined_df)  

    # Create a box plot with log scale for y-axis
    fig = Figure(size=(700, 400))

    ax = Axis(fig[1, 1], xlabel="Noise Level", ylabel="Log RMSE error", title="Log RMSE vs Noise Level for $(hz)Hz Data") 

    # Define colors for each method 
    ssr_coeff_color = colorant"#800080"  # Purple
    ssr_res_color = colorant"#1a5f8f"  # A slightly lighter blue
    sindy_color = colorant"#00cc00"  # Green
    gpsindy_color = colorant"#cc6600"  # A slightly lighter orange

    # Add some jittered points for better data representation
    scatter!(ax, sindy_df.noise .- 0.0005 .+ 0.001 .* randn(length(sindy_df.noise)),
             log.(sindy_df.rmse), color=(sindy_color, 0.2), markersize=7)
    
    scatter!(ax, gpsindy_df.noise .+ 0.0005 .+ 0.001 .* randn(length(gpsindy_df.noise)),
             log.(gpsindy_df.rmse), color=(gpsindy_color, 0.2), markersize=7)


    # Create boxplots with improved aesthetics, transparency, and outline
    create_boxplot!(ax, ssr_coeff_df, "SSR Coeff", ssr_coeff_color, -0.001)
    create_boxplot!(ax, ssr_res_df, "SSR Res", ssr_res_color, -0.0005)
    create_boxplot!(ax, sindy_df, "SINDy", sindy_color, 0.0005)
    create_boxplot!(ax, gpsindy_df, "GPSINDy", gpsindy_color, 0.001)

    # Add legend
    Legend(fig[1,2], ax, framevisible = false)

    return fig 
end 


## ============================================ ## 

hz = 25 
fig = boxplot_rmse_noise( combined_df, hz )  