using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 


## ============================================ ##
# test single file 

freq_hz  = 50 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )
csv_file = "rollout_26.csv" 

# extract data 
data_train, data_test = car_data_struct( string(csv_path, csv_file) ) 



    # # wrap in data frame --> Matrix 
    # df   = CSV.read( string( csv_path, csv_file ), DataFrame) 
    # data = Matrix(df) 
    
    # # extract variables 
    # t = data[:,1] 
    # x = data[:,2:end-2]
    # u = data[:,end-1:end] 

    # f = Figure(  ) 
    # ax = f[1,1] = Axis( f, xlabel = "time (s)", ylabel = "x" ) 
    #     lines!( ax, t, x[:,4] )
    # display(f) 

    # # x, dx_fd = unroll( t, x ) 

    # # use forward finite differencing 
    # dx = fdiff(t, x, 1) 

    # # massage data, generate rollovers  
    # rollover_up_ind = findall( x -> x > 10, dx[:,4] ) 
    # rollover_dn_ind = findall( x -> x < -10, dx[:,4] ) 

    # up_length = length(rollover_up_ind) 
    # dn_length = length(rollover_dn_ind) 

    # ind_min = minimum( [ up_length, dn_length ] ) 

    # for i in 1 : ind_min  
        
    #     if rollover_up_ind[i] < rollover_dn_ind[i] 
    #         i0   = rollover_up_ind[i] + 1 
    #         ifin = rollover_dn_ind[i]     
    #         rollover_rng = x[ i0 : ifin , 4 ]
    #         dθ = π .- rollover_rng 
    #         θ  = -π .- dθ 
    #     else 
    #         i0   = rollover_dn_ind[i] + 1 
    #         ifin = rollover_up_ind[i]     
    #         rollover_rng = x[ i0 : ifin , 4 ]
    #         dθ = π .+ rollover_rng 
    #         θ  = π .+ dθ     
    #     end 
    #     x[ i0 : ifin , 4 ] = θ

    # end 

    # if up_length > dn_length 
    #     i0   = rollover_up_ind[end] + 1 
    #     rollover_rng = x[ i0 : end , 4 ]
    #     dθ = π .- rollover_rng 
    #     θ  = -π .- dθ 
    # elseif up_length < dn_length 
    #     i0   = rollover_dn_ind[end] + 1 
    #     rollover_rng = x[ i0 : end , 4 ]
    #     dθ = π .+ rollover_rng 
    #     θ  = π .+ dθ     
    # end 
    # x[ i0 : end , 4 ] = θ

    # lines!( ax, t, x[:,4], color = :red ) 
    # display(f)  



## ============================================ ##


# smooth with GPs 
# σ_n = 1e-2 
# x_train_GP  = gp_post( data_train.t, 0*data_train.x_noise, data_train.t, 0*data_train.x_noise, data_train.x_noise, ) 
# dx_train_GP = gp_post( x_train_GP, 0*data_train.dx_noise, x_train_GP, 0*data_train.dx_noise, data_train.dx_noise, ) 
# x_test_GP   = gp_post( data_test.t, 0*data_test.x_noise, data_test.t, 0*data_test.x_noise, data_test.x_noise ) 
x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 

# get λ_vec 
λ_vec = λ_vec_fn() 


# ----------------------- #
# test cross_validate_λ for sindy and gpsindy 

x_err_hist = x_train_test_err_struct( [], [], [], [] ) 
for i_λ = eachindex( λ_vec ) 

    λ   = λ_vec[i_λ] 
    println( "λ = ", @sprintf "%.3g" λ ) 

    data_pred_train, data_pred_test = sindy_gpsindy_λ( data_train, data_test, x_train_GP, dx_train_GP, x_test_GP, λ ) 
    
    # plot and save metrics     
    f = plot_err_train_test( data_pred_train, data_pred_test, data_train, data_test, λ, freq_hz, csv_file)     
    display(f) 

    x_err_hist = push_err_metrics( x_err_hist, data_train, data_test, data_pred_train, data_pred_test ) 
    
end 

df_λ_vec, df_sindy, df_gpsindy = df_metrics( x_err_hist, λ_vec )







