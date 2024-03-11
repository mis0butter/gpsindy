using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 


## ============================================ ##
# test single file 

freq_hz  = 50 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )
csv_file = "rollout_25.csv" 

# extract data 
data_train, data_test = car_data_struct( string(csv_path, csv_file) ) 

# smooth with GPs 
σ_n = 1e-2 
x_train_GP  = gp_post( data_train.t, 0*data_train.x_noise, data_train.t, 0*data_train.x_noise, data_train.x_noise, ) 
dx_train_GP = gp_post( x_train_GP, 0*data_train.dx_noise, x_train_GP, 0*data_train.dx_noise, data_train.dx_noise, ) 
x_test_GP   = gp_post( data_test.t, 0*data_test.x_noise, data_test.t, 0*data_test.x_noise, data_test.x_noise ) 
dx_test_GP  = gp_post( x_test_GP, 0*data_test.dx_noise, x_test_GP, 0*data_test.dx_noise, data_test.dx_noise ) 

# get λ_vec 
λ_vec = λ_vec_fn() 


## ============================================ ## 
# test cross_validate_λ for gpsindy 

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

# try i = 1 
i_λ = 1 

x_err_hist = x_train_test_err_struct( [], [], [], [] ) 
for i_λ = eachindex( λ_vec ) 

    λ   = λ_vec[i_λ] 
    println( "λ = ", @sprintf "%.3g" λ ) 

    data_pred_train, data_pred_test = sindy_gpsindy_λ( data_train, data_test, x_train_GP, dx_train_GP, x_test_GP, λ ) 
    
    # ----------------------- #
    # plot and save metrics 
    
    f = plot_err_train_test( data_pred_train, data_pred_test, data_train, data_test, λ, freq_hz, csv_file)     
    display(f) 

    x_err_hist = push_err_metrics( x_err_hist, data_train, data_test, data_pred_train, data_pred_test ) 
    
end 

df_λ_vec, df_sindy, df_gpsindy = df_metrics( x_err_hist, λ_vec )











## ============================================ ##
## ============================================ ##
# plotting fns 

function plot_err_train_test( data_pred_train, data_pred_test, data_train, data_test, λ, freq_hz, csv_file) 
    
    x_train_GP, x_sindy_train, x_gpsindy_train, x_test_GP, x_sindy_test, x_gpsindy_test = extract_preds( data_pred_train, data_pred_test ) 

    x_train_noise   = data_train.x_noise 
    x_test_noise    = data_test.x_noise 

    x_gp_err_train       = @sprintf "%.3g" norm( x_train_noise - x_train_GP ) 
    x_sindy_err_train    = @sprintf "%.3g" norm( x_train_noise - x_sindy_train ) 
    x_gpsindy_err_train  = @sprintf "%.3g" norm( x_train_noise - x_gpsindy_train ) 

    x_gp_err_test        = @sprintf "%.3g" norm( x_test_noise - x_test_GP ) 
    x_sindy_err_test     = @sprintf "%.3g" norm( x_test_noise - x_sindy_test ) 
    x_gpsindy_err_test   = @sprintf "%.3g" norm( x_test_noise - x_gpsindy_test ) 

    f = Figure( size = ( 800,700 ) ) 
    f = plot_ix_err_train_test( f, 1, data_pred_train, data_pred_test, data_train, data_test) 
    f = plot_ix_err_train_test( f, 2, data_pred_train, data_pred_test, data_train, data_test) 
    f = plot_ix_err_train_test( f, 3, data_pred_train, data_pred_test, data_train, data_test) 
    f = plot_ix_err_train_test( f, 4, data_pred_train, data_pred_test, data_train, data_test) 
    # ax = Axis( f[5,1:2] ) 
    ax_text = "total err: GP = $x_gp_err_train, \n SINDy = $x_sindy_err_train, GPSINDy = $x_gpsindy_err_train" 
    Textbox( f[5,1:2], placeholder = ax_text, textcolor_placeholder = :black ) 

    ax_text = "total err: GP = $x_gp_err_test, \n SINDy = $x_sindy_err_test, GPSINDy = $x_gpsindy_err_test" 
    Textbox( f[5,3:4], placeholder = ax_text, textcolor_placeholder = :black ) 

    λ = @sprintf "%.3g" λ 
    ax_text = "λ = $λ \n $freq_hz Hz \n $csv_file" 
    Textbox( f[5,5], placeholder = ax_text, textcolor_placeholder = :black ) 

    return f 
end 

## ============================================ ##

function plot_ix_err_train_test( f, i_x, data_pred_train, data_pred_test, data_train, data_test) 
    
    x_train_GP, x_sindy_train, x_gpsindy_train, x_test_GP, x_sindy_test, x_gpsindy_test = extract_preds( data_pred_train, data_pred_test ) 

    x_train_noise = data_train.x_noise ; x_test_noise  = data_test.x_noise 
    t_train       = data_train.t       ; t_test        = data_test.t 

    x_gp_err_train       = @sprintf "%.3g" norm( x_train_noise[:,i_x] - x_train_GP[:,i_x] ) 
    x_sindy_err_train    = @sprintf "%.3g" norm( x_train_noise[:,i_x] - x_sindy_train[:,i_x] ) 
    x_gpsindy_err_train  = @sprintf "%.3g" norm( x_train_noise[:,i_x] - x_gpsindy_train[:,i_x] ) 
        
    x_gp_err_test        = @sprintf "%.3g" norm( x_test_noise[:,i_x] - x_test_GP[:,i_x] ) 
    x_sindy_err_test     = @sprintf "%.3g" norm( x_test_noise[:,i_x] - x_sindy_test[:,i_x] ) 
    x_gpsindy_err_test   = @sprintf "%.3g" norm( x_test_noise[:,i_x] - x_gpsindy_test[:,i_x] ) 
    
    min_y, max_y = min_max_y( x_train_noise[:,i_x] ) 
    ax = Axis( f[i_x,1:2], limits = ( nothing, nothing, min_y, max_y ), title = "train x$i_x err: GP = $x_gp_err_train, \n SINDy = $x_sindy_err_train, GPSINDy = $x_gpsindy_err_train" )
        noise   = CairoMakie.scatter!( ax, t_train, data_train.x_noise[:,i_x], color = :black, label = "noise" ) 
        GP      = lines!( ax, t_train, x_train_GP[:,i_x], color = :red, label = "GP" )
        sindy   = lines!( ax, t_train, x_sindy_train[:,i_x], label = "sindy") 
        gpsindy = lines!( ax, t_train, x_gpsindy_train[:,i_x], label = "gpsindy" ) 
    
    min_y, max_y = min_max_y( x_test_noise[:,i_x] ) 
    ax = Axis( f[i_x,3:4], limits = ( nothing, nothing, min_y, max_y ), title = "test x$i_x err: GP = $x_gp_err_test, \n SINDy = $x_sindy_err_test, GPSINDy = $x_gpsindy_err_test" )
        noise   = CairoMakie.scatter!( ax, t_test, data_test.x_noise[:,i_x], color = :black, label = "noise" ) 
        GP      = lines!( ax, t_test, x_test_GP[:,i_x], color = :red, label = "GP" ) 
        sindy   = lines!( ax, t_test, x_sindy_test[:,i_x], label = "sindy") 
        gpsindy = lines!( ax, t_test, x_gpsindy_test[:,i_x], label = "gpsindy" ) 
    
        if i_x == 1 
            Legend( f[i_x,5], [noise, GP, sindy, gpsindy], ["noise", "GP", "sindy", "gpsindy"], )  
        end 

    return f 
end 


