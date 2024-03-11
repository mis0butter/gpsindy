using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 


## ============================================ ##
# test single file 

csv_path = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz/" 
csv_file = "rollout_3.csv" 

# extract data 
data_train, data_test = car_data_struct( string(csv_path, csv_file) ) 

t_train        = data_train.t           ; t_test        = data_test.t 
x_train_noise  = data_train.x_noise     ; x_test_noise  = data_test.x_noise 
dx_train_noise = data_train.dx_noise    ; dx_test_noise = data_test.dx_noise 
u_train        = data_train.u           ; u_test        = data_test.u 

# smooth with GPs 
σ_n = 1e-2 
x_train_GP  = gp_post( data_train.t, 0*data_train.x_noise, data_train.t, 0*data_train.x_noise, data_train.x_noise, ) 
dx_train_GP = gp_post( x_train_GP, 0*data_train.dx_noise, x_train_GP, 0*data_train.dx_noise, data_train.dx_noise, ) 
x_test_GP   = gp_post( data_test.t, 0*data_test.x_noise, data_test.t, 0*data_test.x_noise, data_test.x_noise ) 
dx_test_GP  = gp_post( x_test_GP, 0*data_test.dx_noise, x_test_GP, 0*data_test.dx_noise, data_test.dx_noise ) 

# get x0 from smoothed data 
x0_train    = data_train.x_noise[1,:] ; x0_train_GP = x_train_GP[1,:] 
x0_test     = data_test.x_noise[1,:]  ; x0_test_GP  = x_test_GP[1,:] 

# get λ_vec 
λ_vec = λ_vec_fn() 


## ============================================ ##
# test cross_validate_λ for gpsindy 

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

# build function library from smoothed data 
Θx_gp  = pool_data_test( [ x_train_GP u_train ], n_vars, poly_order ) 

# try i = 1 
i_λ = 25 

for i_λ = eachindex( λ_vec )

    λ   = λ_vec[i_λ] 
    println( "λ = ", @sprintf "%.3g" λ ) 
    
    # ----------------------- # 
    # SINDY-lasso ! 
    Ξ_sindy = sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 
    dx_sindy = Θx_gp * Ξ_sindy 
    
    # integrate discovered dynamics 
    dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
    x_sindy_train = integrate_euler( dx_fn_sindy, x0_train, data_train.t, data_train.u ) 
    x_sindy_test  = integrate_euler( dx_fn_sindy, x0_test, data_test.t, data_test.u ) 
    
    # ----------------------- #
    # GPSINDy-lasso ! 
    Ξ_gpsindy  = sindy_lasso( x_train_GP, dx_train_GP, λ, u_train ) 
    dx_gpsindy = Θx_gp * Ξ_gpsindy 
    
    # integrate discovered dynamics 
    dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
    x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train, data_train.t, data_train.u ) 
    x_gpsindy_test  = integrate_euler( dx_fn_gpsindy, x0_test, data_test.t, data_test.u ) 
    
    # ----------------------- #
    # collect data 
    
    data_pred_train = data_predicts( x_train_GP, dx_train_GP, x_sindy_train, dx_sindy, x_gpsindy_train, dx_gpsindy ) 
    data_pred_test  = data_predicts( x_test_GP, dx_test_GP, x_sindy_test, [], x_gpsindy_test, [] ) 
    
    # ----------------------- #
    # plot 
    
    f = plot_err_train_test( data_pred_train, data_pred_test, data_train, data_test, λ, csv_file)     
    display(f) 
    
end 















## ============================================ ##
## ============================================ ##
# plotting fns 

function plot_err_train_test( data_pred_train, data_pred_test, data_train, data_test, λ, csv_file) 
    
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

    ax_text = "λ = $λ, \n $csv_file" 
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
    
    dx_gp_err_train      = @sprintf "%.3g" norm( dx_train_noise[:,i_x] - dx_train_GP[:,i_x] ) 
    dx_sindy_err_train   = @sprintf "%.3g" norm( dx_train_noise[:,i_x] - dx_sindy[:,i_x] ) 
    dx_gpsindy_err_train = @sprintf "%.3g" norm( dx_train_noise[:,i_x] - dx_gpsindy[:,i_x] ) 
    
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
    
        Legend( f[i_x,5], [noise, GP, sindy, gpsindy], ["noise", "GP", "sindy", "gpsindy"], )  

    return f 
end 



# ax = Axis( f[i_x,1:2], title = "train dx$i_x err: GP = $dx_gp_err_train, \n SINDy = $dx_sindy_err_train, GPSINDy = $dx_gpsindy_err_train" )
#     noise   = CairoMakie.scatter!( ax, t_train, data_train.dx_noise[:,i_x], color = :black, label = "noise" ) 
#     GP      = lines!( ax, t_train, dx_train_GP[:,i_x], color = :red, label = "GP" )
#     sindy   = lines!( ax, t_train, dx_sindy[:,i_x], label = "sindy") 
#     gpsindy = lines!( ax, t_train, dx_gpsindy[:,i_x], label = "gpsindy" ) 
