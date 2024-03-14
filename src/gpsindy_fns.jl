
## ============================================ ##
# smooth training and test data with GPs 

export gp_train_test 
function gp_train_test( data_train, data_test, σ_n = 0.1, opt_σn = true ) 

    # first - smooth measurements with Gaussian processes 
    x_train_GP  = gp_post( data_train.t, 0*data_train.x_noise, data_train.t, 0 * data_train.x_noise, data_train.x_noise, σ_n, opt_σn ) 
    dx_train_GP = gp_post( x_train_GP, 0*data_train.dx_noise, x_train_GP, 0 * data_train.dx_noise, data_train.dx_noise, σ_n, opt_σn ) 
    x_test_GP   = gp_post( data_test.t, 0*data_test.x_noise, data_test.t, 0 * data_test.x_noise, data_test.x_noise, σ_n, opt_σn ) 
    dx_test_GP  = gp_post( x_test_GP, 0*data_test.dx_noise, x_test_GP, 0 * data_test.dx_noise, data_test.dx_noise, σ_n, opt_σn ) 

    return x_train_GP, dx_train_GP, x_test_GP, dx_test_GP 
end 


## ============================================ ##

export cross_validate_csv_path 

function cross_validate_csv_path( freq_hz, noise, σn, opt_σn, plot_option = false )  

    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

    csv_files_vec, save_path, save_path_fig, save_path_dfs = mkdir_save_path( csv_path ) 
    
    # create dataframe to store results 
    header = [ "csv_file", "λ_min", "train_err", "test_err", "train_traj", "test_traj" ] 
    df_min_err_csvs_sindy   = DataFrame( fill( [], 6 ), header ) 
    df_min_err_csvs_gpsindy = DataFrame( fill( [], 6 ), header ) 
    
    # loop 
    for i_csv in eachindex( csv_files_vec ) 
    
        csv_path_file = csv_files_vec[i_csv] 
    
        df_min_err_sindy, df_min_err_gpsindy, f_train, f_test = cross_validate_csv_path_file( csv_path_file, save_path_dfs, σn, opt_σn, freq_hz, noise ) 
        if plot_option == true 
            display(f_train) ; display(f_test) 
        end 
    
        # save figs 
        csv_file = replace( split( csv_path_file, "/" )[end], ".csv" => "" )  
        save( string( save_path_fig, csv_file, "_train.png" ), f_train )  
        save( string( save_path_fig, csv_file, "_test.png" ), f_test )  
    
        push!( df_min_err_csvs_sindy, df_min_err_sindy[1,:] ) 
        push!( df_min_err_csvs_gpsindy, df_min_err_gpsindy[1,:] ) 
    
    end 
    
    # save df_min_err for gpsindy and sindy 
    CSV.write( string( save_path, "df_min_err_csvs_sindy.csv" ), df_min_err_csvs_sindy ) 
    CSV.write( string( save_path, "df_min_err_csvs_gpsindy.csv" ), df_min_err_csvs_gpsindy ) 

    return df_min_err_csvs_sindy, df_min_err_csvs_gpsindy 
end 


## ============================================ ##

export cross_validate_csv_path_file 

function cross_validate_csv_path_file( csv_path_file, save_path_dfs, σn, opt_σn, freq_hz, noise ) 

    # extract data 
    data_train, data_test = car_data_struct( csv_path_file ) 

    x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test, σn, opt_σn ) 

    # cross-validate gpsindy 
    λ_vec      = λ_vec_fn() 
    header     = [ "λ", "train_err", "test_err", "train_traj", "test_traj" ] 
    df_gpsindy = DataFrame( fill( [], 5 ), header ) 
    df_sindy   = DataFrame( fill( [], 5 ), header ) 
    for i_λ = eachindex( λ_vec ) 

        λ = λ_vec[i_λ] 
        
        # sindy!!! 
        x_sindy_train, x_sindy_test = sindy_lasso_int( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 
        push!( df_sindy, [ λ, norm( data_train.x_noise - x_sindy_train ),  norm( data_test.x_noise - x_sindy_test ), x_sindy_train, x_sindy_test ] ) 

        # gpsindy!!! 
        x_gpsindy_train, x_gpsindy_test = sindy_lasso_int( x_train_GP, dx_train_GP, λ, data_train, data_test ) 
        push!( df_gpsindy, [ λ, norm( data_train.x_noise - x_gpsindy_train ),  norm( data_test.x_noise - x_gpsindy_test ), x_gpsindy_train, x_gpsindy_test ] ) 

    end 

    # save df_sindy and df_gpsindy 
    csv_string = split( csv_path_file, "/" )
    csv_file   = replace( csv_string[end], ".csv" => "" )   
    CSV.write( string( save_path_dfs, csv_file, "_df_sindy.csv" ), df_sindy ) 
    CSV.write( string( save_path_dfs, csv_file, "_df_gpsindy.csv" ), df_gpsindy ) 

    # save gpsindy min err stats 
    df_min_err_sindy   = df_min_err_fn( df_sindy, csv_path_file ) 
    df_min_err_gpsindy = df_min_err_fn( df_gpsindy, csv_path_file ) 

    # plot 
    f_train = plot_train( data_train, x_train_GP, df_min_err_sindy, df_min_err_gpsindy, σn, opt_σn, freq_hz, noise, csv_file ) 
    f_test  = plot_test( data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, σn, opt_σn, freq_hz, noise, csv_file )  

    return df_min_err_sindy, df_min_err_gpsindy, f_train, f_test 
end 


## ============================================ ##

export cross_validate_csv_path

function cross_validate_csv_path( csv_path, freq_hz, plot_option = false ) 
    
    # create save path 
    csv_files_vec, save_path, save_path_fig, save_path_dfs = mkdir_save_path( csv_path ) 

    λ_vec = λ_vec_fn() 

    header = [ "csv_file", "λ_min_sindy", "x_sindy_train_err", "x_sindy_test_err", "λ_min_gpsindy", "x_gpsindy_train_err", "x_gpsindy_test_err" ] 
    df_min_err_hist = DataFrame( fill( [], 7 ), header ) 

    for i in eachindex(csv_files_vec) 

        csv_file_path = csv_files_vec[i] 
        csv_file = replace( csv_file_path, csv_path => "") 
    
        x_err_hist = cross_validate( csv_path, csv_file_path, freq_hz, false ) 
    
        df_λ_vec, df_sindy, df_gpsindy = df_metrics( x_err_hist, λ_vec, csv_path, csv_file_path ) 

        if plot_option == true 
            
            f = plot_λ_err_log( λ_vec, df_λ_vec, df_sindy, df_gpsindy, freq_hz, csv_file ) 

            # save fig 
            fig_save = replace( csv_file, ".csv" => "_λ_err_log.png" ) 
            fig_save = string( save_path_fig, fig_save )  
            save( fig_save, f ) 

        end 
        
        data_sindy = Matrix( df_sindy ) ; data_gpsindy = Matrix( df_gpsindy ) 
        push!( df_min_err_hist, [ data_sindy data_gpsindy[:,2:end] ] )  
    
    end 

    # save df_min_err_hist as CSV 
    CSV.write( string( save_path, "df_min_err_hist.csv" ), df_min_err_hist ) 

    return df_min_err_hist 
end 


## ============================================ ##

export cross_validate 

function cross_validate( csv_path, csv_path_file, freq_hz, plot_option = false ) 

    # extract data 
    data_train, data_test = car_data_struct( csv_path_file ) 

    # smooth with GPs 
    x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 

    λ_vec = λ_vec_fn() 

    # ----------------------- #
    # test cross_validate_λ for sindy and gpsindy 

    x_err_hist = x_train_test_err_struct( Float64[], Float64[], Float64[], Float64[] ) 
    for i_λ = eachindex( λ_vec ) 

        λ = λ_vec[i_λ] 

        data_pred_train, data_pred_test = sindy_gpsindy_λ( data_train, data_test, x_train_GP, dx_train_GP, x_test_GP, λ ) 
        
        # plot and save metrics 
        if plot_option == true     
            csv_string = replace( csv_path_file, csv_path => "" ) 
            f = plot_err_train_test( data_pred_train, data_pred_test, data_train, data_test, λ, freq_hz, csv_string)     
            display(f) 
        end 

        x_err_hist = push_err_metrics( x_err_hist, data_train, data_test, data_pred_train, data_pred_test ) 
        
    end 

    return x_err_hist
end 

## ============================================ ##

export sindy_gpsindy_λ 

function sindy_gpsindy_λ( data_train, data_test, x_train_GP, dx_train_GP, x_test_GP, λ ) 

    # get sizes 
    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 
    
    # get x0 from noisy and smoothed data 
    x0_train    = data_train.x_noise[1,:] ; x0_train_GP = x_train_GP[1,:] 
    x0_test     = data_test.x_noise[1,:]  ; x0_test_GP  = x_test_GP[1,:] 
    
    # ----------------------- # 
    # SINDY-lasso ! 
    Ξ_sindy = sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 
    
    # integrate discovered dynamics 
    dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
    x_sindy_train = integrate_euler( dx_fn_sindy, x0_train, data_train.t, data_train.u ) 
    x_sindy_test  = integrate_euler( dx_fn_sindy, x0_test, data_test.t, data_test.u ) 
    
    # ----------------------- #
    # GPSINDy-lasso ! 
    Ξ_gpsindy  = sindy_lasso( x_train_GP, dx_train_GP, λ, data_train.u ) 
    
    # integrate discovered dynamics 
    dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
    x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train, data_train.t, data_train.u ) 
    x_gpsindy_test  = integrate_euler( dx_fn_gpsindy, x0_test, data_test.t, data_test.u ) 
    
    # ----------------------- #
    # collect data 
    
    data_pred_train = data_predicts( x_train_GP, dx_train_GP, x_sindy_train, [], x_gpsindy_train, [] ) 
    data_pred_test  = data_predicts( x_test_GP, [], x_sindy_test, [], x_gpsindy_test, [] ) 

    return data_pred_train, data_pred_test 
end 





















## ============================================ ##
# run cross validation, save plots, and return metrics 

function cross_validate_all_csvs( csv_path, save_path ) 

    csv_files_vec = readdir( csv_path ) 

    # if csv_files_vec contains "fig", remove from vector 
    deleteat!( csv_files_vec, findall( csv_files_vec .== "figs" ) )  

    # add entire path to csv_files_vec 
    for i in eachindex(csv_files_vec)  
        csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
    end 
    
    # check if save_path exists 
    if !isdir( save_path ) 
        mkdir( save_path ) 
    end 
    fig_save_path = string( save_path, "figs/" ) 
    if !isdir( fig_save_path ) 
        mkdir( fig_save_path ) 
    end 
    
    x_err_train = x_err_struct( Float64[], Float64[], Float64[], Float64[] ) 
    x_err_test  = x_err_struct( Float64[], Float64[], Float64[], Float64[] ) 
    for i = eachindex( csv_files_vec ) 
    # for i = [ 4 ] 
        # i = 42 
        csv_file = csv_files_vec[i] 

        t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_lasso, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy, fig_train, fig_test = cross_validate_sindy_gpsindy( csv_file, 1 ) 
    
        # t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_lasso, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy, fig_train, fig_test = cross_validate_sindy_gpsindy_traindouble( csv_file, 1 ) 
    
        push!( x_err_train.sindy_lasso, norm( x_train_noise - x_train_sindy ) ) 
        push!( x_err_train.gpsindy,     norm( x_train_noise - x_train_gpsindy ) ) 
        push!( x_err_test.sindy_lasso,  norm( x_test_noise - x_test_sindy ) ) 
        push!( x_err_test.gpsindy,      norm( x_test_noise - x_test_gpsindy ) ) 
    
        # save fig_train 
        fig_train_save = replace( csv_file, csv_path => fig_save_path )  
        fig_train_save = replace( fig_train_save, ".csv" => "_train.png" ) 
        save( fig_train_save, fig_train ) 
    
        # save fig_test 
        fig_test_save = replace( csv_file, csv_path => fig_save_path )  
        fig_test_save = replace( fig_test_save, ".csv" => "_test.png" ) 
        save( fig_test_save, fig_test ) 
        
    end 

    # reject 3-sigma outliers 
    sindy_3sigma_mean_train   = mean( filter( !isnan, reject_outliers( x_err_train.sindy_lasso ) ) ) 
    gpsindy_3sigma_mean_train = mean( filter( !isnan, reject_outliers( x_err_train.gpsindy ) ) ) 
    sindy_3sigma_mean_test    = mean( filter( !isnan, reject_outliers( x_err_test.sindy_lasso ) ) ) 
    gpsindy_3sigma_mean_test  = mean( filter( !isnan, reject_outliers( x_err_test.gpsindy ) ) ) 

    # save x_err_hist as CSV 
    header = [ "sindy_train", "gpsindy_train", "sindy_test", "gpsindy_test" ] 
    df     = DataFrame( [ x_err_train.sindy_lasso, x_err_train.gpsindy, x_err_test.sindy_lasso, x_err_test.gpsindy ], header ) 
    CSV.write( string( save_path, "err_x_hist.csv" ), df ) 

    # save 3-sigma mean as CSV 
    header = [ "sindy_3sigma_mean_train", "gpsindy_3sigma_mean_train", "sindy_3sigma_mean_test", "gpsindy_3sigma_mean_test" ] 
    df     = DataFrame( [ [ sindy_3sigma_mean_train ], [ gpsindy_3sigma_mean_train ], [ sindy_3sigma_mean_test ], [ gpsindy_3sigma_mean_test ] ], header )
    CSV.write( string( save_path, "err_3sigma_mean.csv" ), df ) 

    return sindy_3sigma_mean_test, gpsindy_3sigma_mean_test 
end 

export cross_validate_all_csvs 



## ============================================ ##

export cross_validate_sindy_gpsindy_traindouble

function cross_validate_sindy_gpsindy_traindouble( csv_file, plot_option = false ) 

    data_train, data_test = car_data_struct( csv_file ) 

    # smooth with GPs 
    t_train_GP, u_train_GP, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_double_test( data_train, data_test ) 
    
    # get x0 from smoothed data 
    x0_train = data_train.x_noise[1,:] ; x0_train_GP = x_train_GP[1,:] 
    # x0_test     = x_test_GP[1,:] 
    x0_test  = data_test.x_noise[1,:]  ; x0_test_GP  = x_test_GP[1,:] 

    # cross-validate SINDy!!! 
    λ_vec = λ_vec_fn() 
    Ξ_sindy_vec, err_x_sindy, err_dx_sindy = cross_validate_λ( data_train.t, data_train.x_noise, data_train.dx_noise, data_train.u, λ_vec ) 
    
    # save ξ with smallest x error  
    Ξ_sindy_lasso = Ξ_minerr( Ξ_sindy_vec, err_x_sindy ) 
    
    # build dx_fn from Ξ and integrate 
    x_train_sindy, x_test_sindy = dx_Ξ_integrate( data_train, data_test, Ξ_sindy_lasso, x0_train, x0_test ) 
    
    # cross-validate GPSINDy!!!  
    λ_vec = λ_vec_fn() 
    Ξ_gpsindy_vec, err_x_gpsindy, err_dx_gpsindy = cross_validate_λ( t_train_GP, x_train_GP, dx_train_GP, u_train_GP, λ_vec ) 
    
    # save ξ with smallest x error  
    Ξ_gpsindy = Ξ_minerr( Ξ_gpsindy_vec, err_x_gpsindy ) 
    
    # build dx_fn from Ξ and integrate 
    x_train_gpsindy, x_test_gpsindy = dx_Ξ_integrate( data_train, data_test, Ξ_gpsindy, x0_train, x0_test ) 

    # downsample x_train_GP (again) 
    x_train_GP = x_train_GP[1:2:end,:] 

    # plot 
    if plot_option == 1 

        # plot training 
        # fig_train = plot_noise_GP_sindy_gpsindy_traindouble( data_train.t, data_train.x_noise, t_train_GP, x_train_GP, x_train_sindy, x_train_gpsindy, "training data" ) 
        fig_train = plot_noise_GP_sindy_gpsindy( data_train.t, data_train.x_noise, x_train_GP, x_train_sindy, x_train_gpsindy, "training data" ) 

        # plot testing 
        fig_test = plot_noise_GP_sindy_gpsindy( data_test.t, data_test.x_noise, x_test_GP, x_test_sindy, x_test_gpsindy, "testing data" ) 

    end 

    return data_train.t, data_test.t, data_train.x_noise, data_test.x_noise, Ξ_sindy_lasso, x_train_sindy, x_test_sindy, Ξ_gpsindy, x_train_gpsindy, x_test_gpsindy, fig_train, fig_test 
  
end 



## ============================================ ##

export cross_validate_sindy_gpsindy

function cross_validate_sindy_gpsindy( csv_file, plot_option = false ) 

    data_train, data_test = car_data_struct( csv_file ) 

    # smooth with GPs 
    x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 
    
    # get x0 from smoothed data 
    x0_train    = data_train.x_noise[1,:] ; x0_train_GP = x_train_GP[1,:] 
    # x0_test     = x_test_GP[1,:] 
    x0_test     = data_test.x_noise[1,:]  ; x0_test_GP  = x_test_GP[1,:] 

    # ----------------------- #
    
    # cross-validate SINDy!!! 
    λ_vec = λ_vec_fn() 
    Ξ_sindy_vec, err_x_sindy, err_dx_sindy = cross_validate_λ( data_train.t, data_train.x_noise, data_train.dx_noise, data_train.u, λ_vec ) 
    
    # save ξ with smallest x error  
    Ξ_sindy_lasso = Ξ_minerr( Ξ_sindy_vec, err_x_sindy ) 
    
    # build dx_fn from Ξ and integrate 
    x_train_sindy, x_test_sindy = dx_Ξ_integrate( data_train, data_test, Ξ_sindy_lasso, x0_train, x0_test ) 
    
    # ----------------------- #
    
    # cross-validate GPSINDy!!!  
    λ_vec = λ_vec_fn() 
    Ξ_gpsindy_vec, err_x_gpsindy, err_dx_gpsindy = cross_validate_λ( data_train.t, x_train_GP, dx_train_GP, data_train.u, λ_vec ) 
    
    # save ξ with smallest x error  
    Ξ_gpsindy = Ξ_minerr( Ξ_gpsindy_vec, err_x_gpsindy ) 
    
    # build dx_fn from Ξ and integrate 
    x_train_gpsindy, x_test_gpsindy = dx_Ξ_integrate( data_train, data_test, Ξ_gpsindy, x0_train_GP, x0_test ) 

    # ----------------------- #
    
    # plot 
    if plot_option == 1 

        # plot training 
        fig_train = plot_noise_GP_sindy_gpsindy( data_train.t, data_train.x_noise, x_train_GP, x_train_sindy, x_train_gpsindy, "training data" ) 

        # plot testing 
        fig_test = plot_noise_GP_sindy_gpsindy( data_test.t, data_test.x_noise, x_test_GP, x_test_sindy, x_test_gpsindy, "testing data" ) 

    end 

    return data_train.t, data_test.t, data_train.x_noise, data_test.x_noise, Ξ_sindy_lasso, x_train_sindy, x_test_sindy, Ξ_gpsindy, x_train_gpsindy, x_test_gpsindy, fig_train, fig_test 
  
end 


## ============================================ ##
# cross-validate λ 

export cross_validate_λ 
function cross_validate_λ( t_train, x_train, dx_train, u_train, λ_vec ) 

    # get sizes 
    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x_train, u_train ) 
    
    # build function library from smoothed data 
    Θx_gp  = pool_data_test( [ x_train u_train ], n_vars, poly_order ) 
    
    # get x0 from smoothed data
    x0_train_GP = x_train[1,:] 

    err_x_vec     = [] 
    err_dx_vec    = [] 
    Ξ_gpsindy_vec = [] 

    # loop through each state 
    for j = 1 : x_vars 
    
        err_xj_vec    = [] 
        err_dxj_vec   = [] 
        ξ_gpsindy_vec = []
    
        # CROSS-VALIDATION 
        for i = eachindex(λ_vec) 
        
            λ = λ_vec[i] 
            # println( "x", j, ": λ = ", λ ) 
    
            # GPSINDy-lasso ! 
            Ξ_gpsindy  = sindy_lasso( x_train, dx_train, λ, u_train ) 
            dx_gpsindy = Θx_gp * Ξ_gpsindy 
    
            if sum(Ξ_gpsindy[:,j]) == 0 
                break 
            end 
    
            # integrate discovered dynamics 
            dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
            x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train_GP, t_train, u_train ) 

            # save ξ coefficients 
            ξ = Ξ_gpsindy[:,j] 
            push!( ξ_gpsindy_vec, ξ ) 
    
            # push error stuff 
            err_xj_norm  = norm( x_gpsindy_train[:,j] - x_train[:,j] ) 
            err_dxj_norm = norm( dx_gpsindy[:,j] - dx_train[:,j] ) 
            push!( err_xj_vec, err_xj_norm ) 
            push!( err_dxj_vec, err_dxj_norm ) 
    
        end 
    
        push!( err_x_vec, err_xj_vec ) 
        push!( err_dx_vec, err_dxj_vec ) 
        push!( Ξ_gpsindy_vec, ξ_gpsindy_vec ) 
    
    end 

    return Ξ_gpsindy_vec, err_x_vec, err_dx_vec 
end 


## ============================================ ##
# smooth training and test data with GPs 

export gp_train_double_test 
function gp_train_double_test( data_train, data_test ) 

    # let's double the points 
    t_train = data_train.t 
    dt = t_train[2] - t_train[1] 

    t_train_double = Float64[ ] 
    for i in eachindex(t_train) 
        push!( t_train_double, t_train[i] ) 
        push!( t_train_double, t_train[i] + dt/2 ) 
    end 

    x_col, x_row = size( data_train.x_noise ) 
    u_col, u_row = size( data_train.u ) 

    # first - smooth training data with Gaussian processes 
    x_train_GP  = gp_post( t_train_double, zeros( 2 * x_col, x_row ), data_train.t, 0*data_train.x_noise, data_train.x_noise ) 
    dx_train_GP = gp_post( x_train_GP, zeros( 2 * x_col, x_row ), data_train.x_noise, 0*data_train.dx_noise, data_train.dx_noise ) 
    # u_train_GP  = gp_post( t_train_double, zeros( 2 * u_col, u_row ), data_train.t, 0*data_train.u, data_train.u ) 

    # linearly interpolate u 
    u_train_GP = [ ]
    for i = 1 : size(data_train.u, 1) - 1 
        du = ( data_train.u[i+1,:] - data_train.u[i,:] ) / 2 
        push!( u_train_GP, data_train.u[i,:] ) 
        push!( u_train_GP, data_train.u[i,:] + du ) 
    end 
    push!( u_train_GP, data_train.u[end,:] )  
    push!( u_train_GP, data_train.u[end,:] )  
    u_train_GP = vv2m( u_train_GP )

    # smooth testing data 
    x_test_GP   = gp_post( data_test.t, 0*data_test.x_noise, data_test.t, 0*data_test.x_noise, data_test.x_noise ) 
    dx_test_GP  = gp_post( x_test_GP, 0*data_test.dx_noise, x_test_GP, 0*data_test.dx_noise, data_test.dx_noise ) 

    return t_train_double, u_train_GP, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP 
end 






















## ============================================ ##
## ============================================ ##
# compare sindy, gpsindy, and gpsindy_gpsindy 

export sindy_nn_gpsindy
function sindy_nn_gpsindy( fn, noise, λ, Ξ_hist, Ξ_err_hist, x_hist, x_err_hist ) 
    
    # generate true states 
    x0, dt, t, x_true, dx_true, dx_fd, p, u = ode_states(fn, 0, 2)
    if u == false 
        u_train = false ; u_test  = false 
    else 
        u_train, u_test = split_train_test(u, 0.2, 5) 
    end 

    # truth coeffs 
    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x_true, u ) 

    poly_order = 1 
    Ξ_true = sindy_stls(x_true, dx_true, λ, u)

    # add noise 
    println("noise = ", noise)
    x_noise  = x_true + noise * randn(size(x_true, 1), size(x_true, 2))
    dx_noise = dx_true + noise * randn(size(dx_true, 1), size(dx_true, 2))
    # dx_noise = dx_fd 

    # split into training and test data 
    test_fraction = 0.2
    portion = 5
    t_train, t_test               = split_train_test(t, test_fraction, portion)
    x_train_true, x_test_true     = split_train_test(x_true, test_fraction, portion)
    dx_train_true, dx_test_true   = split_train_test(dx_true, test_fraction, portion)
    x_train_noise, x_test_noise   = split_train_test(x_noise, test_fraction, portion)
    dx_train_noise, dx_test_noise = split_train_test(dx_noise, test_fraction, portion)

    # ----------------------- # 
    # standardize  
    x_stand_noise  = stand_data(t_train, x_train_noise)
    x_stand_true   = stand_data(t_train, x_train_true)
    dx_stand_true  = dx_true_fn(t_train, x_stand_true, p, fn)
    dx_stand_noise = dx_stand_true + noise * randn(size(dx_stand_true, 1), size(dx_stand_true, 2))

    # set training data for GPSINDy 
    # x_train  = x_stand_noise 
    # dx_train = dx_stand_noise

    x_train  = x_train_noise 
    dx_train = dx_train_noise 

    ## ============================================ ##
    # SINDy vs. NN vs. GPSINDy 

    # SINDy by itself 
    Ξ_sindy_stls  = sindy_stls(x_train, dx_train, λ, u_train)
    Ξ_sindy_lasso = sindy_lasso(x_train, dx_train, λ, u_train)

    # GPSINDy (first) 
    x_train_GP  = gp_post(t_train, 0 * x_train, t_train, 0 * x_train, x_train)
    dx_train_GP = gp_post(x_train_GP, 0 * dx_train, x_train_GP, 0 * dx_train, dx_train)
    Ξ_gpsindy   = sindy_lasso(x_train_GP, dx_train_GP, λ, u_train)

    # Train NN on the data
    Ξ_nn_lasso = nn_lasso_Ξ_fn( dx_train, x_train, λ, u_train ) 

    # ----------------------- # 
    # validate data 

    x_vars = size(x_true, 2)
    # dx_fn_true        = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true ) 
    dx_fn_sindy_stls  = build_dx_fn(poly_order, x_vars, u_vars, Ξ_sindy_stls)
    dx_fn_sindy_lasso = build_dx_fn(poly_order, x_vars, u_vars, Ξ_sindy_lasso)
    dx_fn_gpsindy     = build_dx_fn(poly_order, x_vars, u_vars, Ξ_gpsindy)
    dx_fn_nn          = build_dx_fn(poly_order, x_vars, u_vars, Ξ_nn_lasso)
    
    # # integrate !! 
    x0 = x_test_true[1,:] 
    # x_test_true        = integrate_euler( dx_fn_true, x0, t_test, u_test )
    x_sindy_stls_test  = integrate_euler( dx_fn_sindy_stls, x0, t_test, u_test ) 
    x_sindy_lasso_test = integrate_euler( dx_fn_sindy_lasso, x0, t_test, u_test ) 
    x_nn_test          = integrate_euler( dx_fn_nn, x0, t_test, u_test ) 
    x_gpsindy_test     = integrate_euler( dx_fn_gpsindy, x0, t_test, u_test ) 
    
    # # COMMENT OUT / DELETE THIS 
    # x0 = x_train_true[1,:] 
    # x_true             = integrate_euler( dx_fn_true, x0, t, u )
    # x_sindy_stls_test  = integrate_euler( dx_fn_sindy_stls, x0, t, u ) 
    # x_sindy_lasso_test = integrate_euler( dx_fn_sindy_lasso, x0, t, u ) 
    # x_nn_test          = integrate_euler( dx_fn_nn, x0, t, u ) 
    # x_gpsindy_test     = integrate_euler( dx_fn_gpsindy, x0, t, u ) 

    # ----------------------- # 
    # save outputs  

    # save Ξ_hist 
    push!( Ξ_hist.truth,       Ξ_true ) 
    push!( Ξ_hist.sindy_stls,  Ξ_sindy_stls ) 
    push!( Ξ_hist.sindy_lasso, Ξ_sindy_lasso ) 
    push!( Ξ_hist.gpsindy,     Ξ_gpsindy ) 
    push!( Ξ_hist.nn,          Ξ_nn_lasso ) 

    # save x_hist 
    push!( x_hist.t,            t_test ) 
    push!( x_hist.truth,        x_test_true ) 
    push!( x_hist.sindy_stls,   x_sindy_stls_test  ) 
    push!( x_hist.sindy_lasso,  x_sindy_lasso_test  ) 
    push!( x_hist.gpsindy,      x_gpsindy_test ) 
    push!( x_hist.nn,           x_nn_test ) 

    # # COMMENT OUT / DELETE 
    # push!( x_hist.t,            t ) 
    # push!( x_hist.truth,        x_true) 
    # push!( x_hist.sindy_stls,   x_sindy_stls_test  ) 
    # push!( x_hist.sindy_lasso,  x_sindy_lasso_test  ) 
    # push!( x_hist.gpsindy,      x_gpsindy_test ) 
    # push!( x_hist.nn,           x_nn_test ) 

    # save Ξ_err_hist 
    push!( Ξ_err_hist.sindy_stls,  norm( Ξ_true - Ξ_sindy_stls ) ) 
    push!( Ξ_err_hist.sindy_lasso, norm( Ξ_true - Ξ_sindy_lasso ) ) 
    push!( Ξ_err_hist.gpsindy,     norm( Ξ_true - Ξ_gpsindy ) ) 
    push!( Ξ_err_hist.nn,          norm( Ξ_true - Ξ_nn_lasso ) ) 

    # save Ξ_err_hist 
    push!( x_err_hist.sindy_stls,  norm( x_test_true - x_sindy_stls_test ) ) 
    push!( x_err_hist.sindy_lasso, norm( x_test_true - x_sindy_lasso_test ) ) 
    push!( x_err_hist.gpsindy,     norm( x_test_true - x_gpsindy_test ) ) 
    push!( x_err_hist.nn,          norm( x_test_true - x_nn_test ) ) 

    # # COMMENT OUT / DELETE 
    # push!( x_err_hist.sindy_stls,  norm( x_true - x_sindy_stls_test ) ) 
    # push!( x_err_hist.sindy_lasso, norm( x_true - x_sindy_lasso_test ) ) 
    # push!( x_err_hist.gpsindy,     norm( x_true - x_gpsindy_test ) ) 
    # push!( x_err_hist.nn,          norm( x_true - x_nn_test ) ) 
    
    return Ξ_hist, Ξ_err_hist, x_hist, x_err_hist

end 



## ============================================ ##

export cross_validate_gpsindy
function cross_validate_gpsindy( csv_file, plot_option = false ) 

    data_train, data_test = car_data_struct( csv_file ) 

    # smooth with GPs 
    x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 
    
    # get x0 from smoothed data 
    x0_train_GP = x_train_GP[1,:] 
    x0_test_GP  = x_test_GP[1,:] 
    
    # SINDy 
    # λ = 0.1 
    # Ξ_sindy_stls  = sindy_stls( data_train.x_noise, data_train.dx_noise, λ, false, data_train.u ) 
    # Ξ_sindy_lasso  = sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 

    # cross-validate SINDy!!! 
    λ_vec = λ_vec_fn() 
    Ξ_sindy_vec, err_x_sindy, err_dx_sindy = cross_validate_λ( data_train.t, data_train.x_noise, data_train.dx_noise, data_train.u, λ_vec ) 
    
    # save ξ with smallest x error  
    Ξ_sindy_lasso = Ξ_minerr( Ξ_sindy_vec, err_x_sindy ) 
    
    # build dx_fn from Ξ and integrate 
    x_train_sindy, x_test_sindy = dx_Ξ_integrate( data_train, data_test, Ξ_sindy_lasso, x0_train_GP, x0_test_GP )
    
    # cross-validate GPSINDy!!!  
    λ_vec = λ_vec_fn() 
    Ξ_gpsindy_vec, err_x_gpsindy, err_dx_gpsindy = cross_validate_λ( data_train.t, x_train_GP, dx_train_GP, data_train.u, λ_vec ) 
    
    # save ξ with smallest x error  
    Ξ_gpsindy = Ξ_minerr( Ξ_gpsindy_vec, err_x_gpsindy ) 
    
    # build dx_fn from Ξ and integrate 
    x_train_gpsindy, x_test_gpsindy = dx_Ξ_integrate( data_train, data_test, Ξ_gpsindy, x0_train_GP, x0_test_GP ) 

    # plot 
    if plot_option == 1 

        # plot training 
        plot_noise_sindy_gpsindy( data_train.t, data_train.x_noise, x_train_GP, x_train_sindy, x_train_gpsindy, "training data" ) 

        # plot testing 
        plot_noise_sindy_gpsindy( data_test.t, data_test.x_noise, x_test_GP, x_test_sindy, x_test_gpsindy, "testing data" ) 

    end 

    return data_train.t, data_test.t, data_train.x_noise, data_test.x_noise, Ξ_sindy_lasso, x_train_sindy, x_test_sindy, Ξ_gpsindy, x_train_gpsindy, x_test_gpsindy 
end 


## ============================================ ##

export x_Ξ_fn 
function x_Ξ_fn( data_train, data_test, data_train_stand )

    λ = 0.1 
    
    # truth 
    Ξ_true_stls       = sindy_stls( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
    Ξ_true_stls_terms = pretty_coeffs( Ξ_true_stls, data_train.x_true, data_train.u ) 
    
    # SINDy and GPSINDy 
    Ξ_sindy_stls, Ξ_sindy_lasso, Ξ_gpsindy, Ξ_sindy_stls_terms, Ξ_sindy_lasso_terms, Ξ_gpsindy_terms = gpsindy_Ξ_fn( data_train_stand.t, data_train_stand.x_noise, data_train_stand.dx_noise, λ, data_train_stand.u ) 
    
    # NN 
    Ξ_nn_lasso = nn_lasso_Ξ_fn( data_train.dx_noise, data_train.x_noise , λ, data_train.u ) 
    
    # ----------------------- # 
    # validate 
    
    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 
    
    dx_fn_nn          = build_dx_fn( poly_order, x_vars, u_vars, Ξ_nn_lasso ) 
    dx_fn_true        = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true_stls ) 
    dx_fn_sindy_stls  = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
    dx_fn_sindy_lasso = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_lasso ) 
    dx_fn_gpsindy     = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
    
    # integrate !! 
    x0 = data_test.x_true[1,:] 
    x_sindy_stls_test  = integrate_euler( dx_fn_sindy_stls, x0, data_test.t, data_test.u ) 
    x_sindy_lasso_test = integrate_euler( dx_fn_sindy_lasso, x0, data_test.t, data_test.u ) 
    x_nn_test          = integrate_euler( dx_fn_nn, x0, data_test.t, data_test.u ) 
    x_gpsindy_test     = integrate_euler( dx_fn_gpsindy, x0, data_test.t, data_test.u ) 
    
    
    # ----------------------- #
    # plot smoothed data and validation test data 
    plot_validation_test( data_test.t, data_test.x_true, x_sindy_stls_test, x_sindy_lasso_test, x_nn_test, x_gpsindy_test ) 
    
    # save outputs 
    Ξ = Ξ_struct( Ξ_true_stls, Ξ_sindy_stls, Ξ_sindy_lasso, Ξ_nn_lasso, Ξ_gpsindy ) 
    x = x_struct( data_test.t, data_test.x_true, x_sindy_stls_test, x_sindy_lasso_test, x_nn_test, x_gpsindy_test )  
    
    return x, Ξ 
end 


## ============================================ ##

export err_Ξ_x 
function err_Ξ_x( Ξ, x, Ξ_err, x_err ) 

    Ξ_true          = Ξ.truth 
    Ξ_sindy_stls    = Ξ.sindy_stls 
    Ξ_sindy_lasso   = Ξ.sindy_lasso 
    Ξ_nn_lasso      = Ξ.nn 
    Ξ_gpsindy       = Ξ.gpsindy 

    x_true          = x.truth 
    x_sindy_stls    = x.sindy_stls 
    x_sindy_lasso   = x.sindy_lasso 
    x_nn            = x.nn 
    x_gpsindy       = x.gpsindy 

    x_vars = size(x_true, 2) 

    # x 
    x_sindy_stls_err = [] 
    for i = 1 : x_vars 
        push!( x_sindy_stls_err, norm( x_true[:,i] - x_sindy_stls[:,i] )  ) 
    end 
    x_sindy_lasso_err = [] 
    for i = 1 : x_vars 
        push!( x_sindy_lasso_err, norm( x_true[:,i] - x_sindy_lasso[:,i] )  ) 
    end
    x_nn_err = [] 
    for i = 1 : x_vars 
        push!( x_nn_err, norm( x_true[:,i] - x_nn[:,i] )  ) 
    end 
    x_gpsindy_err = [] 
    for i = 1 : x_vars 
        push!( x_gpsindy_err, norm( x_true[:,i] - x_gpsindy[:,i] )  ) 
    end 
    push!( x_err.sindy_stls, x_sindy_stls_err ) 
    push!( x_err.sindy_lasso, x_sindy_lasso_err )
    push!( x_err.nn, x_nn_err ) 
    push!( x_err.gpsindy, x_gpsindy_err )

    # Ξ 
    Ξ_sindy_stls_err = [] 
    for i = 1 : x_vars 
        push!( Ξ_sindy_stls_err, norm( Ξ_true[:,i] - Ξ_sindy_stls[:,i] ) ) 
    end 
    Ξ_sindy_lasso_err = [] 
    for i = 1 : x_vars 
        push!( Ξ_sindy_lasso_err, norm( Ξ_true[:,i] - Ξ_sindy_lasso[:,i] ) ) 
    end 
    Ξ_nn_err = [] 
    for i = 1 : x_vars 
        push!( Ξ_nn_err, norm( Ξ_true[:,i] - Ξ_nn_lasso[:,i] ) ) 
    end 
    Ξ_gpsindy_err = [] 
    for i = 1 : x_vars 
        push!( Ξ_gpsindy_err, norm( Ξ_true[:,i] - Ξ_gpsindy[:,i] ) ) 
    end 
    push!( Ξ_err.sindy_stls, Ξ_sindy_stls_err ) 
    push!( Ξ_err.sindy_lasso, Ξ_sindy_lasso_err ) 
    push!( Ξ_err.nn, Ξ_nn_err ) 
    push!( Ξ_err.gpsindy, Ξ_gpsindy_err ) 

    return Ξ_err, x_err 
end 


## ============================================ ##
# return Ξ with minimum x error 

export Ξ_minerr 
function Ξ_minerr( Ξ_gpsindy_vec, err_x_vec ) 

    x_vars = size(Ξ_gpsindy_vec, 1) 
    p      = size(Ξ_gpsindy_vec[1][1], 1)

    Ξ_gpsindy_minerr = zeros(p, x_vars) 
    for i = 1 : x_vars 
    
        err_xi_vec = err_x_vec[i]
        ξi_vec     = Ξ_gpsindy_vec[i] 
    
        for i = eachindex(err_xi_vec) 
            if isnan(err_xi_vec[i])  
                err_xi_vec[i] = 1e10 
            end 
        end 
    
        # find index with smallest element 
        min_idx = argmin( err_xi_vec ) 
        ξi      = ξi_vec[min_idx] 
        println( "min_idx = ", min_idx ) 
    
        Ξ_gpsindy_minerr[:,i] = ξi 
    
    end     

    return Ξ_gpsindy_minerr 
end 

## ============================================ ##

export nn_lasso_Ξ_fn
# function gpsindy_Ξ_fn( t_train, x_train, dx_train, λ, u_train ) 
function nn_lasso_Ξ_fn( dx_noise, x_noise, λ, u_train ) 

    x_vars = size(x_noise, 2) 

    # Train NN on the data
    # Define the 2-layer MLP
    dx_noise_nn = 0 * dx_noise 
    for i = 1 : x_vars 
        dx_noise_nn[:, i] = train_nn_predict(x_noise, dx_noise[:, i], 100, x_vars)
    end 

    # Concanate the two outputs to make a Matrix
    Ξ_nn_lasso  = sindy_lasso( x_noise, dx_noise_nn, λ, u_train )

    return Ξ_nn_lasso 
end 

## ============================================ ##

export nn_stls_Ξ_fn
# function gpsindy_Ξ_fn( t_train, x_train, dx_train, λ, u_train ) 
function nn_stls_Ξ_fn( dx_noise, x_noise, λ, u_train ) 

    x_vars = size(x_noise, 2) 

    # Train NN on the data
    # Define the 2-layer MLP
    dx_noise_nn = 0 * dx_noise 
    for i = 1 : x_vars 
        dx_noise_nn[:, i] = train_nn_predict(x_noise, dx_noise[:, i], 100, x_vars)
    end 

    # Concanate the two outputs to make a Matrix
    Ξ_nn_stls  = sindy_lasso( x_noise, dx_noise_nn, λ, u_train )

    return Ξ_nn_stls
end 


## ============================================ ##

export gpsindy_Ξ_fn
function gpsindy_Ξ_fn( t_train, x_train, dx_train, λ, u_train ) 

    # get sizes 
    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x_train, u_train ) 

    # GP smooth data 
    x_GP_train  = gp_post( t_train, 0*x_train, t_train, 0*x_train, x_train ) 
    dx_GP_train = gp_post( x_GP_train, 0*dx_train, x_GP_train, 0*dx_train, dx_train ) 

    # run SINDy (STLS) 
    Ξ_sindy_stls       = sindy_lasso( x_train, dx_train, λ, u_train ) 
    Ξ_sindy_stls_terms = pretty_coeffs( Ξ_sindy_stls, x_train, u_train ) 

    # run SINDy (LASSO) 
    Ξ_sindy_lasso       = sindy_lasso( x_train, dx_train, λ, u_train ) 
    Ξ_sindy_lasso_terms = pretty_coeffs( Ξ_sindy_lasso, x_train, u_train ) 

    # run GPSINDy 
    Ξ_gpsindy       = sindy_lasso( x_GP_train, dx_GP_train, λ, u_train ) 
    Ξ_gpsindy_terms = pretty_coeffs( Ξ_gpsindy, x_GP_train, u_train ) 

    return Ξ_sindy_stls, Ξ_sindy_lasso, Ξ_gpsindy, Ξ_sindy_stls_terms, Ξ_sindy_lasso_terms, Ξ_gpsindy_terms  
end 


## ============================================ ##

export admm_lasso 
function admm_lasso( t, dx, x, (ξ, z, u), hp, λ, α, ρ, abstol, reltol, hist ) 
# ----------------------- #
# PURPOSE: 
#       Run one iteration of ADMM LASSO 
# INPUTS: 
#       t           : training data ordinates time  
#       dx          : training data ( f(x) )
#       x           : training data ( x ) 
#       (ξ, z, u)   : dynamics coefficients primary and dual vars 
#       hp          : hyperparameters 
#       λ           : L1 norm threshold  
#       α           : relaxation parameter 
#       ρ           : idk what this does, but Boyd sets it to 1 
#       print_vars  : option to display ξ, z, u, hp 
#       abstol      : abs tol 
#       reltol      : rel tol 
#       hist        : diagnostics struct 
# OUTPUTS: 
#       ξ           : output dynamics coefficients (ADMM primary variable x)
#       z           : output dynamics coefficients (ADMM primary variable z)
#       u           : input dual variable 
#       hp          : log-scaled hyperparameters 
#       hist        : diagnostis struct 
#       plt         : plot for checking performance of coefficients 
# ----------------------- #

    n_vars = size(x, 2) ; poly_order = n_vars 
    Θx     = pool_data_test( x, n_vars, poly_order ) 

    # objective fns 
    f_hp, g, aug_L = obj_fns( t, dx, x, λ, ρ )

    # hp-update (optimization) 
    # hp = opt_hp(t, dx, x, ξ) 

    # ξ-update 
    ξ = opt_ξ( aug_L, ξ, z, u, hp ) 
    
    # z-update (soft thresholding) 
    z_old = z 
    ξ_hat = α*ξ + (1 .- α)*z_old 
    z     = shrinkage( ξ_hat + u, λ/ρ ) 

    # u-update 
    u += (ξ_hat - z) 
    
    # ----------------------- #
    # plot 
    plt = scatter( t, dx, label = "train (noise)", c = :black, ms = 3 ) 
    plot!( plt, t, Θx*ξ, label = "Θx*ξ", c = :green, ms = 3 ) 

    # ----------------------- #
    # push diagnostics 
    n = length(ξ) 
    push!( hist.objval, f_hp(ξ, hp) + g(z) )
    push!( hist.fval, f_hp( ξ, hp ) )
    push!( hist.gval, g(z) ) 
    push!( hist.hp, hp )
    push!( hist.r_norm, norm(ξ - z) )
    push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
    push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(ξ), norm(-z)) ) 
    push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 
    
    return ξ, z, u, hp, hist, plt 
end 

## ============================================ ##

export gpsindy 
function gpsindy( t, dx_train, x, λ, α, ρ, abstol, reltol ) 
# ----------------------- # 
# PURPOSE: 
#       Main gpsindy function (iterate j = 1 : n_vars) 
# INPUTS: 
#       t       : training data ordinates ( x ) 
#       dx      : training data ( f(x) )
#       x       : training data ( x )  
#       λ       : L1 norm threshold  
#       α       : relaxation parameter 
#       ρ       : idk what this does, but Boyd sets it to 1 
#       abstol  : abs tol 
#       reltol  : rel tol 
# OUTPUTS: 
#       Ξ       : sparse dynamics coefficient (hopefully) 
#       hist    : diagnostics struct 
# ----------------------- # 

# set up 
n_vars = size(dx_train, 2) ; poly_order = n_vars 
Θx     = pool_data_test( x, n_vars, poly_order ) 
Ξ      = zeros( size(Θx, 2), size(dx_train, 2) ) 
hist_nvars = [] 
for j = 1 : n_vars 

    dx = dx_train[:,j] 
    
    # start animation 
    # a = Animation() 
    # plt = scatter( t, dx, label = "train (noise)", c = :black, ms = 3 ) 
    # plot!( plt, legend = :outerright, size = [800 300], title = string("Fitting ξ", j), xlabel = "Time (s)" ) 
    # frame(a, plt) 

    # ξ-update 
    n = size(Θx, 2); ξ = z = u = zeros(n) 
    f_hp, g, aug_L = obj_fns( t, dx, x, λ, ρ )
    hp = [1.0, 1.0, 0.1] 
    ξ  = opt_ξ( aug_L, ξ, z, u, hp ) 
    # dx_GP, Σ_dxsmooth, hp = post_dist_SE( x, x, dx )  
    println( "hp = ", hp ) 

    hist = Hist( [], [], [], [], [], [], [], [] )  

    # ----------------------- #
    # hp = opt_hp(t, dx, x, ξ) 
    # loop until convergence or max iter 
    for k = 1 : 1000  

        # ADMM LASSO! 
        z_old = z 
        ξ, z, u, hp, hist, plt = admm_lasso( t, dx, x, (ξ, z, u), hp, λ, α, ρ, abstol, reltol, hist )    

        # end condition 
        # if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
        if hist.r_norm[end] < abstol[end] && hist.s_norm[end] < abstol[end] 
            break 
        end 

    end 
    # plot!( plt, legend = :outerright, size = [800 300], title = string("Fitting ξ", j), xlabel = "Time (s)" )  
    # frame(a, plt) 

    # ----------------------- #
    # optimize HPs 
    hp = opt_hp( t, dx, x, ξ ) 
    # loop until convergence or max iter 
    for k = 1 : 1000  

        # ADMM LASSO! 
        z_old = z 
        ξ, z, u, hp, hist, plt = admm_lasso( t, dx, x, (ξ, z, u), hp, λ, α, ρ, abstol, reltol, hist )    

        # end condition 
        # if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
        if hist.r_norm[end] < abstol[end] && hist.s_norm[end] < abstol[end] 
            break 
        end 

    end 
    # plot!( plt, legend = :outerright, size = [800 300], title = string("Fitting ξ", j), xlabel = "Time (s)" )  
    # frame(a, plt) 

    # ----------------------- #
    # optimize HPs 
    hp = opt_hp(t, dx, x, ξ) 
    # loop until convergence or max iter 
    for k = 1 : 1000  

        # ADMM LASSO! 
        z_old = z 
        ξ, z, u, hp, hist, plt = admm_lasso( t, dx, x, (ξ, z, u), hp, λ, α, ρ, abstol, reltol, hist )    

        # end condition 
        # if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
        if hist.r_norm[end] < abstol[end] && hist.s_norm[end] < abstol[end] 
            break 
        end 

    end 
    # plot!( plt, legend = :outerright, size = [800 300], title = string("Fitting ξ", j), xlabel = "Time (s)" )  
    # frame(a, plt) 

    # push diagnostics 
    push!( hist_nvars, hist ) 
    Ξ[:,j] = z 

    # g = gif(a, fps = 2) 
    # display(g) 
    # display(plt) 
    
    end 

    return Ξ, hist_nvars 
end 


## ============================================ ##
# monte carlo gpsindy (with all different cases)

export monte_carlo_gpsindy 
function monte_carlo_gpsindy( fn, noise_vec, λ, abstol, reltol, case ) 
# ----------------------- #
# PURPOSE:  
#       Run GPSINDy monte carlo 
# INPUTS: 
#       fn              : ODE function 
#       noise_vec       : dx noise vector for iterations 
#       λ               : L1 norm threshold 
#       abstol          : abs tol 
#       reltol          : rel tol 
#       case            : 0 = true, 1 = noise, 2 = norm 
# OUTPUTS: 
#       sindy_err_vec   : sindy error stats 
#       gpsindy_err_vec : gpsindy error stats 
# ----------------------- # 
    
    # choose ODE, plot states --> measurements 
    # fn = predator_prey 
    x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 
    
    # truth coeffs 
    n_vars = size(x_true, 2) ; poly_order = n_vars 
    Ξ_true = sindy_stls( x_true, dx_true, λ ) 

    # constants 
    α = 1.0  ; ρ = 1.0     

    sindy_err_vec = [] ; gpsindy_err_vec = [] ; hist_nvars_vec = [] 
    sindy_vec = [] ; gpsindy_vec = [] 
    for noise = noise_vec 
    
        # use true data 
        if case == 0 

            # set noise = true 
            x_noise  = x_true ; dx_noise = dx_true 
            
            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 
    
        # use true x, finite difference dx 
        elseif case == 1 
    
            # set noise = true 
            x_noise  = x_true ; dx_noise = dx_fd 
            
            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 

        # use noisy data  
        elseif case == 2 

            # add noise 
            println( "noise = ", noise ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 
    
        # use standardized true data 
        elseif case == 3 

            # set noise = standardized 
            x_true  = stand_data( t, x_true ) 
            dx_true = dx_true_fn( t, x_true, p, fn ) 
            x_noise = x_true ; dx_noise = dx_true  

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 

        # use standardized noisy data 
        elseif case == 4 

            # add noise 
            println( "noise = ", noise ) 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 

        # standardize true x, finite difference dx 
        elseif case == 5 

            # set noise = standardized 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true 
            dx_noise = fdiff( t, x_true, 2 ) 

            Θx_sindy   = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy    = sindy_stls( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 

        # standardize and just use GP to smooth states (TEMPORAL) 
        elseif case == 6 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true  = stand_data( t, x_true ) 
            dx_true = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 

            # smooth measurements 
            # t_test = collect( t[1] : 0.01 : t[end] )  
            t_test = t 
            x_GP,  Σ_test, hp_test = post_dist_SE( t, x_noise, t_test ) 
            dx_GP, Σ_test, hp_test = post_dist_SE( t, dx_noise, t_test ) 

            Θx_gpsindy = pool_data_test( x_GP, n_vars, poly_order ) 
            Ξ_gpsindy  = sindy_stls( x_GP, dx_GP, λ ) 

        # standardize --> smooth states into SINDy w/ GP (NON-temporal)  
        elseif case == 7 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 

            # smooth measurements 
            x_GP, Σ_xsmooth, hp   = post_dist_SE( t, x_noise, t )  
            dx_GP, Σ_dxsmooth, hp = post_dist_SE( x_GP, dx_noise, x_GP )  
            
            Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
            Ξ_gpsindy  = sindy_stls( x_GP, dx_GP, λ ) 

        # standardize --> smooth states w/ GP (NON-temporal) --> GPSINDy 
        elseif case == 8 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true  = stand_data( t, x_true ) 
            dx_true = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 

            # I guess .... let's try this again
            x_GP, Σ_xsmooth, hp = post_dist_SE( t, x_noise, t )  
            dx_GP, Σ_dxsmooth   = post_dist_SE( x_GP, dx_noise, x_GP )  
            
            Θx_gpsindy          = pool_data_test(x_GP, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_GP, x_GP, λ, α, ρ, abstol, reltol )  

        # same as 7, but GP --> SINDy --> GP --> SINDy 
        elseif case == 9 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 
            
            # smooth measurements 
            # x_GP, Σ_xGP, hp   = post_dist_SE( t, x_noise, t )           # step -1 
            x_GP  = gp_post( t, 0*x_noise, t, 0*x_noise, x_noise ) 
            # dx_GP, Σ_dxGP, hp = post_dist_SE( x_GP, dx_noise, x_GP )    # step 0 
            dx_GP = gp_post( x_GP, 0*dx_noise, x_GP, 0*dx_noise, dx_noise ) 
            
            Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
            Ξ_gpsindy  = sindy_stls( x_GP, dx_GP, λ )                   # step 1 

            # ----------------------- #
            # GPSINDy (second) 

            # step 2: GP 
            dx_mean = Θx_gpsindy * Ξ_gpsindy 
            dx_post = gp_post( x_train_GP, dx_mean, x_train_GP, dx_train, dx_mean ) 

            # step 3: SINDy 
            Θx_gpsindy   = pool_data_test( x_train_GP, n_vars, poly_order ) 
            Ξ_gpsindy    = sindy_stls( x_train_GP, dx_post, λ ) 

        # standardize --> smooth states into SINDy w/ GP (NON-temporal) --> get accelerations (derivatives of derivatives) 
        elseif case == 10 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = sindy_stls( x_noise, dx_noise, λ ) 

            # smooth measurements 
            x_GP, Σ_xsmooth, hp   = post_dist_SE( t, x_noise, t )  
            dx_GP, Σ_dxsmooth, hp = post_dist_SE( x_GP, dx_noise, x_GP )  
            
            Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
            Ξ_gpsindy  = sindy_stls( x_GP, dx_GP, λ ) 
                
        end 

        # plot 
        # plot_dx_sindy_gpsindy( t, dx_true, dx_noise, Θx_sindy, Ξ_sindy, Θx_gpsindy, Ξ_gpsindy ) 

        # metrics & diagnostics 
        Ξ_sindy_err, Ξ_gpsindy_err = l2_metric( n_vars, dx_noise, Θx_gpsindy, Ξ_true, Ξ_sindy, Ξ_gpsindy, sindy_err_vec, gpsindy_err_vec )
        push!( sindy_vec, Ξ_sindy ) 
        push!( gpsindy_vec, Ξ_gpsindy ) 


    end 

    # make matrices 
    Ξ_sindy_err   = mapreduce(permutedims, vcat, Ξ_sindy_err)
    Ξ_gpsindy_err = mapreduce(permutedims, vcat, Ξ_gpsindy_err)

    return Ξ_sindy_err, Ξ_gpsindy_err, hist_nvars_vec, Ξ_true, sindy_vec, gpsindy_vec 
end 
