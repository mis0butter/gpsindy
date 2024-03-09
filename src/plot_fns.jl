

using CairoMakie 

export plot_noise_GP_sindy_gpsindy_traindouble 

function plot_noise_GP_sindy_gpsindy_traindouble( t, x_noise, t_double, x_GP, x_sindy, x_gpsindy, csv_string ) 

    # csv_string = replace( csv_file, path => "" ) 

    fig = Figure( size = (1000, 800) ) 
    
        # trajectory subplots 
        noise = 0 ; GP = 0 ; sindy = 0 ; gpsindy = 0 
        for i = 1 : 4 
    
            if i == 1 
                Axis( fig[i, 1], ylabel = string( "x", i ), title = string("Trajectory: ", csv_string) ) 
            elseif i == 4 
                Axis( fig[i,1], xlabel = "t", ylabel = string( "x", i ) ) 
            else 
                Axis( fig[i,1], ylabel = string( "x", i ) ) 
            end 

            noise   = lines!( fig[i, 1], t, x_noise[:,i], linewidth = 4, label = "noise" ) 
            GP      = lines!( fig[i, 1], t_double, x_GP[:,i], linewidth = 2, label = "GP" ) 
            sindy   = lines!( fig[i, 1], t, x_sindy[:,i], linestyle = :dash, label = "sindy" ) 
            gpsindy = lines!( fig[i, 1], t, x_gpsindy[:,i], linestyle = :dash, label = "gpsindy" ) 
    
        end 
        
        # trajectory plots legend 
        Legend( fig[5, 1],
        [ noise, GP, sindy, gpsindy ],
        ["GT", "GP", "sindy", "gpsindy"], 
        orientation = :horizontal ) 

        t_dbl_dwnsmpl, x_GP_dwnsmpl  = downsample( t, t_double, x_GP ) 
    
        # error subplots 
        for i = 1 : 4  
            
            err_GP          = x_noise[:,i] - x_GP_dwnsmpl[:,i] 
            err_GP_str      = @sprintf "%.3g" norm(err_GP) 
            err_sindy       = x_noise[:,i] - x_sindy[:,i] 
            err_sindy_str   = @sprintf "%.3g" norm(err_sindy) 
            err_gpsindy     = x_noise[:,i] - x_gpsindy[:,i] 
            err_gpsindy_str = @sprintf "%.3g" norm(err_gpsindy) 

            err_title = string( "GP = ", err_GP_str, " | sindy = ", err_sindy_str, " | gpsindy = ", err_gpsindy_str ) 
    
            if i == 1 
                Axis( fig[i, 2], ylabel = string( "x", i ), title = string("Error: ", err_title) ) 
            elseif i == 4 
                Axis( fig[i, 2], xlabel = "t", ylabel = string( "x", i ), title = err_title ) 
            else 
                Axis( fig[i, 2], ylabel = string( "x", i ), title = err_title ) 
            end 

                noise   = lines!( fig[i, 2], NaN, NaN ) 
                GP      = lines!( fig[i, 2], t, err_GP, linewidth = 2, label   = string( "x", i, "_GP" ) ) 
                sindy   = lines!( fig[i, 2], t, err_sindy, linestyle = :dash, label = string( "x", i, "_sindy" ) ) 
                gpsindy = lines!( fig[i, 2], t, err_gpsindy, linestyle = :dash, label = string( "x", i, "_gpsindy" ) ) 
    
        end 
        
        # error subplots legend 
        Legend( fig[5, 2], 
        [ GP, sindy, gpsindy ], 
        [ "GP", "sindy", "gpsindy" ], 
        orientation = :horizontal ) 

    return fig 
end 


## ============================================ ## 

using CairoMakie 

export plot_noise_GP_sindy_gpsindy 

function plot_noise_GP_sindy_gpsindy( t, x_noise, x_GP, x_sindy, x_gpsindy, csv_string ) 

    # csv_string = replace( csv_file, path => "" ) 

    fig = Figure( size = (1000, 800) ) 
    
        # trajectory subplots 
        noise = 0 ; GP = 0 ; sindy = 0 ; gpsindy = 0 
        for i = 1 : 4 
    
            if i == 1 
                Axis( fig[i, 1], ylabel = string( "x", i ), title = string("Trajectory: ", csv_string) ) 
            elseif i == 4 
                Axis( fig[i,1], xlabel = "t", ylabel = string( "x", i ) ) 
            else 
                Axis( fig[i,1], ylabel = string( "x", i ) ) 
            end 

            noise   = lines!( fig[i, 1], t, x_noise[:,i], linewidth = 4, label = "noise" ) 
            GP      = lines!( fig[i, 1], t, x_GP[:,i], linewidth = 2, label = "GP" ) 
            sindy   = lines!( fig[i, 1], t, x_sindy[:,i], linestyle = :dash, label = "sindy" ) 
            gpsindy = lines!( fig[i, 1], t, x_gpsindy[:,i], linestyle = :dash, label = "gpsindy" ) 
    
        end 
        
        # trajectory plots legend 
        Legend( fig[5, 1],
        [ noise, GP, sindy, gpsindy ],
        ["GT", "GP", "sindy", "gpsindy"], 
        orientation = :horizontal ) 
    
        # error subplots 
        for i = 1 : 4  
            
            err_GP          = x_noise[:,i] - x_GP[:,i] 
            err_GP_str      = @sprintf "%.3g" norm(err_GP) 
            err_sindy       = x_noise[:,i] - x_sindy[:,i] 
            err_sindy_str   = @sprintf "%.3g" norm(err_sindy) 
            err_gpsindy     = x_noise[:,i] - x_gpsindy[:,i] 
            err_gpsindy_str = @sprintf "%.3g" norm(err_gpsindy) 

            err_title = string( "GP = ", err_GP_str, " | sindy = ", err_sindy_str, " | gpsindy = ", err_gpsindy_str ) 
    
            if i == 1 
                Axis( fig[i, 2], ylabel = string( "x", i ), title = string("Error: ", err_title) ) 
            elseif i == 4 
                Axis( fig[i, 2], xlabel = "t", ylabel = string( "x", i ), title = err_title ) 
            else 
                Axis( fig[i, 2], ylabel = string( "x", i ), title = err_title ) 
            end 

                noise   = lines!( fig[i, 2], NaN, NaN ) 
                GP      = lines!( fig[i, 2], t, err_GP, linewidth = 2, label   = string( "x", i, "_GP" ) ) 
                sindy   = lines!( fig[i, 2], t, err_sindy, linestyle = :dash, label = string( "x", i, "_sindy" ) ) 
                gpsindy = lines!( fig[i, 2], t, err_gpsindy, linestyle = :dash, label = string( "x", i, "_gpsindy" ) ) 
    
        end 
        
        # error subplots legend 
        Legend( fig[5, 2], 
        [ GP, sindy, gpsindy ], 
        [ "GP", "sindy", "gpsindy" ], 
        orientation = :horizontal ) 

    return fig 
end 


## ============================================ ##

function plot_car_x_dx_noise_GP( t, x_noise, dx_noise, x_GP, dx_GP ) 
    # plot car x and dx data 
    
    fig = Figure() 
    
        Axis( fig[1:2, 1], xlabel = "x", ylabel = "y" ) 
            lines!( fig[1:2,1] , x_noise[:,1], x_noise[:,2], label = "noise" ) 
            lines!( fig[1:2,1] , x_GP[:,1], x_GP[:,2], label = "GP" )
            axislegend()  
        Axis( fig[3, 1], xlabel = "t", ylabel = "v" ) 
            lines!( fig[3,1] , t, x_noise[:,3], label = "v" ) 
            lines!( fig[3,1] , t, x_GP[:,3], label = "v" ) 
        Axis( fig[4, 1], xlabel = "t", ylabel = "θ" ) 
            lines!( fig[4,1] , t, x_noise[:,4], label = "θ" ) 
            lines!( fig[4,1] , t, x_GP[:,4], label = "θ" )     
        Axis( fig[1, 2], ylabel = "xdot" ) 
            lines!( fig[1,2] , t, dx_noise[:,1], label = "raw" ) 
            lines!( fig[1,2] , t, dx_GP[:,1], label = "raw" ) 
        Axis( fig[2, 2], ylabel = "ydot" ) 
            lines!( fig[2,2] , t, dx_noise[:,2], label = "raw" ) 
            lines!( fig[2,2] , t, dx_GP[:,2], label = "raw" ) 
        Axis( fig[3, 2], ylabel = "vdot" ) 
            lines!( fig[3,2] , t, dx_noise[:,3], label = "v" ) 
            lines!( fig[3,2] , t, dx_GP[:,3], label = "v" ) 
        Axis( fig[4, 2], xlabel = "t", ylabel = "θdot" ) 
            lines!( fig[4,2] , t, dx_noise[:,4], label = "θ" ) 
            lines!( fig[4,2] , t, dx_GP[:,4], label = "θ" ) 
    
    return fig 
end 
    
export plot_car_x_dx_noise_GP 


## ============================================ ##

function plot_car_x_dx_noise( t, x, dx, x_noise, dx_noise ) 
# plot car x and dx data 

fig = Figure() 

    Axis( fig[1:2, 1], xlabel = "x", ylabel = "y" ) 
        lines!( fig[1:2,1] , x[:,1], x[:,2], label = "raw" ) 
        lines!( fig[1:2,1] , x_noise[:,1], x_noise[:,2], linestyle = :dash, label = "noise" ) 
    Axis( fig[3, 1], xlabel = "t", ylabel = "v" ) 
        lines!( fig[3,1] , t, x[:,3], label = "v" ) 
        lines!( fig[3,1] , t, x_noise[:,3], linestyle = :dash, label = "v" ) 
    Axis( fig[4, 1], xlabel = "t", ylabel = "θ" ) 
        lines!( fig[4,1] , t, x[:,4], label = "θ" ) 
        lines!( fig[4,1] , t, x_noise[:,4], linestyle = :dash, label = "θ" ) 

    Axis( fig[1, 2], ylabel = "xdot" ) 
        lines!( fig[1,2] , t, dx[:,1], label = "raw" ) 
        lines!( fig[1,2] , t, dx_noise[:,1], linestyle = :dash, label = "raw" ) 
    Axis( fig[2, 2], ylabel = "ydot" ) 
        lines!( fig[2,2] , t, dx[:,2], label = "raw" ) 
        lines!( fig[2,2] , t, dx_noise[:,2], linestyle = :dash, label = "raw" ) 
    Axis( fig[3, 2], ylabel = "vdot" ) 
        lines!( fig[3,2] , t, dx[:,3], label = "v" ) 
        lines!( fig[3,2] , t, dx_noise[:,3], linestyle = :dash, label = "v" ) 
    Axis( fig[4, 2], xlabel = "t", ylabel = "θdot" ) 
        lines!( fig[4,2] , t, dx[:,4], label = "θ" ) 
        lines!( fig[4,2] , t, dx_noise[:,4], linestyle = :dash, label = "θ" ) 

    return fig 
end 

export plot_car_x_dx_noise 


## ============================================ ##

function plot_car_x_dx( t, x, dx ) 
# plot car x and dx data 

fig = Figure() 

    Axis( fig[1:2, 1], xlabel = "x", ylabel = "y" ) 
        lines!( fig[1:2,1] , x[:,1], x[:,2], label = "raw" ) 
    Axis( fig[3, 1], xlabel = "t", ylabel = "v" ) 
        lines!( fig[3,1] , t, x[:,3], label = "v" ) 
    Axis( fig[4, 1], xlabel = "t", ylabel = "θ" ) 
        lines!( fig[4,1] , t, x[:,4], label = "θ" ) 

    Axis( fig[1, 2], ylabel = "xdot" ) 
        lines!( fig[1,2] , t, dx[:,1], label = "raw" ) 
    Axis( fig[2, 2], ylabel = "ydot" ) 
        lines!( fig[2,2] , t, dx[:,2], label = "raw" ) 
    Axis( fig[3, 2], ylabel = "vdot" ) 
        lines!( fig[3,2] , t, dx[:,3], label = "v" ) 
    Axis( fig[4, 2], xlabel = "t", ylabel = "θdot" ) 
        lines!( fig[4,2] , t, dx[:,4], label = "θ" ) 

    return fig 
end 

export plot_car_x_dx 


## ============================================ ##

export plot_med_quarts_sindy_nn_gpsindy
function plot_med_quarts_sindy_nn_gpsindy(x_sindy_lasso_err, x_nn_err, x_gpsindy_err, noise_vec, plot_title)

    n_vars = size(x_sindy_lasso_err, 2)
    unique_i = unique(i -> noise_vec[i], 1:length(noise_vec))
    push!(unique_i, length(noise_vec) + 1)

    sindy_med   = []
    sindy_q13   = []
    nn_med      = []
    nn_q13      = []
    gpsindy_med = []
    gpsindy_q13 = []
    for i = 1:length(unique_i)-1

        ji = unique_i[i]
        jf = unique_i[i+1] - 1

        sindy_med_i   = []
        sindy_q13_i   = []
        nn_med_i   = []
        nn_q13_i   = []
        gpsindy_med_i = []
        gpsindy_q13_i = []
        for j = 1:n_vars
            push!(sindy_med_i, median(x_sindy_lasso_err[ji:jf, j]))
            push!(sindy_q13_i, [quantile(x_sindy_lasso_err[ji:jf, j], 0.25), quantile(x_sindy_lasso_err[ji:jf, j], 0.75)])
            push!(nn_med_i, median(x_nn_err[ji:jf, j]))
            push!(nn_q13_i, [quantile(x_nn_err[ji:jf, j], 0.25), quantile(x_nn_err[ji:jf, j], 0.75)])
            push!(gpsindy_med_i, median(x_gpsindy_err[ji:jf, j]))
            push!(gpsindy_q13_i, [quantile(x_gpsindy_err[ji:jf, j], 0.25), quantile(x_gpsindy_err[ji:jf, j], 0.75)])
        end

        push!(sindy_med, sindy_med_i)
        push!(sindy_q13, sindy_q13_i)
        push!(nn_med, nn_med_i)
        push!(nn_q13, nn_q13_i)
        push!(gpsindy_med, gpsindy_med_i)
        push!(gpsindy_q13, gpsindy_q13_i)

    end
    sindy_med = vv2m(sindy_med)
    sindy_q13 = vv2m(sindy_q13)
    nn_med = vv2m(nn_med)
    nn_q13 = vv2m(nn_q13)
    gpsindy_med = vv2m(gpsindy_med)
    gpsindy_q13 = vv2m(gpsindy_q13)

    noise_vec_iter = unique(noise_vec)
    p_nvars = []
    for i = 1:n_vars

        plt = plot(legend=:outerright, size=[800 300], title=string("|| true - discovered ||_2"), xlabel="noise")

        ymed = sindy_med[:, i]
        yq13 = vv2m(sindy_q13[:, i])
        plot!(plt, noise_vec_iter, ymed, c=:orange, label="SINDy", ribbon=(ymed - yq13[:, 1], yq13[:, 2] - ymed), fillalpha=0.35)
        scatter!(plt, noise_vec, x_sindy_lasso_err[:, i], c=:orange, markerstrokewidth=0, ms=3, markeralpha=0.35)

        ymed = nn_med[:, i]
        yq13 = vv2m(nn_q13[:, i])
        plot!(plt, noise_vec_iter, ymed, c=:magenta, label="NNSINDy", ribbon=(ymed - yq13[:, 1], yq13[:, 2] - ymed), fillalpha=0.35)
        scatter!(plt, noise_vec, x_nn_err[:, i], c=:magenta, markerstrokewidth=0, ms=3, markeralpha=0.35)

        ymed = gpsindy_med[:, i]
        yq13 = vv2m(gpsindy_q13[:, i])
        plot!(plt, noise_vec_iter, ymed, c=:cyan, label="GPSINDy", ribbon=(ymed - yq13[:, 1], yq13[:, 2] - ymed), fillalpha=0.35)
        scatter!(plt, noise_vec, x_gpsindy_err[:, i], c=:cyan, markerstrokewidth=0, ms=3, markeralpha=0.35)

        push!(p_nvars, plt)

    end
    p_nvars = plot(p_nvars...,
        layout=(2, 1),
        size=[800 600],
        plot_title = plot_title 
    )
    display(p_nvars)

end



## ============================================ ##
# plot 

export plot_x_sindy_nn_gpsindy_err
function plot_x_sindy_nn_gpsindy_err( noise_vec, x_sindy_err, x_nn_err, x_gpsindy_err)  
    
    # xmin, dx, xmax = min_d_max(noise_vec)
    
        # ymin, dy, ymax = min_d_max( [ x_sindy_err ; x_nn_err ; x_gpsindy_err ] )
        p = scatter( noise_vec, x_sindy_err, 
            c       = :red, 
            ls      = :dashdot, 
            legend  = :outerright, 
            xlabel  = "Time (s)", 
            label   = "SINDy (LASSO)", 
            # yticks  = ymin:dy:ymax,
            # ylim    = (ymin, ymax), 
            title   = "x error",
        ) 
        scatter!( p, noise_vec, x_nn_err, 
            c       = :blue, 
            label   = "NN (LASSO)", 
            # yticks  = ymin:dy:ymax,
            ls      = :dashdot, 
        ) 
        scatter!( p, noise_vec, x_gpsindy_err, 
            # c       = :blue, 
            label   = "GPSINDy (LASSO)", 
            # yticks = ymin:dy:ymax,
            ls      = :dot, 
        )
        
    
    display(p)     

end 


## ============================================ ##
# plot 

export plot_x_sindy_nn_gpsindy 
function plot_x_sindy_nn_gpsindy( t_test, x_test, x_sindy_lasso, x_nn, x_gpsindy_stls)  
    
    xmin, dx, xmax = min_d_max(t_test)
    
    x_vars = size(x_test, 2) 
    p_vec = [] 
    for i = 1 : x_vars 
    
        ymin, dy, ymax = min_d_max( x_test[:, i] )
        p = Plots.plot( t_test, x_test[:,i], 
            c       = :gray, 
            label   = "test", 
            legend  = :outerright, 
            xlabel  = "Time (s)", 
            xticks  = xmin:dx:xmax,
            yticks  = ymin:dy:ymax,
            ylim    = (ymin, ymax), 
            title   = string(latexify("x_$(i)")),
        ) 
        Plots.plot!( p, t_test, x_sindy_lasso[:,i], 
            c       = :red, 
            label   = "SINDy (LASSO)", 
            xticks  = xmin:dx:xmax,
            yticks  = ymin:dy:ymax,
            ls      = :dashdot, 
        ) 
        Plots.plot!( p, t_test, x_nn[:,i], 
            c       = :blue, 
            label   = "NN (LASSO)", 
            xticks  = xmin:dx:xmax,
            yticks  = ymin:dy:ymax,
            ls      = :dashdot, 
        ) 
        Plots.plot!( p, t_test, x_gpsindy_stls[:,i], 
            # c       = :blue, 
            label   = "GPSINDy (LASSO)", 
            xticks  = xmin:dx:xmax,
            yticks = ymin:dy:ymax,
            ls      = :dot, 
        )
        push!( p_vec, p ) 
    
    end 
    
    # p = deepcopy( p_vec[end] ) 
    # plot!( p, 
    #     legend = ( -0.1, 0.6 ), 
    #     framestyle = :none, 
    #     title = "",      
    # )  
    # push!( p_vec, p ) 
    
    pfig = Plots.plot(  p_vec ... , 
        layout = grid( x_vars, 1 ), 
        size   = [ 600 x_vars * 400 ],         
        margin = 5Plots.mm,
        bottom_margin = 14Plots.mm,
    )
    
    display(pfig)     

end 


## ============================================ ##
# plot noise, GP, SINDy, GPSINDy 

export plot_noise_sindy_gpsindy 
function plot_noise_sindy_gpsindy( t, x_noise, x_GP, x_sindy, x_gpsindy, plot_suptitle ) 
    
    # get sizes 
    x_vars = size( x_noise, 2 ) 

    plt_x_vec = [] 
    for j = 1 : x_vars 
        ymin, dy, ymax = min_d_max( x_noise[:,j] ) 
        plt_x = Plots.plot( t, x_noise[:,j], 
            xlabel = "t (s)", 
            ylabel = "x", 
            ylim   = ( ymin, ymax ), 
            title  = string("x", j, ": noise vs GP vs SINDy vs GPSINDy"), 
            legend = :outerright,
            size   = (1000, 400), 
            label  = "noise", 
        ) 
        Plots.plot!( plt_x, t, x_GP[:,j], 
            label = "GP", 
            ls    = :dash,   
        ) 
        Plots.plot!( plt_x, t, x_sindy[:,j], 
            label = "SINDy", 
            ls    = :dashdot,   
        ) 
        Plots.plot!( plt_x, t, x_gpsindy[:,j], 
        label = string("GPSINDy"),
        ls    = :dot,   
        ) 
        push!( plt_x_vec, plt_x ) 
    end 
    pfig = Plots.plot( plt_x_vec ... , 
        layout = ( x_vars, 1 ), 
        size = (1000, 400*x_vars), 
        plot_title = plot_suptitle, 
    )
    
    display(pfig)     

end 


## ============================================ ##
# plot 

export plot_validation_test 
function plot_validation_test( t_test, x_test, x_sindy_stls, x_sindy_lasso, x_nn, x_gpsindy_stls)  
    
    xmin, dx, xmax = min_d_max(t_test)
    
    x_vars = size(x_test, 2) 
    p_vec = [] 
    for i = 1 : x_vars 
    
        ymin, dy, ymax = min_d_max( x_test[:, i] )
    
        p = plot( t_test, x_test[:,i], 
            c       = :gray, 
            label   = "test", 
            legend  = :outerright, 
            xlabel  = "Time (s)", 
            xticks  = xmin:dx:xmax,
            yticks  = ymin:dy:ymax,
            ylim    = (ymin, ymax), 
            title   = string(latexify("x_$(i)")),
        ) 
        plot!( p, t_test, x_sindy_stls[:,i], 
            c       = :green, 
            label   = "SINDy (STLS)", 
            xticks  = xmin:dx:xmax,
            yticks  = ymin:dy:ymax,
            ls      = :dash, 
        ) 
        plot!( p, t_test, x_sindy_lasso[:,i], 
            c       = :red, 
            label   = "SINDy (LASSO)", 
            xticks  = xmin:dx:xmax,
            yticks  = ymin:dy:ymax,
            ls      = :dashdot, 
        ) 
        plot!( p, t_test, x_nn[:,i], 
            c       = :blue, 
            label   = "NN (LASSO)", 
            xticks  = xmin:dx:xmax,
            yticks  = ymin:dy:ymax,
            ls      = :dashdot, 
        ) 
        plot!( p, t_test, x_gpsindy_stls[:,i], 
            # c       = :blue, 
            label   = "GPSINDy (LASSO)", 
            xticks  = xmin:dx:xmax,
            yticks = ymin:dy:ymax,
            ls      = :dot, 
        )
        push!( p_vec, p ) 
    
    end 
    
    # p = deepcopy( p_vec[end] ) 
    # plot!( p, 
    #     legend = ( -0.1, 0.6 ), 
    #     framestyle = :none, 
    #     title = "",      
    # )  
    # push!( p_vec, p ) 
    
    pfig = plot(  p_vec ... , 
        layout = grid( x_vars, 1 ), 
        size   = [ 600 x_vars * 400 ],         
        margin = 5Plots.mm,
        bottom_margin = 14Plots.mm,
    )
    
    display(pfig)     

end 

## ============================================ ##
# compare computed mean with training data 

export plot_dx_mean 
function plot_dx_mean( t_train, x_train, x_GP_train, u_train, dx_train, dx_GP_train, Ξ_sindy, Ξ_gpsindy, poly_order ) 

    x_vars = size(x_train, 2) 
    u_vars = size(u_train, 2) 
    n_vars = x_vars + u_vars 

    # SINDy alone 
    Θx = pool_data_test( [x_train u_train], n_vars, poly_order) 
    dx_sindy = Θx * Ξ_sindy 

    # GPSINDy 
    Θx = pool_data_test( [x_GP_train u_train], n_vars, poly_order) 
    dx_gpsindy = Θx * Ξ_gpsindy 

    plt = plot( title = "dx: meas vs. sindy", legend = :outerright )
    scatter!( plt, t_train, dx_train[:,1], c = :black, ms = 3, label = "meas (finite diff)" )
    plot!( plt, t_train, dx_GP_train[:,1], c = :blue, label = "GP" )
    plot!( plt, t_train, dx_sindy[:,1], c = :red, ls = :dash, label = "SINDy" )   
    plot!( plt, t_train, dx_gpsindy[:,1], c = :green, ls = :dashdot, label = "GPSINDy" )   

    display(plt) 

end 

## ============================================ ##
# plotting finite difference vs gp derivative training data  

export plot_fd_gp_train 
function plot_fd_gp_train( t_train, dx_train, dx_GP_train ) 

    x_vars  = size(dx_train, 2) 
    p_nvars = [] 
    for i = 1 : x_vars 
        plt = plot( legend = :outerright, title = string("dx", i) )
            scatter!( plt, t_train, dx_train[:,i], label = "FD" ) 
            plot!( plt, t_train, dx_GP_train[:,i], label = "GP" ) 
        push!( p_nvars, plt ) 
    end 
    p_nvars = plot( p_nvars ... ,  
        layout = (x_vars, 1), 
        size   = [600 1200], 
        plot_title = "FD vs GP training data"
    ) 
    display(p_nvars) 

end 


## ============================================ ##
# plot sindy and gpsindy stats boxplot 

export boxplot_err
function boxplot_err(noise_vec, sindy_err_vec, gpsindy_err_vec)

    xmin, dx, xmax = min_d_max(noise_vec)

    p_Ξ = []
    for i = 1:2
        ymin, dy, ymax = min_d_max([sindy_err_vec[:, i]; gpsindy_err_vec[:, i]])
        p_ξ = scatter(noise_vec, sindy_err_vec[:, i], shape=:circle, ms=2, c=:blue, label="SINDy")
        boxplot!(p_ξ, noise_vec, sindy_err_vec[:, i], bar_width=0.04, lw=1, fillalpha=0.2, c=:blue, linealpha=0.5)
        scatter!(p_ξ, noise_vec, gpsindy_err_vec[:, i], shape=:xcross, c=:red, label="GPSINDy")
        boxplot!(p_ξ, noise_vec, gpsindy_err_vec[:, i], bar_width=0.02, lw=1, fillalpha=0.2, c=:red, linealpha=0.5)
        scatter!(p_ξ,
            legend=false,
            xlabel="noise",
            title=string("\n ||ξ", i, "_true - ξ", i, "_discovered||"),
            xticks=xmin:dx:xmax,
            # yticks = ymin : dy : ymax, 
        )
        push!(p_Ξ, p_ξ)
    end
    p = deepcopy(p_Ξ[end])
    plot!(p,
        legend=(-0.2, 0.6),
        framestyle=:none,
        title="",
    )
    push!(p_Ξ, p)
    p_Ξ = plot(p_Ξ...,
        layout=grid(1, 3, widths=[0.45, 0.45, 0.45]),
        size=[800 300],
        margin=8Plots.mm,
    )
    display(p_Ξ)

    return p_Ξ
end

## ============================================ ##
# plot prey vs. predator 

export plot_test_data
function plot_test_data(t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_x2_val, x_gpsindy_x2_val, t_nn_val, x_nn_val)


    # determine xtick range 
    xmin, dx, xmax = min_d_max(t_test)

    n_vars = size(x_test, 2)
    plot_vec = []

    for i = 1:n_vars

        ymin, dy, ymax = min_d_max(x_test[:, i])

        # display test data 
        p = plot(t_test, x_test[:, i],
            c=RGB(0, 0.35, 1),
            lw=3,
            label="test (20%)",
            xlim=(xmin, xmax),
            ylim=(ymin - dy / 3, ymax + dy / 3),
            xticks=xmin:dx:xmax,
            yticks=ymin:dy:ymax,
            xlabel="Time (s)",
            title=string(latexify("x_$(i)")),
        )
        plot!(t_sindy_val, x_sindy_val[:, i],
            ls=:dash,
            # c     = :red , 
            c=RGB(1, 0.25, 0),
            lw=3,
            label="SINDy",
        )
        plot!(t_gpsindy_val, x_gpsindy_val[:, i],
            ls=:dashdot,
            c=RGB(0, 0.75, 0),
            lw=2,
            label="GPSINDy",
        )
        plot!(t_gpsindy_x2_val, x_gpsindy_x2_val[:, i],
            ls=:dot,
            c=RGB(0, 0, 0.75),
            lw=1,
            label="GPSINDy x2",
        )
        plot!(t_nn_val, x_nn_val[:, i],
            ls=:solid,
            c=RGB(0.75, 0, 0),
            lw=2,
            label="NN",
        )
        push!(plot_vec, p)

    end

    p = deepcopy(plot_vec[end])
    plot!(p,
        legend=(-0.1, 0.6),
        # foreground_color_legend = nothing , 
        framestyle=:none,
        title="",
    )
    push!(plot_vec, p)

    p_train_val = plot(plot_vec...,
        # layout = (1, n_vars+1), 
        layout=grid(1, n_vars + 1, widths=[0.4, 0.4, 0.45]),
        size=[n_vars * 400 250],
        margin=5Plots.mm,
        bottom_margin=7Plots.mm,
        # plot_title = "Training vs. Validation Data", 
        # titlefont = font(16), 
    )
    display(p_train_val)

end

## ============================================ ##
# plot prey vs. predator 

export plot_states
function plot_states(t_train, x_train, t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_x2_val, x_gpsindy_x2_val, t_nn_val, x_nn_val)

    # scalefontsizes(1.1)
    # ptitles = ["Prey", "Predator"]

    # determine xtick range 
    t = [t_train; t_test]
    x = [x_train; x_test]
    xmin, dx, xmax = min_d_max(t)

    n_vars = size(x_train, 2)
    plot_vec = []

    # determine if test data in middle 
    tdiff = diff(vec(t_train))
    ttol = 1 / 2 * (maximum(t_test) - minimum(t_test))
    if any(tdiff .> ttol)
        portion_mid = true
        ind = findfirst(tdiff .> ttol)
        t_train_B = t_train[ind+1:end, :]
        x_train_B = x_train[ind+1:end, :]
        t_train = t_train[1:ind, :]
        x_train = x_train[1:ind, :]
    else
        portion_mid = false
    end

    for i = 1:n_vars

        ymin, dy, ymax = min_d_max(x[:, i])

        p = plot(t_train, x_train[:, i],
            c=:gray,
            label="train (80%)",
            xlim=(xmin, xmax),
            ylim=(ymin - dy / 3, ymax + dy / 3),
            xticks=xmin:dx:xmax,
            yticks=ymin:dy:ymax,
            xlabel="Time (s)",
            title=string(latexify("x_$(i)")),
        )
        # display training data 
        if portion_mid
            plot!(t_train_B, x_train_B[:, i],
                c=:gray,
                primary=false,
            )
        end
        plot!(t_test, x_test[:, i],
            # ls = :dash, 
            # c     = :blue,
            c=RGB(0, 0.35, 1),
            lw=3,
            label="test (20%)",
        )
        plot!(t_sindy_val, x_sindy_val[:, i],
            ls=:dash,
            # c     = :red , 
            c=RGB(1, 0.25, 0),
            lw=3,
            label="SINDy",
        )
        plot!(t_gpsindy_val, x_gpsindy_val[:, i],
            ls=:dashdot,
            c=RGB(0, 0.75, 0),
            lw=2,
            label="GPSINDy",
        )
        plot!(t_gpsindy_x2_val, x_gpsindy_x2_val[:, i],
            ls=:dot,
            c=RGB(0, 0, 0.75),
            lw=1,
            label="GPSINDy x2",
        )
        plot!(t_nn_val, x_nn_val[:, i],
            ls=:solid,
            c=RGB(0.75, 0, 0.75),
            lw=2,
            label="NN",
        )

        push!(plot_vec, p)

    end

    p = deepcopy(plot_vec[end])
    plot!(p,
        legend=(-0.1, 0.6),
        # foreground_color_legend = nothing , 
        framestyle=:none,
        title="",
    )
    push!(plot_vec, p)

    p_train_val = plot(plot_vec...,
        # layout = (1, n_vars+1), 
        layout=grid(1, n_vars + 1, widths=[0.4, 0.4, 0.45]),
        size=[n_vars * 400 250],
        margin=5Plots.mm,
        bottom_margin=7Plots.mm,
        # plot_title = "Training vs. Validation Data", 
        # titlefont = font(16), 
    )
    display(p_train_val)

end


## ============================================ ##
# plot derivatives 

export plot_deriv
function plot_deriv(t, dx_true, dx_fd, dx_tv, str)

    n_vars = size(dx_true, 2)

    xmin, dx, xmax = min_d_max(t)

    plot_vec_dx = []
    for j in 1:n_vars
        ymin, dy, ymax = min_d_max(dx_true[:, j])
        plt = plot(t, dx_true[:, j],
            title="dx $(j)", label="true",
            xticks=xmin:dx:xmax,
            yticks=ymin:dy:ymax,
            xlabel="Time (s)"
        )
        plot!(t, dx_fd[:, j], ls=:dash, label="finite diff")
        # plot!(t, dx_tv[:,j], ls = :dash, label = "var diff" )
        push!(plot_vec_dx, plt)
    end

    plot_dx = plot(plot_vec_dx...,
        layout=(1, n_vars),
        size=[n_vars * 400 250],
        # plot_title = "Derivatives. ODE fn = $( str )" 
    )
    display(plot_dx)

    return plot_dx

end


## ============================================ ##
# plot state 

export plot_dyn
function plot_dyn(t, x, str)

    n_vars = size(x, 2)

    # construct empty vector for plots 
    plot_vec_x = []
    for i = 1:n_vars
        plt = plot(t, x[:, i], title="State $(i)")
        push!(plot_vec_x, plt)
    end
    plot_x = plot(plot_vec_x...,
        layout=(1, n_vars),
        size=[n_vars * 400 250],
        xlabel="Time (s)",
        plot_title="Dynamics. ODE fn = $( str )")
    display(plot_x)

    return plot_x

end


## ============================================ ##

export plot_dx_sindy_gpsindy
function plot_dx_sindy_gpsindy(t, dx_true, dx_noise, Θx_sindy, Ξ_sindy, Θx_gpsindy, Ξ_gpsindy)

    n_vars = size(dx_true, 2)
    plt_nvars = []
    for i = 1:n_vars
        plt = plot(t, dx_true[:, i], label="true", c=:black)
        scatter!(plt, t, dx_noise[:, i], label="train (noise)", c=:black, ms=3)
        plot!(plt, t, Θx_sindy * Ξ_sindy[:, i], label="SINDy", c=:red)
        plot!(plt, t, Θx_gpsindy * Ξ_gpsindy[:, i], label="GPSINDy", ls=:dash, c=:cyan)
        plot!(plt, legend=:outerright, size=[800 300], title=string("Fitting ξ", i), xlabel="Time (s)")
        push!(plt_nvars, plt)
    end
    plt_nvars = plot(plt_nvars...,
        layout=(2, 1),
        size=[800 600]
    )
    display(plt_nvars)

    return plt_nvars
end


## ============================================ ##

export plot_med_quarts
function plot_med_quarts(sindy_err_vec, gpsindy_err_vec, noise_vec)

    n_vars = size(sindy_err_vec, 2)
    unique_i = unique(i -> noise_vec[i], 1:length(noise_vec))
    push!(unique_i, length(noise_vec) + 1)

    sindy_med = []
    sindy_q13 = []
    gpsindy_med = []
    gpsindy_q13 = []
    for i = 1:length(unique_i)-1

        ji = unique_i[i]
        jf = unique_i[i+1] - 1

        smed = []
        sq13 = []
        gpsmed = []
        gpsq13 = []
        for j = 1:n_vars
            push!(smed, median(sindy_err_vec[ji:jf, j]))
            push!(sq13, [quantile(sindy_err_vec[ji:jf, j], 0.25), quantile(sindy_err_vec[ji:jf, j], 0.75)])
            push!(gpsmed, median(gpsindy_err_vec[ji:jf, j]))
            push!(gpsq13, [quantile(gpsindy_err_vec[ji:jf, j], 0.25), quantile(gpsindy_err_vec[ji:jf, j], 0.75)])
        end

        push!(sindy_med, smed)
        push!(sindy_q13, sq13)
        push!(gpsindy_med, gpsmed)
        push!(gpsindy_q13, gpsq13)

    end
    sindy_med = vv2m(sindy_med)
    sindy_q13 = vv2m(sindy_q13)
    gpsindy_med = vv2m(gpsindy_med)
    gpsindy_q13 = vv2m(gpsindy_q13)

    noise_vec_iter = unique(noise_vec)
    p_nvars = []
    for i = 1:n_vars

        plt = plot(legend=:outerright, size=[800 300], title=string("|| ξ", i, "_true - ξ", i, "_discovered ||"), xlabel="noise")

        ymed = sindy_med[:, i]
        yq13 = vv2m(sindy_q13[:, i])
        plot!(plt, noise_vec_iter, ymed, c=:orange, label="SINDy", ribbon=(ymed - yq13[:, 1], yq13[:, 2] - ymed), fillalpha=0.35)
        scatter!(plt, noise_vec, sindy_err_vec[:, i], c=:orange, markerstrokewidth=0, ms=3, markeralpha=0.35)

        ymed = gpsindy_med[:, i]
        yq13 = vv2m(gpsindy_q13[:, i])
        plot!(plt, noise_vec_iter, ymed, c=:cyan, label="GPSINDy", ribbon=(ymed - yq13[:, 1], yq13[:, 2] - ymed), fillalpha=0.35)
        scatter!(plt, noise_vec, gpsindy_err_vec[:, i], c=:cyan, markerstrokewidth=0, ms=3, markeralpha=0.35)
        
        push!(p_nvars, plt)
    end
    p_nvars = plot(p_nvars...,
        layout=(2, 1),
        size=[800 600],
        plot_title="1/4 Quartile, Median, and 3/4 Quartile "
    )
    display(p_nvars)

end


## ============================================ ##

export plot_med_quarts_sindy_nn_gpsindy
function plot_med_quarts_sindy_nn_gpsindy(sindy_err_vec, nnsindy_err_vec, gpsindy_err_vec, noise_vec)

    n_vars = size(sindy_err_vec, 2)
    unique_i = unique(i -> noise_vec[i], 1:length(noise_vec))
    push!(unique_i, length(noise_vec) + 1)

    sindy_med = []
    sindy_q13 = []
    gpsindy_med = []
    gpsindy_q13 = []
    nnsindy_med = []
    nnsindy_q13 = []
    for i = 1:length(unique_i)-1

        ji = unique_i[i]
        jf = unique_i[i+1] - 1

        smed = []
        gpsmed = []
        nnsmed = []
        sq13 = []
        gpsq13 = []
        nnsq13 = []
        for j = 1:n_vars
            push!(smed,   median(sindy_err_vec[ji:jf, j]))
            push!(gpsmed, median(gpsindy_err_vec[ji:jf, j]))
            push!(nnsmed, median(nnsindy_err_vec[ji:jf, j]))
            push!(sq13,   [quantile(sindy_err_vec[ji:jf, j], 0.25), quantile(sindy_err_vec[ji:jf, j], 0.75)])
            push!(gpsq13, [quantile(gpsindy_err_vec[ji:jf, j], 0.25), quantile(gpsindy_err_vec[ji:jf, j], 0.75)])
            push!(nnsq13, [quantile(nnsindy_err_vec[ji:jf, j], 0.25), quantile(nnsindy_err_vec[ji:jf, j], 0.75)])
        end

        push!(sindy_med, smed)
        push!(sindy_q13, sq13)
        push!(gpsindy_med, gpsmed)
        push!(gpsindy_q13, gpsq13)
        push!(nnsindy_med, nnsmed)
        push!(nnsindy_q13, nnsq13)

    end
    sindy_med = vv2m(sindy_med)
    sindy_q13 = vv2m(sindy_q13)
    gpsindy_med = vv2m(gpsindy_med)
    gpsindy_q13 = vv2m(gpsindy_q13)
    nnsindy_med = vv2m(nnsindy_med)
    nnsindy_q13 = vv2m(nnsindy_q13)

    noise_vec_iter = unique(noise_vec)
    p_nvars = []
    for i = 1:n_vars
        plt = plot(legend=:outerright, size=[800 300], ylabel=string("|| ξ", i, "_true - ξ", i, "_discovered ||"), xlabel="noise")

        # sindy 
        ymed = sindy_med[:, i]
        yq13 = vv2m(sindy_q13[:, i])
        plot!(plt, noise_vec_iter, ymed, c=:green, ribbon=(ymed - yq13[:, 1], yq13[:, 2] - ymed), fillalpha=0.35)
        scatter!(plt, noise_vec, sindy_err_vec[:, i], c=:green, markerstrokewidth=0, ms=3, markeralpha=0.35, label="SINDy")

        # gpsindy_gpsindy
        ymed = nnsindy_med[:, i]
        yq13 = vv2m(nnsindy_q13[:, i])
        plot!(plt, noise_vec_iter, ymed, c=:cyan, ls=:dashdot, ribbon=(ymed - yq13[:, 1], yq13[:, 2] - ymed), fillalpha=0.25)
        scatter!(plt, noise_vec, nnsindy_err_vec[:, i], c=:cyan, markerstrokewidth=0, ms=3, markeralpha=0.35, label="NNSINDy")

        # gpsindy 
        ymed = gpsindy_med[:, i]
        yq13 = vv2m(gpsindy_q13[:, i])
        plot!(plt, noise_vec_iter, ymed, c=:orange, ls=:dash, ribbon=(ymed - yq13[:, 1], yq13[:, 2] - ymed), fillalpha=0.3)
        scatter!(plt, noise_vec, gpsindy_err_vec[:, i], c=:orange, markerstrokewidth=0, ms=3, markeralpha=0.35, label="GPSINDy")

        push!(p_nvars, plt)
    end
    p_nvars = plot(p_nvars...,
        layout=(2, 1),
        size=[800 600],
        plot_title="1/4 Quartile, Median, and 3/4 Quartile "
    )
    display(p_nvars)

end

