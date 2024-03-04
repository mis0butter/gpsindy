
## ============================================ ##

function plot_car_raw_GP( t_train, x_train, dx_train, x_train_GP, dx_train_GP ) 

    fig = Figure() 

    Axis( fig[1:2, 1], xlabel = "x", ylabel = "y" ) 
        lines!( fig[1:2,1] , x_train[:,1], x_train[:,2], label = "raw" ) 
        lines!( fig[1:2,1] , x_train_GP[:,1], x_train_GP[:,2], linestyle = :dash, label = "GP" ) 
        axislegend() 
    
    Axis( fig[3, 1], xlabel = "t", ylabel = "v" ) 
        lines!( fig[3,1] , t_train, x_train[:,3], label = "v" ) 
        lines!( fig[3,1] , t_train, x_train_GP[:,3], linestyle = :dash, label = "v" ) 
    Axis( fig[4, 1], xlabel = "t", ylabel = "θ" ) 
        lines!( fig[4,1] , t_train, x_train[:,4], label = "θ" ) 
        lines!( fig[4,1] , t_train, x_train_GP[:,4], linestyle = :dash, label = "θ" ) 
    
    Axis( fig[1, 2], ylabel = "xdot" ) 
        lines!( fig[1,2] , t_train, dx_train[:,1], label = "raw" ) 
        lines!( fig[1,2] , t_train, dx_train_GP[:,1], label = "raw" ) 
    Axis( fig[2, 2], ylabel = "ydot" ) 
        lines!( fig[2,2] , t_train, dx_train[:,2], label = "raw" ) 
        lines!( fig[2,2] , t_train, dx_train_GP[:,2], label = "raw" ) 
    Axis( fig[3, 2], ylabel = "vdot" ) 
        lines!( fig[3,2] , t_train, dx_train[:,3], label = "v" ) 
        lines!( fig[3,2] , t_train, dx_train_GP[:,3], label = "v" ) 
    Axis( fig[4, 2], xlabel = "t", ylabel = "θdot" ) 
        lines!( fig[4,2] , t_train, dx_train[:,4], label = "θ" ) 
        lines!( fig[4,2] , t_train, dx_train_GP[:,4], label = "θ" ) 

    return fig 
end 

export plot_car_raw_GP 


## ============================================ ##
# plot quadcopter data 

 function plot_quad_dx_train_GP( t_train, dx_train, dx_train_GP ) 

    fig_train_GP = Figure( size = ( 800,400 ) )

    ax_1x = Axis(
        fig_train_GP[1:4,1],
        ylabel  = "xdot1"
    ) 
        lines!(ax_1x, t_train, dx_train[:,1], label = "Raw") 
        lines!(ax_1x, t_train, dx_train_GP[:,1], label = "GP") 
        axislegend() 
    ax_1y = Axis(
        fig_train_GP[5:8,1],
        ylabel  = "xdot2"
    ) 
        lines!(ax_1y, t_train, dx_train[:,2], label = "xdot2 Raw") 
        lines!(ax_1y, t_train, dx_train_GP[:,2], label = "xdot2 GP") 
    ax_1z = Axis(
        fig_train_GP[9:12,1],
        ylabel  = "xdot3"
    ) 
        lines!(ax_1z, t_train, dx_train[:,3], label = "xdot3 Raw") 
        lines!(ax_1z, t_train, dx_train_GP[:,3], label = "xdot3 GP") 
    
    ax_2 = Axis(
        fig_train_GP[1:4,2],
        ylabel  = "vdot1"
    ) 
        lines!(ax_2, t_train, dx_train[:,4], label = "v1 Raw") 
        lines!(ax_2, t_train, dx_train_GP[:,4], label = "v1 GP") 
        # axislegend()
    ax_3 = Axis( 
        fig_train_GP[5:8,2], 
        ylabel = "vdot2" 
    ) 
        lines!(ax_3, t_train, dx_train[:,5], label = "v2 Raw") 
        lines!(ax_3, t_train, dx_train_GP[:,5], label = "v2 GP") 
        # axislegend()
    ax_4 = Axis( 
        fig_train_GP[9:12,2], 
        xlabel = "t (s)", 
        ylabel = "vdot3", 
        )
        lines!(ax_4, t_train, dx_train[:,6], label = "v3 Raw") 
        lines!(ax_4, t_train, dx_train_GP[:,6], label = "v3 GP") 
    ax_5 = Axis(
        fig_train_GP[1:3,3],
        ylabel = "qdot1",
    ) 
        lines!(ax_5, t_train, dx_train[:,7], label = "Raw") 
        lines!(ax_5, t_train, dx_train_GP[:,7], label = "GP") 
    ax_6 = Axis(
        fig_train_GP[4:6,3],
        ylabel = "qdot2",
    ) 
        lines!(ax_6, t_train, dx_train[:,8], label = "Raw") 
        lines!(ax_6, t_train, dx_train_GP[:,8], label = "GP") 
    ax_7 = Axis(
        fig_train_GP[7:9,3],
        ylabel = "qdot3",
    ) 
        lines!(ax_7, t_train, dx_train[:,9], label = "Raw") 
        lines!(ax_7, t_train, dx_train_GP[:,9], label = "GP") 
    ax_8 = Axis( 
        fig_train_GP[10:12,3], 
        xlabel = "t (s)", 
        ylabel = "qdot4", 
    ) 
        lines!(ax_8, t_train, dx_train[:,10], label = "Raw") 
        lines!(ax_8, t_train, dx_train_GP[:,10], label = "GP") 
    ax_9 = Axis( 
        fig_train_GP[1:4,4], 
        ylabel = "wdot1", 
    ) 
        lines!(ax_9, t_train, dx_train[:,11], label = "Raw") 
        lines!(ax_9, t_train, dx_train_GP[:,11], label = "GP") 
    ax_10 = Axis( 
        fig_train_GP[5:8,4], 
        ylabel = "wdot2", 
    ) 
        lines!(ax_10, t_train, dx_train[:,12], label = "Raw") 
        lines!(ax_10, t_train, dx_train_GP[:,12], label = "GP") 
    ax_11 = Axis( 
        fig_train_GP[9:12,4], 
        xlabel = "t (s)", 
        ylabel = "wdot3", 
    ) 
        lines!(ax_11, t_train, dx_train[:,13], label = "Raw") 
        lines!(ax_11, t_train, dx_train_GP[:,13], label = "GP") 

    return fig_train_GP 
 end 

 export plot_quad_dx_train_GP 

## ============================================ ##
function plot_quad_x_train_GP( t_train, x_train, x_train_GP ) 

    fig_train_GP = Figure( size = ( 800,400 ) )
    ax_1 = Axis3(
        fig_train_GP[1:12,1],
        aspect = :data,
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
        title  = "Raw vs. GP Data Position"
    ) 
        lines!(ax_1, x_train[:,1], x_train[:,2], x_train[:,3], linewidth = 1, linestyle = :solid, label = "Raw") 
        lines!(ax_1, x_train_GP[:,1], x_train_GP[:,2], x_train_GP[:,3], linewidth = 1, linestyle = :solid, label = "GP") 
        axislegend()
    
    ax_2 = Axis(
        fig_train_GP[1:4,2],
        ylabel  = "v1"
    ) 
        lines!(ax_2, t_train, x_train[:,4], label = "v1 Raw") 
        lines!(ax_2, t_train, x_train_GP[:,4], label = "v1 GP") 
        # axislegend()
    ax_3 = Axis( 
        fig_train_GP[5:8,2], 
        ylabel = "v2" 
    ) 
        lines!(ax_3, t_train, x_train[:,5], label = "v2 Raw") 
        lines!(ax_3, t_train, x_train_GP[:,5], label = "v2 GP") 
        # axislegend()
    ax_4 = Axis( 
        fig_train_GP[9:12,2], 
        xlabel = "t (s)", 
        ylabel = "v3", 
        )
        lines!(ax_4, t_train, x_train[:,6], label = "v3 Raw") 
        lines!(ax_4, t_train, x_train_GP[:,6], label = "v3 GP") 
    ax_5 = Axis(
        fig_train_GP[1:3,3],
        ylabel = "q1",
    ) 
        lines!(ax_5, t_train, x_train[:,7], label = "Raw") 
        lines!(ax_5, t_train, x_train_GP[:,7], label = "GP") 
    ax_6 = Axis(
        fig_train_GP[4:6,3],
        ylabel = "q2",
    ) 
        lines!(ax_6, t_train, x_train[:,8], label = "Raw") 
        lines!(ax_6, t_train, x_train_GP[:,8], label = "GP") 
    ax_7 = Axis(
        fig_train_GP[7:9,3],
        ylabel = "q3",
    ) 
        lines!(ax_7, t_train, x_train[:,9], label = "Raw") 
        lines!(ax_7, t_train, x_train_GP[:,9], label = "GP") 
    ax_8 = Axis( 
        fig_train_GP[10:12,3], 
        xlabel = "t (s)", 
        ylabel = "q4", 
    ) 
        lines!(ax_8, t_train, x_train[:,10], label = "Raw") 
        lines!(ax_8, t_train, x_train_GP[:,10], label = "GP") 
    ax_9 = Axis( 
        fig_train_GP[1:4,4], 
        ylabel = "w1", 
    ) 
        lines!(ax_9, t_train, x_train[:,11], label = "Raw") 
        lines!(ax_9, t_train, x_train_GP[:,11], label = "GP") 
    ax_10 = Axis( 
        fig_train_GP[5:8,4], 
        ylabel = "w2", 
    ) 
        lines!(ax_10, t_train, x_train[:,12], label = "Raw") 
        lines!(ax_10, t_train, x_train_GP[:,12], label = "GP") 
    ax_11 = Axis( 
        fig_train_GP[9:12,4], 
        xlabel = "t (s)", 
        ylabel = "w3", 
    ) 
        lines!(ax_11, t_train, x_train[:,13], label = "Raw") 
        lines!(ax_11, t_train, x_train_GP[:,13], label = "GP") 

    return fig_train_GP 
 end 

 export plot_quad_x_train_GP 

## ============================================ ##

"""
Plot training and predicted training data 
""" 
function plot_train_test( 
    x_train,            # [N,3] matrix of training data 
    x_test,             # [N,3] matrix of testing data 
    N_train = 0,        # number of training points to plot 
    fig     = nothing   # figure handle 
) 

    # create figure 
    if isnothing(fig) 
        fig = Figure() 
        ax = Axis3( fig[1,1] ) 
    else 
        ax = fig.content[1] 
    end 
    
    if N_train == 0 
        ax.title = "Trajectory of Training and Testing Data" 
    else 
        ax.title = string( "Trajectory of Training and Testing Data, N train points = ", N_train ) 
    end 

    # plot training data 
    lines_train = lines!( ax, x_train[:,1], x_train[:,2], x_train[:,3], linewidth = 2, label = "Training Data" )
    lines_test  = lines!( ax, x_test[:,1], x_test[:,2], x_test[:,3], linewidth = 2, label = "Testing Data" ) 

    axislegend( ax, position = :rt ) 

    return fig 
end 

export plot_train_test


## ============================================ ##

"""
Plot training and predicted training data 
""" 
function plot_train_pred( 
    x_train,                    # [N,3] matrix of training data 
    x_train_pred,               # [N,3] matrix of predicted training data 
    N_points      = 0,          # number of points to plot 
    fig           = nothing     # figure handle 
) 

    # create figure 
    if isnothing(fig) 
        fig = Figure() 
        ax  = Axis3( fig[1,1] ) 
    else 
        ax  = fig.content[1] 
    end 

    if N_points == 0 
        ax.title = "Trajectory of Predicted Training Data" 
    else 
        ax.title = string( "Trajectory of Predicted Training Data, N points = ", N_points ) 
    end 

    # plot training data 
    lines_train = lines!( ax, x_train[:,1], x_train[:,2], x_train[:,3], linewidth = 2, label = "Training Data" )
    lines_train_pred = lines!( ax, x_train_pred[:,1], x_train_pred[:,2], x_train_pred[:,3], color = :red, linewidth = 2, label = "Predicted Training Data" ) 

    axislegend( ax, position = :rt ) 

    return fig 
end 

export plot_train_pred 


## ============================================ ##
# add title to Axis3 of CairoMakie figures 

""" 
Add title to Axis3 of CairoMakie figures 
""" 
function add_title3d( 
    fig,        # figure handle 
    title       # title string 
) 
    ax = fig.content[1] 
    ax.title = title 

    return fig 
end 

export add_title3d 

## ============================================ ##
# plot Cartesian axes 

"Plot x, y, and z Cartesian axes "
function plot_axes3d( 
    r   = 0.5,            # radius of axes 
    fig = nothing,      # figure handle 
) 

    if isnothing(fig) 
        fig = Figure() 
        Axis3(fig[1, 1]) 
    end 

    xyz = [ zeros(3) for i in 1:3 ] 
    uvw = r .* [ [1,0,0] , [0,1,0] , [0,0,1] ] 

    width = r/20
    fig = plot_vector3d( [ xyz[1] ] , [ uvw[1] ], nothing, width, :red ) 
    fig = plot_vector3d( [ xyz[2] ] , [ uvw[2] ], fig, width, :blue ) 
    fig = plot_vector3d( [ xyz[3] ] , [ uvw[3] ], fig, width, :green  )  

    return fig 
end

export plot_axes3d 

## ============================================ ## 

""" 
Plot a 3D line 

Example usage: 

    x = collect( range(-pi, pi, 100) ) 
    y = sin.(x) 
    z = cos.(x) 

    fig = plot_3d( x, y, z )
"""
function plot_line3d( xyz, fig = nothing ) 
    plot_line3d( xyz[:,1], xyz[:,2], xyz[:,3], fig ) 
end 

function plot_line3d( 
    x,              # [N,1] grid of points 
    y,              # [N,1] grid of points 
    z,              # [N,1] grid of points  
    fig = nothing,  # figure handle 
) 

    if isnothing(fig) 
        fig = Figure() 
        Axis3(fig[1, 1]) 
    end 

    # plot orbit 
    lines!( x, y, z; linewidth = 2 ) 

    return fig 
end 
    
export plot_line3d  

## ============================================ ## 

""" 
Plot an orbit 

Example usage: 

    x = collect( range(-pi, pi, 100) ) 
    y = sin.(x) 
    z = cos.(x) 

    fig = plot_orbit( [x y z] )
"""
function plot_orbit( 
    rv,                 # [N,3] matrix of state vectors 
    fig    = nothing,   # figure handle 
    labels = false      # boolean for labeling start and end points 
) 

    text_offset = (0,10) 

    if isnothing(fig) 
        fig = Figure() 
        Axis3(fig[1, 1]) 
    end 

    # if isnothing(fig) 
    #     fig = Figure() 
    #     Axis3(fig[1, 1], aspect=DataAspect(), 
    #         xlabel = "X (km)", ylabel = "Y (km)", zlabel = "Z (km)", 
    #         title = "Transfer Solution") 
    # end 

    # plot orbit 
    lines!( rv[:,1], rv[:,2], rv[:,3]; linewidth = 2 ) 
    scatter!( rv[1,1], rv[1,2], rv[1,3]; marker = :circle, markersize = 10, color = :black ) 
    scatter!( rv[end,1], rv[end,2], rv[end,3]; marker = :utriangle, markersize = 10, color = :black ) 

    # add labels 
    if labels 
        text!( rv[1,1], rv[1,2], rv[1,3]; text = "start", color = :gray, offset = text_offset, align = (:center, :bottom) ) 
        text!( rv[end,1], rv[end,2], rv[end,3]; text = "end", color = :gray, offset = text_offset, align = (:center, :bottom) ) 
    end 

    Auto() 

    return fig 
end 
    
export plot_orbit 

## ============================================ ## 

# colormap options: 
#   jblue 
#   copper 
#   diverging_tritanopic_cwr_75_98_c20_n256 <-- this one 

"""
Plot a surface with a colorbar 

Example usage: 

    x = y = range(-pi, pi, 100)
    z = sin.(x) .* cos.(y') 

    fig = plot_surface( x, y, z ) 
"""
function plot_surface( 
    x,                  # [N,1] grid of points 
    y,                  # [N,1] grid of points 
    z,                  # [N,N] grid of points evaluated at x and y 
    fig   = nothing,    # figure handle 
    alpha = 1.0,        # transparency 
) 

    fignothing = false 
    if isnothing(fig) 
        fignothing = true 
        fig = Figure() 
        Axis3(fig[1, 1]) 
    end 

    cmap = ( :diverging_tritanopic_cwr_75_98_c20_n256, alpha )
    hm   = CairoMakie.surface!( x, y, z, colormap = cmap ) 

    if fignothing 
        Colorbar( fig[1,2], hm, height = Relative(0.5) )
    end 

    return fig 
end 

export plot_surface 

## ============================================ ##

"""
Plot scatter 

Example usage: 

    x = y = range(-pi, pi, 100)
    z = sin.(x) .* cos.(y') 

    fig = plot_surface( x, y, z ) 
    fig = plot_contour3d( x, y, z ) 
    fig = plot_scatter3d( x, y, z ) 
"""
function plot_scatter3d( xyz, fig = nothing ) 
    plot_scatter3d( xyz[:,1], xyz[:,2], xyz[:,3], fig ) 
end 

function plot_scatter3d( 
    x,                      # [N,1] grid of points 
    y,                      # [N,1] grid of points 
    z,                      # [N,N] grid of points evaluated at x and y 
    fig    = nothing,       # figure handle 
    marker = :utriangle,    # marker type 
    color  = :black,        # marker color 
    text   = nothing,       # text to add to plot 
) 

    if isnothing(fig) 
        fig = Figure() 
        Axis3(fig[1, 1]) 
    end 

    if isequal(length(z), 1)
        CairoMakie.scatter!( x, y, z, marker = marker, markersize = 20, color = color, strokecolor = color ) 
        if !isnothing(text) 
            text!( x, y, z; text = text, color = :black, offset = (0,15), align = (:center, :bottom) ) 
        end
    else 
        hm = CairoMakie.scatter!( x, y, z, markersize = 5, color = color, strokecolor = color ) 
    end 

    return fig 
end 

export plot_scatter3d 

## ============================================ ##

""" 
Plot a contour with a colorbar 

Example usage: 

    x = y = range(-pi, pi, 100)
    z = sin.(x) .* cos.(y') 

    fig = plot_contour3d( x, y, z ) 
""" 
function plot_contour3d( 
    x,              # [N,1] grid of points 
    y,              # [N,1] grid of points 
    z,              # [N,N] grid of points evaluated at x and y 
    fig = nothing,  # figure handle 
    levels = 20,    # number of contour levels 
) 

    if isnothing(fig) 
        fig = Figure() 
        Axis3(fig[1, 1]) 
    end 

    hm  = CairoMakie.contour3d!(x, y, z, levels = levels) 

    if fignothing 
        clim = ( minimum(z), maximum(z) ) 
        Colorbar( fig[1, 2], limits = clim, height = Relative(0.5) )
    end 

    return fig 
end 

export plot_contour3d 

## ============================================ ## 

"""
Plot vector 

Example usage: 

    r   = 6378.0
    xyz = [ zeros(3) for i in 1:3 ] 
    uvw = r .* [ [1,0,0] , [0,1,0] , [0,0,1] ] 

    fig = plot_vector3d( [ xyz[1] ] , [ uvw[1] ], nothing, r/100, :red ) 
    fig = plot_vector3d( [ xyz[2] ] , [ uvw[2] ], fig, r/100, :blue ) 
    fig = plot_vector3d( [ xyz[3] ] , [ uvw[3] ], fig, r/100, :green ) 
"""
function plot_vector3d( 
    xyz,                        # [N] vector of (x,y,z) origin points 
    uvw,                        # [N] vector of (u,v,w) vector directions 
    fig    = nothing,           # figure handle 
    width  = norm(uvw[1])/100,  # arrow width 
    color  = :black,            # marker color 
    text   = nothing,           # text to add to plot 
) 

    # check type --> must be vectors of vectors 
    if xyz isa AbstractMatrix 
        xyz = m2vv(xyz)
    end 
    if uvw isa AbstractMatrix 
        uvw = m2vv(uvw)
    end 

    # adjust because stupid arrows plots the tails at the Point3f points 
    xyz += uvw 

    # convert to Points3f and Vec3f for arrows function 
    ps  = [ Point3f(x,y,z) for (x,y,z) in xyz ] 
    ns  = [ Vec3f(x,y,z) for (x,y,z) in uvw ] 

    if isnothing(fig) 
        fig = Figure() 
        Axis3(fig[1, 1]) 
    end 

    arrows!(  
        ps, ns, fxaa = true, # turn on anti-aliasing
        linecolor = color, arrowcolor = color,
        linewidth = width, arrowsize = 2 * width .* Vec3f(1, 1, 1),
        align = :center, 
    )

    if !isnothing(text) 
        if size(xyz, 1) > 1 
            error("Can only label one vector at time.")
        end 
        x = xyz[1][1] ; y = xyz[1][2] ; z = xyz[1][3] 
        text!( x, y, z; text = text, color = :black, offset = (0,15), align = (:center, :bottom) ) 
    end

    return fig 
end 

export plot_vector3d 





