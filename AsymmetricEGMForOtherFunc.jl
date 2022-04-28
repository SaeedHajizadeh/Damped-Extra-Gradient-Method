import Base.show
using Plots
using Pkg
using DifferentialEquations
using LinearAlgebra

using NLsolve
# using PyPlot
import Contour: contours, levels, level, lines, coordinates



function show(io::IO, m::MIME"text/html", tuple::Tuple{Plots.Plot,Vararg{Plots.Plot}})
    inline(plot( tuple... , size=(1000,1000)))
end


function f(x)
    (x-3)*(x-1)*(x+1)*(x+3)
end
function gradf(x)
    (4*x^3 - 20*x)
end
function g(y)
    (y-3)*(y-1)*(y+1)*(y+3)
end
function gradg(y)
    (4*y^3 - 20*y)
end






A = 50
β = 132 #Upperbound that holds whenever |x|<3
ρ = 20 #Computed at the origin, which has the largest negative curavture
α = -ρ + A^2/(2*(β+ρ))

### Define algorithm paramters
    η = 4*ρ
    λ = 0.1#1/(8+16*ρ/α)
    T = 10000
    ϵ = 0.0000001 #Level of precision that computations should be run t







function extragradient_step(A,x,y)

    μ = η - ρ
    L = β + η
    s = μ/L^2 #Objective is beta+A-smooth, so this is a natural stepsize choice
    x1 = x - s * (gradf(x) +A*y)
    y1 = y + s * (x*A - gradg(y))
    x2 = x - λ * s * (gradf(x1) +A*y1)
    y2 = y + λ * s * (x1*A - gradg(y1))
    return (x2,y2)
end

# Our tests will all start at a point and record the first T iterates produced by the PPM
function runEGTest(A,(x0,y0))
    x = zeros(T)
    y = zeros(T)
    x[1] = x0
    y[1] = y0
    for i in 2:T
        (px, py) = extragradient_step(A,x[i-1], y[i-1])
        x[i] = px
        y[i] = py
    end
    return (x,y)
end



### Define approximate prox step computation using gradient descent ascent subroutine
function proxB(A,x,y)
    #theory constants for subproblem
    GNonAccelerated = 0
    μ = η - ρ
    L = β + η
    stepsize = μ/L^2

    xcur=x
    ycur=y
    xprev=xcur
    yprev=ycur
    xcur = xcur +stepsize*(η*(x-xcur) -gradf(xcur) -A*ycur)
    ycur = ycur +stepsize*(η*(y-ycur) -gradg(ycur) +A*xcur)

    GNonAccelerated += 1

    while sqrt((xcur-xprev)^2 + (ycur-yprev)^2)>ϵ
        xprev=xcur
        yprev=ycur
        xcur = xcur +stepsize*(η*(x-xcur) -gradf(xcur) -A*ycur)
        ycur = ycur +stepsize*(η*(y-ycur) -gradg(ycur) +A*xcur)
        GNonAccelerated += 1
    end
    return (xcur,ycur,GNonAccelerated)
end

# Our tests will all start at a point and record the first T iterates produced by the PPM
function runPPMNonAccelerated(A,(x0,y0))
    x = zeros(T)
    y = zeros(T)
    GNonAccelerated = 0
    iteration = 0
    x[1] = x0
    y[1] = y0
    for i in 2:T
        (px, py, iteration) = proxB(A,x[i-1], y[i-1])
        x[i] = (1-λ)*x[i-1] + λ*px
        y[i] = (1-λ)*y[i-1] + λ*py
        GNonAccelerated += iteration
    end
    println("(" , x0 , " , " , y0 , ")" , " : " , GNonAccelerated)
    return (x,y)
end




testStarts = [(4,0),(0,4),(-4,0),(0,-4),(1,0),(0,1),(-1,0),(0,-1),(4,2),(2,4),(4,-2),(2,-4),(-4,2),(-2,4),(-4,-2),(-2,-4)]


function makePlot(A, algorithm)
    tests = Array{Tuple{Array{Float64,1},Array{Float64,1}}}(undef, 16)
    for i in 1:16
        tests[i] = algorithm(A,testStarts[i])
    end
    p = plot(tests[1], xlims=(-4.1,4.1),ylims=(-4.1,4.1), size=(500,500),xtickfontsize=18,ytickfontsize=18,xguidefontsize=18, st=:path, legend=false, fmt = :png, arrow=1, linewidth=2)#, xaxis=string("A=",A))
    for i in 2:16
        plot!(p,tests[i], st=:path, arrow=1, linewidth=2)
    end
    return p
end


function makePlotNonaccelerated(A, algorithm)
    tests = Array{Tuple{Array{Float64,1},Array{Float64,1}}}(undef, 16)
    for i in 1:16
        tests[i] = algorithm(A,testStartsNonAccelerated[i])
    end
    p = plot(tests[1], xlims=(-4.2,4.2),ylims=(-4,4), size=(500,500),xtickfontsize=18,ytickfontsize=18,xguidefontsize=18, markershape = :circle, st=:path, legend=false, fmt = :png, arrow=1, linewidth=2)#, xaxis=string("A=",A))
    for i in 2:16
        plot!(p,tests[i], markershape = :circle, st=:path, arrow=1, linewidth=2)
    end
    return p
end









p = makePlotNonaccelerated(A, runPPMNonAccelerated)
p = makePlot(A, runPPMNonAccelerated)
p = makePlot(A, runEGTest)
