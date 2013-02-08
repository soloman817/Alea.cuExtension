module Alea.CUDA.Extension.NormalDistribution.Acklam

open Alea.CUDA

let [<ReflectedDefinition>] oneOverSqrtPi = 0.564189583547756286948079
let [<ReflectedDefinition>] sqrtTwoPi = 2.50662827463100050241577
let [<ReflectedDefinition>] sqrtTwo = 1.4142135623730950488016887

[<ReflectedDefinition>]
let inverseNormalCdf v =
    let a0 = -3.969683028665376e+01
    let a1 =  2.209460984245205e+02
    let a2 = -2.759285104469687e+02
    let a3 =  1.383577518672690e+02
    let a4 = -3.066479806614716e+01 
    let a5 =  2.506628277459239e+00

    let b0 = -5.447609879822406e+01
    let b1 =  1.615858368580409e+02
    let b2 = -1.556989798598866e+02 
    let b3 =  6.680131188771972e+01
    let b4 = -1.328068155288572e+01

    let c0 = -7.784894002430293e-03  
    let c1 = -3.223964580411365e-01 
    let c2 = -2.400758277161838e+00  
    let c3 = -2.549732539343734e+00 
    let c4 =  4.374664141464968e+00   
    let c5 =  2.938163982698783e+00

    let d0 = 7.784695709041462e-03  
    let d1 = 3.224671290700398e-01
    let d2 = 2.445134137142996e+00  
    let d3 = 3.754408661907416e+00
        
    let aaa0 = 1.161110663653770e-002
    let aaa1 = 3.951404679838207e-001
    let aaa2 = 2.846603853776254e+001
    let aaa3 = 1.887426188426510e+002
    let aaa4 = 3.209377589138469e+003
         
    let bbb0 = 1.767766952966369e-001
    let bbb1 = 8.344316438579620e+000
    let bbb2 = 1.725514762600375e+002
    let bbb3 = 1.813893686502485e+003
    let bbb4 = 8.044716608901563e+003
    
    let ccc0 = 2.15311535474403846e-8
    let ccc1 = 5.64188496988670089e-1
    let ccc2 = 8.88314979438837594e00
    let ccc3 = 6.61191906371416295e01
    let ccc4 = 2.98635138197400131e02
    let ccc5 = 8.81952221241769090e02
    let ccc6 = 1.71204761263407058e03
    let ccc7 = 2.05107837782607147e03
    let ccc8 = 1.23033935479799725E03
        
    let ddd0 = 1.00000000000000000e00
    let ddd1 = 1.57449261107098347e01
    let ddd2 = 1.17693950891312499e02
    let ddd3 = 5.37181101862009858e02
    let ddd4 = 1.62138957456669019e03
    let ddd5 = 3.29079923573345963e03
    let ddd6 = 4.36261909014324716e03
    let ddd7 = 3.43936767414372164e03
    let ddd8 = 1.23033935480374942e03

    let ppp0 = 1.63153871373020978e-2
    let ppp1 = 3.05326634961232344e-1
    let ppp2 = 3.60344899949804439e-1
    let ppp3 = 1.25781726111229246e-1
    let ppp4 = 1.60837851487422766e-2
    let ppp5 = 6.58749161529837803e-4

    let qqq0 = 1.00000000000000000e00
    let qqq1 = 2.56852019228982242e00
    let qqq2 = 1.87295284992346047e00
    let qqq3 = 5.27905102951428412e-1
    let qqq4 = 6.05183413124413191e-2
    let qqq5 = 2.33520497626869185e-3
        
    let mutable q = 0.0
    let mutable t = 0.0
    let mutable u = 0.0
    let mutable p = v
        
    if (p < 1.0 - p) then q <- p else q <- 1.0 - p

    if (q > 0.02425) then
        // Rational approximation for central region.  
        u <- q - 0.5
        t <- u*u
        u <- u*(((((a0*t+a1)*t+a2)*t+a3)*t+a4)*t+a5) / (((((b0*t+b1)*t+b2)*t+b3)*t+b4)*t+1.0)
    else
        // Rational approximation for tail region. 
        t <- sqrt(-2.0*log(q))
        u <- (((((c0*t+c1)*t+c2)*t+c3)*t+c4)*t+c5) / ((((d0*t+d1)*t+d2)*t+d3)*t+1.0)
        
    // The relative error of the approximation has absolute value less than 1.15e-9.  
    // One iteration of Halley's rational method (third order) gives full machine precision.
        
    let mutable yyy = abs u
    let mutable zzz = 0.0

    if yyy <= 0.46875*sqrtTwo then
        zzz <- yyy*yyy
        yyy <- u*((((aaa0*zzz+aaa1)*zzz+aaa2)*zzz+aaa3)*zzz+aaa4) / ((((bbb0*zzz+bbb1)*zzz+bbb2)*zzz+bbb3)*zzz+bbb4)
        t <- 0.5+yyy-q
    else
        zzz <- exp(-yyy*yyy/2.0)/2.0
        if yyy <= 4.0 then
            yyy <- yyy / sqrtTwo
            yyy <- ((((((((ccc0*yyy+ccc1)*yyy+ccc2)*yyy+ccc3)*yyy+ccc4)*yyy+ccc5)*yyy+ccc6)*yyy+ccc7)*yyy+ccc8)
                    / ((((((((ddd0*yyy+ddd1)*yyy+ddd2)*yyy+ddd3)*yyy+ddd4)*yyy+ddd5)*yyy+ddd6)*yyy+ddd7)*yyy+ddd8)
            yyy <- zzz*yyy
        else
            zzz <- zzz*sqrtTwo/yyy
            yyy <- 2.0/(yyy*yyy)
            yyy <- yyy*(((((ppp0*yyy+ppp1)*yyy+ppp2)*yyy+ppp3)*yyy+ppp4)*yyy+ppp5)
                    / (((((qqq0*yyy+qqq1)*yyy+qqq2)*yyy+qqq3)*yyy+qqq4)*yyy+qqq5)
            yyy <- zzz*(oneOverSqrtPi-yyy)

        if u < 0.0 then t <- yyy-q else t <- 1.0-yyy-q  // error 
                
    t <- t*sqrtTwoPi*exp(u*u/2.0)       // f(u)/df(u) 
    u <- u - t/(1.0+u*t/2.0)            // Halley's method 
        
    if p > 0.5 then u <- -u
    u

