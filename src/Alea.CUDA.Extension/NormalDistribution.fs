module Alea.CUDA.Extension.NormalDistribution

open Alea.CUDA

module ShawBrickman32 =
    [<ReflectedDefinition>]
    let inverseNormalCdf (u:float32) =
        let half_minus_u = 0.5f - u
        let mutable x = DeviceFunction.copysignf(2.0f*u, half_minus_u)
        if half_minus_u < 0.0f then x <- x + 2.0f
        let v = -DeviceFunction.__logf x

        let mutable p = 1.1051591117060895699e-4f
        p <- p*v + 0.011900603295838260268f
        p <- p*v + 0.23753954196273241709f
        p <- p*v + 1.3348090307272045436f
        p <- p*v + 2.4101601285733391215f
        p <- p*v + 1.2533141012558299407f

        let mutable q = 2.5996479253181457637e-6f
        q <- q*v + 0.0010579909915338770381f
        q <- q*v + 0.046292707412622896113f
        q <- q*v + 0.50950202270351517687f
        q <- q*v + 1.8481138350821456213f
        q <- q*v + 2.4230267574304831865f
        q <- q*v + 1.0f
        -DeviceFunction.__fdividef(p, q) * DeviceFunction.copysignf(v, half_minus_u)

module ShawBrickman =
    [<ReflectedDefinition>]
    let inverseNormalCdf v =
        let P1 = 1.2533141373154822808
        let P2 = 5.1066889621115428678
        let P3 = 8.1217283420696808953
        let P4 = 6.478609976611846182
        let P5 = 2.7951801400933882501
        let P6 = 0.66364526254138709386
        let P7 = 0.08573550202628730526
        let P8 = 0.0058460183117205552962
        let P9 = 0.00020014949980131463558
        let P10 = 3.1784054957970863347e-6
        let P11 = 2.0424451498824914329e-8
        let P12 = 4.0218581361785749956e-11
        let P13 = 1.0259243358417535196e-14
        let Q1 = 1.0
        let Q2 = 4.574548279686728261
        let Q3 = 8.3390097367731850572
        let Q4 = 7.8130148358693434858
        let Q5 = 4.0712687287892113712
        let Q6 = 1.2037213955961447198
        let Q7 = 0.20007493257106482488
        let Q8 = 0.018179904683140742758
        let Q9 = 0.00086377649498937910731
        let Q10 = 0.000020082359687444181104
        let Q11 = 2.0563457281652766307e-7
        let Q12 = 7.6592682221644671397e-10
        let Q13 = 6.6234512664266726236e-13
    
        let sgn = if v >= 0.5 then 1 else -1
        let vv = if sgn = -1 then v else 1.0 - v
        let z = -log(2.0 * vv)

        let num = (P1+z*(P2+z*(P3+z*(P4+z*(P5+z*(P6+z*(P7+z*(P8+z*(P9+z*(P10+z*(P11+z*(P12+P13*z))))))))))))
        let den = (Q1+z*(Q2+z*(Q3+z*(Q4+z*(Q5+z*(Q6+z*(Q7+z*(Q8+z*(Q9+z*(Q10+z*(Q11+z*(Q12+Q13*z))))))))))))
    
        float(sgn) * z * num / den

module ShawBrickmanExtended =
    [<ReflectedDefinition>]
    let inverseNormalCdf v =
        let P1 = 1.2533141373154989811
        let P2 = 5.5870183514814983104
        let P3 = 9.9373788223105148469
        let P4 = 9.11745910783758368
        let P5 = 4.6865666928347513004
        let P6 = 1.3841649695441184484
        let P7 = 0.23434950424605615377
        let P8 = 0.022306824510199724768
        let P9 = 0.0011538603964070818722
        let P10 = 0.000030796620691411567563
        let P11 = 3.9115723028719510263e-7
        let P12 = 2.0589573468131996933e-9
        let P13 = 3.3944224725087481454e-12
        let P14 = 7.3936480912071325978e-16 
        let Q1 = 1.0
        let Q2 = 4.9577956835689939051
        let Q3 = 9.9793129245112074476
        let Q4 = 10.574454910639356539
        let Q5 = 6.4247521669505779535
        let Q6 = 2.3008904864351121026
        let Q7 = 0.48545999687461771635
        let Q8 = 0.059283082737079006352
        let Q9 = 0.0040618506206078995821
        let Q10 = 0.00014919732843986856251
        let Q11 = 2.7477061392049947066e-6
        let Q12 = 2.2815008011613816939e-8
        let Q13 = 7.0445790305953963457e-11
        let Q14 = 5.1535907808963289678e-14

        let sgn = if v >= 0.5 then 1 else -1
        let vv = if sgn = -1 then v else 1.0 - v
        let z = -log(2.0 * vv)   

        let num = (P1+z*(P2+z*(P3+z*(P4+z*(P5+z*(P6+z*(P7+z*(P8+z*(P9+z*(P10+z*(P11+z*(P12+z*(P13+P14*z)))))))))))))
        let den = (Q1+z*(Q2+z*(Q3+z*(Q4+z*(Q5+z*(Q6+z*(Q7+z*(Q8+z*(Q9+z*(Q10+z*(Q11+z*(Q12+z*(Q13+Q14*z)))))))))))))
 
        float(sgn) * z * num / den

module Acklam =
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

module AbramowitzStegun =
    /// <summary>
    /// Cumulative normal distribution function. 
    /// </summary>
    /// <remarks>   
    /// Calculates the cumulative normal distribution  
    /// 
    ///     \Phi(x) = \int_{-\infty}^x \phi(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} \exp\left(\frac{-u^2}{2}\right) du
    /// 
    /// by a simple rational approximation. It uses the asymptotic expansion (26.2.12)
    /// in Abramowitz-Stegun, p. 932. See e.g. http://www.math.sfu.ca/~cbm/aands/page_932.htm
    ///
    /// For most cases this version is accurage enough, in particular for x \in [-6, 6].
    /// </remarks>
    /// <param name="x"></param>
    [<ReflectedDefinition>]
    let normalCdf x =
        let a1 =  0.31938153
        let a2 = -0.356563782
        let a3 =  1.781477937
        let a4 = -1.821255978
        let a5 =  1.330274429

        let K = 1.0 / (1.0 + 0.2316419 * abs(x))
        let cnd = 0.3989422804014327
                    * exp(-0.5 * x * x)
                    * (K * (a1 + K * (a2 + K * (a3 + K * (a4 + K * a5)))))

        if x > 0.0 then 1.0 - cnd else cnd

module NormalCdfErf =
    /// <summary>
    /// Cumulative normal distribution function. 
    /// </summary>
    /// <remarks>   
    /// Calculates the cumulative normal distribution  
    /// 
    ///   \Phi(x) = \int_{-\infty}^x \phi(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} \exp\left(\frac{-u^2}{2}\right) du
    /// 
    /// based on the erf implementation of the Sun FDMLib version 5.3 and http://www.netlib.org/specfun/erf.
    /// It is more accurate than cdfAbramowitzStegun. 
    /// </remarks>
    /// <param name="x"></param>

    // http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__MATH__DOUBLE_gbd196c4f3bc4260ffe99944b2400b951.html#gbd196c4f3bc4260ffe99944b2400b951
    //TODO
    let erfCuda x = x

    // http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__MATH__DOUBLE_ge5fb0600e76f923d822e51b6148a9d1a.html#ge5fb0600e76f923d822e51b6148a9d1a
    // TODO
    let erfcCuda x = x

    let [<ReflectedDefinition>] bySqrt2 = 0.707106781186547524400845 // = 1/sqrt(2)

    let [<ReflectedDefinition>] normalCdfErf (x : float) =
        if 0.0 <= x then 
            0.5 + erfCuda(x * bySqrt2) * 0.5
        else
            erfcCuda(-x * bySqrt2) * 0.5


    // TODO 
    // we should also provide the SUNLib erf version of the normal cdf function 
    // to test accuracy of NVIDIA erf and erfc or use it to test against it


