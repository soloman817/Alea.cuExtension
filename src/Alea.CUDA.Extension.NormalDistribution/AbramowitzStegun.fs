module Alea.CUDA.Extension.NormalDistribution.AbramowitzStegun

open Alea.CUDA

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
