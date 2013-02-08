module Alea.CUDA.Extension.NormalDistribution.NormalCdfErf

open Alea.CUDA

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

