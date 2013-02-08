module Alea.CUDA.Extension.NormalDistribution.ShawBrickman32

open Alea.CUDA

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
