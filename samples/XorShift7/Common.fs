module Sample.XorShift7.Common

let LCG_A = 1664525u
let LCG_C = 1013904223u

let generateStartState (seed:uint32) =
    let state = Array.zeroCreate 8
    state.[0] <- seed
    for i = 1 to 7 do state.[i] <- LCG_A * state.[i - 1] + LCG_C
    state

/// Transforms an uint32 random number to a float value 
/// on the interval [0,1] by dividing by 2^32-1
let [<ReflectedDefinition>] toFloat32 (x:uint32) = float32(x) * 2.3283064E-10f

/// Transforms an uint32 random number to a float value 
/// on the interval [0,1] by dividing by 2^32-1
let [<ReflectedDefinition>] toFloat64 (x:uint32) = float(x) * 2.328306437080797e-10   
