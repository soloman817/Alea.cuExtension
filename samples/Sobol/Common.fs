module Sample.Sobol.Common

let numDirections = 32

let directions numDimensions =
    let directions = Array.zeroCreate<uint32> (numDimensions * numDirections)
        
    let directionsOffset = ref 0

    let setV (idx:int) (value:uint32) = directions.[!directionsOffset + idx] <- value
    let getV (idx:int) = directions.[!directionsOffset + idx]

    for dim = 0 to numDimensions - 1 do
        if dim = 0 then
            for i = 0 to numDirections - 1 do
                setV i (1u <<< (31 - i))
        else
            let d = SobolJoeKuo_6_21201.d.[dim]
            let a = SobolJoeKuo_6_21201.a.[dim]
            let o = SobolJoeKuo_6_21201.o.[dim]

            // the first direction numbers (up to the degree of the polynomial)
            // are simply v[i] = m[i] / 2^i (stored in Q0.32 format)
            for i = 0 to (d - 1) do
                setV i (SobolJoeKuo_6_21201.m.[o + i] <<< (31 - i))
                // the following is the old method, which is slow on x64 platform
                //setV i (m.[i] <<< (31 - i))

            // the remaining direction numbers are computed as described in the Bratley and Fox paper according to
            // v[i] = a[1]v[i-1] ^ a[2]v[i-2] ^ ... ^ a[v-1]v[i-d+1] ^ v[i-d] ^ v[i-d]/2^d
            for i = d to numDirections - 1 do
                setV i (getV(i-d) ^^^ (getV(i-d) >>> d))

                for j = 1 to d - 1 do
                    setV i (getV(i) ^^^ (((a >>> (d - 1 - j)) &&& 1u) * getV(i-j)))

        directionsOffset := !directionsOffset + numDirections

    directions

/// Transforms an uint32 random number to a float value 
/// on the interval [0,1] by dividing by 2^32-1
let [<ReflectedDefinition>] toFloat32 (x:uint32) = float32(x) * 2.3283064E-10f

/// Transforms an uint32 random number to a float value 
/// on the interval [0,1] by dividing by 2^32-1
let [<ReflectedDefinition>] toFloat64 (x:uint32) = float(x) * 2.328306437080797e-10   


