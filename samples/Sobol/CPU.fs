module Sample.Sobol.CPU

/// Reorder points aligned as produced by Sobol kernel to consecutive points.
let reorderPoints numDimensions numVectors (points:'T[]) =
    let result = Array.zeroCreate points.Length
    for i = 0 to numVectors - 1 do
        for d = 0 to numDimensions - 1 do
            result.[d * numVectors + i] <- points.[i * numDimensions + d]
    result

/// Find rightmost zero bit in i.
let rmz (i : uint32) =
    let mutable n = i
    let mutable j = 0
    while n &&& 1u > 0u do
        n <- n >>> 1
        j <- j + 1
    j

/// Reference Sobol generator. 
/// The generator maintains state to extend sequence and provides 
/// jump ahead functionality to start at every sequence index.
/// The implementation is a bit different and follows the original paper of Bratley and Fox.
type Sobol(dim : int, startIndex : int) =         
    static let log2 = log 2.0

    // number of direction vectors is fixed to 32
    static let numDirections = Common.numDirections

    // generate the direction numbers
    let directions = Common.directions dim
       
    // the sequence index 
    let mutable index = 0

    // the actual state of the generator, one uint32 per dimension
    // note that the direction numbers contain dim consecutive blocks of nDirections direction numbers
    let mutable currentPoint = Array.init<uint32> dim (fun i -> directions.[i * numDirections])

    let reset() =
        for i = 0 to dim - 1 do
            currentPoint.[i] <- directions.[i * numDirections]
        index <- 0

    let jumpTo idx =
        if idx = 0 then
            reset()
        else
            let n = uint32(idx)  
            let ops = int((log (float n))/log2 + 1.0)
            let grayCode = n ^^^ (n >>> 1)
            for i = 0 to dim - 1 do
                currentPoint.[i] <- 0u
                for j = 0 to ops - 1 do
                    if (grayCode >>> j) &&& 1u <> 0u then 
                        currentPoint.[i] <- currentPoint.[i] ^^^ directions.[i * numDirections + j]
        index <- idx      

    do jumpTo startIndex

    member this.Dimensions = dim
    member this.Directions = directions
    member this.CurrentPoint = currentPoint

    /// Reset generator to start at first point.
    member this.Reset = reset()

    /// Jump forward to point of given index.
    member this.JumpTo index = jumpTo index               
                
    /// Generate next point at current index.
    /// Note that this implementation does not produce the first point with all zeros,
    /// which is desirable anyway because there we fail with inverse cumulative normal.
    member this.NextPoint = 
        if index = 0 then 
            index <- index + 1
            Array.copy currentPoint            
        else
            let j = rmz (uint32 index)
            for i = 0 to dim - 1 do
                currentPoint.[i] <- currentPoint.[i] ^^^ directions.[i * numDirections + j]                
            index <- index + 1
            Array.copy currentPoint
              