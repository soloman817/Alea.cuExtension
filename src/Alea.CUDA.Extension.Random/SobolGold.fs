module Alea.CUDA.Extension.Random.SobolGold

/// Reorder points aligned as produced by Sobol kernel to consecutive points.
let reorderPoints dimensions vectors (points:'T[]) =
    Array.init points.Length (fun i -> 
        let vi = i / dimensions 
        let di = i % dimensions 
        points.[di * vectors + vi])

/// Returns the position of the first (least significant) bit set in the word i.
let ffs(i:int) =
    let mutable v = i
    let mutable count = 0

    if v = 0 then
        count <- 0
    else
        count <- 2
        if v &&& 0xffff = 0 then
            v <- v >>> 16
            count <- count + 16
        if v &&& 0xff = 0 then
            v <- v >>> 8
            count <- count + 8
        if v &&& 0xf = 0 then
            v <- v >>> 4
            count <- count + 4
        if v &&& 0x3 = 0 then
            v <- v >>> 2
            count <- count + 2
        count <- count - (v &&& 0x1)

    count

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
    static let nDirections = Sobol.nDirections

    // generate the direction numbers
    let directions = Sobol.directions dim
       
    // the sequence index 
    let mutable index = 0

    // the actual state of the generator, one uint32 per dimension
    // note that the direction numbers contain dim consecutive blocks of nDirections direction numbers
    let mutable currentPoint = Array.init<uint32> dim (fun i -> directions.[i*nDirections])

    let reset() =
        for i = 0 to dim - 1 do
            currentPoint.[i] <- directions.[i*nDirections]
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
                        currentPoint.[i] <- currentPoint.[i] ^^^ directions.[i*nDirections + j]
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
                currentPoint.[i] <- currentPoint.[i] ^^^ directions.[i*nDirections + j]                
            index <- index + 1
            Array.copy currentPoint
       