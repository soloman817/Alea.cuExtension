module Sample.TriDiag.GPU

open Alea.CUDA
open Alea.CUDA.Utilities

/// Parallel tridiagonal linear system solver. The algorithm is implemented according to 
/// 
///     http://www.cse.uiuc.edu/courses/cs554/notes/09_tridiagonal.pdf
///
/// Optimized version for n <= max number of threads per block.
///   
///     n      the dimension of the tridiagonal system, must fit into one block
///     l      lower diagonal
///     d      diagonal
///     u      upper diagonal
///     h      right hand side and solution at exit
///
[<ReflectedDefinition>]
let inline solve n (l:deviceptr<'T>) (d:deviceptr<'T>) (u:deviceptr<'T>) (h:deviceptr<'T>) =
    let rank = threadIdx.x

    let mutable ltemp = 0G
    let mutable utemp = 0G
    let mutable htemp = 0G
        
    let mutable span = 1
    while span < n do
              
        if rank < n then
            if rank - span >= 0 then
                ltemp <- if d.[rank - span] <> 0G then -l.[rank] / d.[rank - span] else 0G
            else
                ltemp <- 0G
            if rank + span < n then
                utemp <- if d.[rank + span] <> 0G then -u.[rank] / d.[rank + span] else 0G
            else
                utemp <- 0G
            htemp <- h.[rank]
            
        __syncthreads()

        if rank < n then    
            if rank - span >= 0 then              
                d.[rank] <- d.[rank] + ltemp * u.[rank - span]
                htemp <- htemp + ltemp * h.[rank - span]
                ltemp <-ltemp * l.[rank - span]
                
            if rank + span < n then               
                d.[rank] <- d.[rank] + utemp * l.[rank + span]
                htemp <- htemp + utemp * h.[rank + span]
                utemp <- utemp * u.[rank + span]
                           
        __syncthreads()
            
        if rank < n then
            l.[rank] <- ltemp
            u.[rank] <- utemp
            h.[rank] <- htemp

        __syncthreads()

        span <- 2*span
               
    if rank < n then
        h.[rank] <- h.[rank] / d.[rank]


