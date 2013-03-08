module Alea.CUDA.Extension.TriDiag

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Util

/// Parallel tridiagonal linear system solver. The algorithm is implemented according to 
/// 
///     http://www.cse.uiuc.edu/courses/cs554/notes/09_tridiagonal.pdf
///
/// Optimized version for n <= max number of threads per block.
///
/// again with multiple if (or loops). 
/// Note that one might need then one temporary variable ltemp, utemp, Htemp, per thread block of size 'size'. 
///   
///     n      the dimension of the tridiagonal system, must fit into one block
///     l      lower diagonal
///     d      diagonal
///     u      upper diagonal
///     h      right hand side and solution at exit
///
//let [<ReflectedDefinition>] inline triDiagPcrSingleBlock n (dl:DevicePtr<'T>) (dd:DevicePtr<'T>) (du:DevicePtr<'T>) (dh:DevicePtr<'T>) =
let [<ReflectedDefinition>] inline triDiagPcrSingleBlock n (l:DevicePtr<'T>) (d:DevicePtr<'T>) (u:DevicePtr<'T>) (h:DevicePtr<'T>) =
        let rank = threadIdx.x

        // this breaks, why?
        // let shared = __shared__<'T>(4*n)
        // let l = shared.Ptr(0)
        // let d = shared.Ptr(n)
        // let u = shared.Ptr(2*n)
        // let h = shared.Ptr(3*n)

        // let shared = __extern_shared__()
        // let l = shared.Reinterpret<'T>()
        // let d = l + n
        // let u = d + n
        // let h = u + n
        // 
        // l.[rank] <- dl.[rank]
        // d.[rank] <- dd.[rank]
        // u.[rank] <- du.[rank]
        // h.[rank] <- dh.[rank]
        // 
        //__syncthreads()

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

        __syncthreads()  


let inline triDiag () = cuda {

    let! kernel =     
        <@ fun n (l:DevicePtr<'T>) (d:DevicePtr<'T>) (u:DevicePtr<'T>) (h:DevicePtr<'T>) ->          
            triDiagPcrSingleBlock n l d u h @> |> defineKernelFunc

    return PFunc(fun (m:Module) (l:'T[]) (d:'T[]) (u:'T[]) (h:'T[])->
        let n = d.Length
        let numThreads = n
        let maxThreads = m.Worker.Device.Attribute DeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        let numBlocks = 1
        use dl = m.Worker.Malloc(l)
        use dd = m.Worker.Malloc(d)
        use du = m.Worker.Malloc(u)
        use dh = m.Worker.Malloc(h)
        let lp = LaunchParam(1, numThreads, 4*n*sizeof<'T>)
        kernel.Launch m lp n dl.Ptr dd.Ptr du.Ptr dh.Ptr
        dh.ToHost()) }


