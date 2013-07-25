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
///     n      the dimension of the tridiagonal system, must fit into one block
///     l      lower diagonal
///     d      diagonal
///     u      upper diagonal
///     h      right hand side and solution at exit
///
let [<ReflectedDefinition>] inline triDiagPcrSingleBlock n (l:SharedPtr<'T>) (d:SharedPtr<'T>) (u:SharedPtr<'T>) (h:SharedPtr<'T>) =
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


let [<ReflectedDefinition>] inline triDiagPcrSingleBlockTempShared (size:int) (tid:int) 
                                                                   (a:SharedPtr<'T>) (b:SharedPtr<'T>) (c:SharedPtr<'T>) (d:SharedPtr<'T>) (x:SharedPtr<'T>) 
                                                                   (aTemp:SharedPtr<'T>) (bTemp:SharedPtr<'T>) (cTemp:SharedPtr<'T>) (dTemp:SharedPtr<'T>) = 
    let mutable k1 = 0G
    let mutable k2 = 0G   
    let mutable s = 1

    while s < size do
        let boundaryMin = tid-s
        let boundaryMax = tid+s

        if boundaryMin < 0 then
            k2 <- c.[tid]/b.[tid+s]
            aTemp.[tid] <- 0G
            bTemp.[tid] <- b.[tid] - a.[tid+s]*k2
            cTemp.[tid] <- -c.[tid+s]*k2
            dTemp.[tid] <- d.[tid] - d.[tid+s]*k2

        if boundaryMax >= size then
            k1 <- a.[tid]/b.[tid-s]
            aTemp.[tid] <- -a.[tid-s]*k1
            bTemp.[tid] <- b.[tid] - c.[tid-s]*k1
            cTemp.[tid] <- 0G
            dTemp.[tid] <- d.[tid] - d.[tid-s]*k1
     
        if boundaryMax < size && boundaryMin >= 0 then
            k1 <- a.[tid]/b.[tid-s]
            k2 <- c.[tid]/b.[tid+s]  
            aTemp.[tid] <- -a.[tid-s]*k1
            bTemp.[tid] <- b.[tid] - c.[tid-s]*k1 - a.[tid+s]*k2
            cTemp.[tid] <- -c.[tid+s]*k2
            dTemp.[tid] <- d.[tid] - d.[tid-s]*k1 - d.[tid+s]*k2
   
        __syncthreads()

        a.[tid] <- aTemp.[tid]
        b.[tid] <- bTemp.[tid]
        c.[tid] <- cTemp.[tid]
        d.[tid] <- dTemp.[tid]

        s <- s*2

    x.[tid] <- d.[tid]/b.[tid] 


/// Tridiagonal solver with temporary memory in registers
let inline triDiagTempReg () = cuda {

    let! kernel =     
        <@ fun n (dl:DevicePtr<'T>) (dd:DevicePtr<'T>) (du:DevicePtr<'T>) (dh:DevicePtr<'T>) ->  
            let tid = threadIdx.x
            
            let shared = __extern_shared__<'T>()
            let l = shared
            let d = l + n
            let u = d + n
            let h = u + n
        
            l.[tid] <- dl.[tid]
            d.[tid] <- dd.[tid]
            u.[tid] <- du.[tid]
            h.[tid] <- dh.[tid]
        
            __syncthreads()       
                
            triDiagPcrSingleBlock n l d u h 
            
            __syncthreads() 
        
            dh.[tid] <- h.[tid] @> |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        fun (l:DArray<'T>) (d:DArray<'T>) (u:DArray<'T>) (h:DArray<'T>) ->
            pcalc {
                do! PCalc.action (fun hint ->
                    let n = d.Length
                    let maxThreads = m.Worker.Device.Attribute DeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
                    if n > maxThreads then failwithf "cannot support systems of dimension larger than %d" maxThreads
                    let lp = LaunchParam(1, n, 4*n*sizeof<'T>) |> hint.ModifyLaunchParam
                    kernel.Launch m lp n l.Ptr d.Ptr u.Ptr h.Ptr
                )
                
                return h } ) }

/// Tridiagonal system solver with temporary memory in shared memory.
/// The additional shared memory requirement limits the dimension of the system.
let inline triDiagTempShared () = cuda {

    let! kernel =     
        <@ fun n (dl:DevicePtr<'T>) (dd:DevicePtr<'T>) (du:DevicePtr<'T>) (dh:DevicePtr<'T>) ->  
            let tid = threadIdx.x
            
            let shared = __extern_shared__<'T>()
            let l = shared
            let d = l + n
            let u = d + n
            let h = u + n
            let x = h + n
            let aTemp = x + n
            let bTemp = aTemp + n
            let cTemp = bTemp + n
            let dTemp = cTemp + n

            l.[tid] <- dl.[tid]
            d.[tid] <- dd.[tid]
            u.[tid] <- du.[tid]
            h.[tid] <- dh.[tid]
           
            triDiagPcrSingleBlockTempShared n tid l d u h x aTemp bTemp cTemp dTemp 
                                                                          
            dh.[tid] <- x.[tid] @> |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        fun (l:DArray<'T>) (d:DArray<'T>) (u:DArray<'T>) (h:DArray<'T>) ->
            pcalc {
                do! PCalc.action (fun hint ->
                    let n = d.Length
                    let maxThreads = m.Worker.Device.Attribute DeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
                    if n > maxThreads then failwithf "cannot support systems of dimension larger than %d" maxThreads
                    let lp = LaunchParam(1, n, 9*n*sizeof<'T>) |> hint.ModifyLaunchParam
                    kernel.Launch m lp n l.Ptr d.Ptr u.Ptr h.Ptr
                )
                
                return h } ) }


