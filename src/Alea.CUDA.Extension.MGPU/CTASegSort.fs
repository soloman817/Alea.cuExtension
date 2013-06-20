module Alea.CUDA.Extension.MGPU.CTASegSort

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore

//type IsegSortOp<'TI, 'TV, 'TR> =
//    abstract Commutative : int
//    abstract Identity : 'TI // the init value
//    abstract HExtract : ('TI -> int -> 'TV)
//    abstract DExtract : Expr<'TI -> int -> 'TV>
//    abstract HPlus : ('TV -> 'TV -> 'TV)
//    abstract DPlus : Expr<'TV -> 'TV -> 'TV>
//    abstract HCombine : ('TI -> 'TV -> 'TR)
//    abstract DCombine : Expr<'TI -> 'TV -> 'TR>
//
//type segSortOpType =
//    | segSortOpTypeAdd
//    | segSortOpTypeMul
//    | segSortOpTypeMin
//    | segSortOpTypeMax
//
//let inline segSortOp (opType:segSortOpType) (ident:'T) =
//    { new IsegSortOp<'T, 'T, 'T> with
//        member this.Commutative = 1
//        member this.Identity = ident
//        member this.HExtract = fun t index -> t
//        member this.DExtract = <@ fun t index -> t @>
//        member this.HPlus =
//            match opType with
//            | segSortOpTypeAdd -> ( + )
//            | segSortOpTypeMul -> ( * )
//            | segSortOpTypeMin -> min
//            | segSortOpTypeMax -> max
//        member this.DPlus =
//            match opType with
//            | segSortOpTypeAdd -> <@ ( + ) @>
//            | segSortOpTypeMul -> <@ ( * ) @>
//            | segSortOpTypeMin -> <@ min @>
//            | segSortOpTypeMax -> <@ max @>
//        member this.HCombine = fun t1 t2 -> t2 
//        member this.DCombine = <@ fun t1 t2 -> t2 @> }
//
//// CTAsegSort
//let ctasegSort (NT:int) (op:IsegSortOp<'TI, 'TV, 'TR>) =
//    let size = NT
//    let capacity = NT + NT / WARP_SIZE
//    let _, sLogPow2OfNT = sLogPow2 NT 1
//    let plus = op.DPlus
//
//    let segSort =
//        <@ fun (tid:int) (x:'TV) (storage:RWPtr<'TV>) ->
//            let plus = %plus
//
//            let mutable x = x
//            let dest = brev(uint32(tid)) >>> (32 - sLogPow2OfNT)
//            let dest = int(dest)
//            storage.[dest + dest / WARP_SIZE] <- x
//            __syncthreads()
//
//            let src = tid + tid / WARP_SIZE
//            let mutable destCount = NT / 2
//            while destCount >= 1 do
//                if tid < destCount then
//                    if (NT / 2 = destCount) then x <- storage.[src]
//                    let src2 = destCount + tid
//                    x <- plus x storage.[src2 + src2 / WARP_SIZE]
//                    storage.[src] <- x
//                __syncthreads()
//                destCount <- destCount / 2
//        
//            let total = storage.[0]
//            __syncthreads()
//            total @>
//
//    capacity, segSort
//
//// CTAsegSort
//let ctasegSort (NT:int) (op:IsegSortOp<'TI, 'TV, 'TR>) =
//    let size = NT
//    let capacity = 2 * NT + 1
//    let plus = op.DPlus
//    let extract = op.DExtract
//    let identity = op.Identity
//
//    let segSort =
//        <@ fun (tid:int) (x:'TV) (storage:RWPtr<'TV>) (total:RWPtr<'TV>) (stype:int) ->
//            let plus = %plus
//            let extract = %extract
//
//            let mutable x = x
//            storage.[tid] <- x
//            let mutable first = 0
//            __syncthreads()
//
//            let mutable offset = 1
//            while offset < NT do 
//                if(tid >= offset) then
//                    x <- plus storage.[first + tid - offset] x
//                first <- NT - first
//                storage.[first + tid] <- x
//                offset <- offset + offset
//                __syncthreads()
//            
//            total.[0] <- storage.[first + NT - 1]
//            if(stype = ExclusivesegSort) then
//                x <- if( tid <> 0 ) then storage.[first + tid - 1] else extract identity -1                        
//            __syncthreads()
//            x @>
//    capacity, segSort
//
//// CTAsegSort2
//let ctasegSort2 (NT:int) (op:IsegSortOp<'TI, 'TV, 'TR>) =
//    let capacity, segSort = ctasegSort NT op
//    let segSort2 =
//        <@ fun (tid:int) (x:'TV) (storage:RWPtr<'TV>) ->
//            let segSort = %segSort
//            let total = __local__<'TV>(1).Ptr(0) 
//            segSort tid x storage total ExclusivesegSort @>
//
//    capacity, segSort2