[<AutoOpen>]
module Alea.cuExtension.MGPU.CTAScan

// this maps to ctascan.cuh. 

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.cuExtension
open Alea.cuExtension.MGPU
open Alea.cuExtension.MGPU.Static
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.Intrinsics


// in c++, mgpu uses template to define the interface, but F# doesn't
// have template, so we have to use a template. Please read carefully
// on the ctascan.cuh, and see how I mapped them into interface.
type IScanOp<'TI, 'TV, 'TR> =
    abstract Commutative : int
    abstract Identity : 'TI // the init value
    abstract HExtract : ('TI -> int -> 'TV)
    abstract DExtract : Expr<'TI -> int -> 'TV>
    abstract HPlus : ('TV -> 'TV -> 'TV)
    abstract DPlus : Expr<'TV -> 'TV -> 'TV>
    abstract HCombine : ('TI -> 'TV -> 'TR)
    abstract DCombine : Expr<'TI -> 'TV -> 'TR>


let inline scanOp (opType:ScanOpType) (ident:'T) =
    { new IScanOp<'T, 'T, 'T> with
        member this.Commutative = 1
        member this.Identity = ident
        member this.HExtract = fun t index -> t
        member this.DExtract = <@ fun t index -> t @>
        member this.HPlus =
            match opType with
            | ScanOpTypeAdd -> ( + )
            | ScanOpTypeMul -> ( * )
            | ScanOpTypeMin -> min
            | ScanOpTypeMax -> max
        member this.DPlus =
            match opType with
            | ScanOpTypeAdd -> <@ ( + ) @>
            | ScanOpTypeMul -> <@ ( * ) @>
            | ScanOpTypeMin -> <@ min @>
            | ScanOpTypeMax -> <@ max @>
        member this.HCombine = fun t1 t2 -> t2 
        member this.DCombine = <@ fun t1 t2 -> t2 @> }

// CTAReduce
let ctaReduce (NT:int) (op:IScanOp<'TI, 'TV, 'TR>) =
    let size = NT
    let capacity = NT + NT / WARP_SIZE
    let _, sLogPow2OfNT = sLogPow2 NT 1
    let plus = op.DPlus

    let reduce =
        <@ fun (tid:int) (x:'TV) (storage:deviceptr<'TV>) ->
            let plus = %plus

            let mutable x = x
            let dest = LibDevice.__nv_brev(uint32(tid)) >>> (32 - sLogPow2OfNT)
            let dest = int(dest)
            storage.[dest + dest / WARP_SIZE] <- x
            __syncthreads()

            let src = tid + tid / WARP_SIZE
            let mutable destCount = NT / 2
            while destCount >= 1 do
                if tid < destCount then
                    if (NT / 2 = destCount) then x <- storage.[src]
                    let src2 = destCount + tid
                    x <- plus x storage.[src2 + src2 / WARP_SIZE]
                    storage.[src] <- x
                __syncthreads()
                destCount <- destCount / 2
        
            let total = storage.[0]
            __syncthreads()
            total @>

    capacity, reduce

//let ctaReduce (NT:int) (op:IScanOp<'TI, 'TV, 'TR>) =
    

// CTAScan
let ctaScan (NT:int) (op:IScanOp<'TI, 'TV, 'TR>) =
    let size = NT
    let capacity = 2 * NT + 1
    let plus = op.DPlus
    let extract = op.DExtract
    let identity = op.Identity

    let scan =
        <@ fun (tid:int) (x:'TV) (storage:deviceptr<'TV>) (total:deviceptr<'TV>) (stype:int) ->
            let plus = %plus
            let extract = %extract

            let mutable x = x
            storage.[tid] <- x
            let mutable first = 0
            __syncthreads()

            let mutable offset = 1
            while offset < NT do 
                if(tid >= offset) then
                    x <- plus storage.[first + tid - offset] x
                first <- NT - first
                storage.[first + tid] <- x
                offset <- offset + offset
                __syncthreads()
            
            total.[0] <- storage.[first + NT - 1]
            if(stype = ExclusiveScan) then
                x <- if( tid <> 0 ) then storage.[first + tid - 1] else extract identity -1                        
            __syncthreads()
            x @>
    capacity, scan

// CTAScan2
// possible to refactor?
let ctaScan2 (NT:int) (op:IScanOp<'TI, 'TV, 'TR>) =
    let capacity, scan = ctaScan NT op
    let scan2 =
        <@ fun (tid:int) (x:'TV) (storage:deviceptr<'TV>) ->
            let scan = %scan
            let total = __local__.Array<'TV>(1) |> __array_to_ptr 
            scan tid x storage total ExclusiveScan @>

    capacity, scan2

