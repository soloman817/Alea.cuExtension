﻿module Alea.CUDA.Extension.MGPU.CTAScan

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Static
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.Intrinsics

type IScanOp<'TI, 'TV, 'TR> =
    abstract Commutative : bool
    abstract Identity : 'TI // the init value
    abstract HExtract : ('TI -> int -> 'TV)
    abstract DExtract : Expr<'TI -> int -> 'TV>
    abstract HPlus : ('TV -> 'TV -> 'TV)
    abstract DPlus : Expr<'TV -> 'TV -> 'TV>
    abstract HCombine : ('TI -> 'TV -> 'TR)
    abstract DCombine : Expr<'TI -> 'TV -> 'TR>

type ScanOpType =
    | ScanOpTypeAdd
    | ScanOpTypeMul
    | ScanOpTypeMin
    | ScanOpTypeMax

let inline scanOp (opType:ScanOpType) (ident:'T) =
    { new IScanOp<'T, 'T, 'T> with
        member this.Commutative = true
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
    let _, sLogPow2OfNT = sLogPow2 NT true
    let plus = op.DPlus

    let reduce =
        <@ fun (tid:int) (x:'TV) (storage:RWPtr<'TV>) ->
            let plus = %plus

            let mutable x = x
            let dest = brev(uint32(tid)) >>> (32 - sLogPow2OfNT)
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