﻿[<AutoOpen>]
module Alea.cuExtension.MGPU.CTAMerge

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.cuExtension
open Alea.cuExtension.MGPU
open Alea.cuExtension.MGPU.Static
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.Intrinsics
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTASearch
open Alea.cuExtension.MGPU.SortedNetwork
//
//
//type IFindMergesortFrame =
//    abstract Host : (int -> int -> int -> int3)
//    abstract Device : Expr<int -> int -> int -> int3>
//
//type IFindMergesortInterval =
//    abstract Host : (int3 -> int -> int -> int -> int -> int -> int -> int4)
//    abstract Device : Expr<int3 -> int -> int -> int -> int -> int -> int -> int4>
//
//type IComputeMergeRange =
//    abstract Host : (int -> int -> int -> int -> int -> int[] -> int4)
//    abstract Device : Expr<int -> int -> int -> int -> int -> deviceptr<int> -> int4>
//
//// SerialMerge
//let serialMerge (VT:int) (rangeCheck:bool) (comp:IComp<'TV>) =
//    let comp = comp.Device
//    <@ fun  (keys_shared    :deviceptr<'TV>) 
//            (aBegin         :int) 
//            (aEnd           :int) 
//            (bBegin         :int) 
//            (bEnd           :int) 
//            (results        :deviceptr<'TV>) 
//            (indices        :deviceptr<int>) 
//            ->
//        
//        let comp = %comp
//
//        let mutable aKey = keys_shared.[aBegin]
//        let mutable bKey = keys_shared.[bBegin]
//        let mutable aBegin = aBegin
//        let mutable bBegin = bBegin
//
//        for i = 0 to VT - 1 do
//            let mutable p = false
//            if rangeCheck then
//                p <- (bBegin >= bEnd) || ((aBegin < aEnd) && not(comp bKey aKey))
//            else
//                p <- not (comp bKey aKey)
//
//            results.[i] <- if p then aKey else bKey
//            indices.[i] <- if p then aBegin else bBegin
//
//            if p then
//                aBegin <- aBegin + 1
//                aKey <- keys_shared.[aBegin]
//                
//            else
//                bBegin <- bBegin + 1
//                bKey <- keys_shared.[bBegin]
//                
//        __syncthreads() @>
//
//////////////////////////////////////////////////////////////////////////////////
//// FindMergeFrame and FindMergesortInterval help mergesort (both CTA and global 
//// merge pass levels) locate lists within the single source array.
//
//// Returns (offset of a, offset of b, length of list).
//let findMergesortFrame = // works
//    { new IFindMergesortFrame with    
//        member fmf.Host =
//            fun (coop:int) (block:int) (nv:int) ->
//                let start = ~~~(coop - 1) &&& block
//                let size = nv * (coop >>> 1)
//                int3((nv * start), (nv * start + size), size)
//         
//        member fmf.Device =
//            <@ fun (coop:int) (block:int) (nv:int) ->
//                let start = ~~~(coop - 1) &&& block
//                let size = nv * (coop >>> 1)
//                int3((nv * start), (nv * start + size), (size)) @> }
//
//// Returns (a0, a1, b0, b1) into mergesort input lists between mp0 and mp1.
//let findMergesortInterval =
//    { new IFindMergesortInterval with
//        member fmi.Host =
//            fun (frame:int3) (coop:int) (block:int) (nv:int) (count:int) (mp0:int) (mp1:int) ->
//                let diag = nv * block - frame.x
//                let a0 = frame.x + mp0
//                let mutable a1 = min count (frame.x + mp1)
//                let b0 = min count (frame.y + diag - mp0)
//                let mutable b1 = min count (frame.y + diag + nv - mp1)
//                if (coop - 1) = ((coop - 1) &&& block) then
//                    a1 <- min count (frame.x + frame.z)
//                    b1 <- min count (frame.y + frame.z)
//                int4(a0,a1,b0,b1)
//
//        // The end partition of the last block for each merge operation is computed
//        // and stored as the begin partition for the subsequent merge. i.e. it is
//        // the same partition but in the wrong coordinate system, so its 0 when it
//        // should be listSize. Correct that by checking if this is the last block
//        // in this merge operation.
//        member fmi.Device =
//            <@ fun (frame:int3) (coop:int) (block:int) (nv:int) (count:int) (mp0:int) (mp1:int) ->
//                let diag = nv * block - frame.x
//                let a0 = frame.x + mp0
//                let mutable a1 = min count (frame.x + mp1)
//                let b0 = min count (frame.y + diag - mp0)
//                let mutable b1 = min count (frame.y + diag + nv - mp1)
//                if (coop - 1) = ((coop - 1) &&& block) then
//                    a1 <- min count (frame.x + frame.z)
//                    b1 <- min count (frame.y + frame.z)
//                int4(a0,a1,b0,b1) @> }
//
//
//////////////////////////////////////////////////////////////////////////////////
//// ComputeMergeRange
//let computeMergeRange = //works
//    { new IComputeMergeRange with
//        member cmr.Host =
//            fun (aCount:int) (bCount:int) (block:int) (coop:int) (nv:int) (mp_global:int[]) ->
//                let mp0 = mp_global.[block]
//                let mp1 = mp_global.[block + 1]
//                let gid = nv * block
//
//                let mutable range = int4(0,0,0,0)
//                if coop <> 0 then
//                    let frame = findMergesortFrame.Host coop block nv
//                    range <- findMergesortInterval.Host frame coop block nv aCount mp0 mp1
//                else
//                    range.x <- mp0
//                    range.y <- mp1
//                    range.z <- gid - range.x
//                    range.w <- (min (aCount + bCount) (gid + nv)) - range.y
//                range
//
//        member cmr.Device =
//            let findMergesortFrame = findMergesortFrame.Device
//            let findMergesortInterval = findMergesortInterval.Device
//            <@ fun (aCount:int) (bCount:int) (block:int) (coop:int) (nv:int) (mp_global:deviceptr<int>) ->
//                let findMergesortFrame = %findMergesortFrame
//                let findMergesortInterval = %findMergesortInterval
//
//                // Load the merge paths computed by the partitioning kernel.
//                let mp0 = mp_global.[block]
//                let mp1 = mp_global.[block + 1]
//                let gid = nv * block
//
//                // Compute the ranges of the sources in global memory.
//                let mutable range = int4(0,0,0,0)
//                if coop <> 0 then
//                    let frame = findMergesortFrame coop block nv
//                    range <- findMergesortInterval frame coop block nv aCount mp0 mp1
//                else                    
//                    range.x <- mp0
//                    range.y <- mp1
//                    range.z <- gid - range.x
//                    range.w <- (min (aCount + bCount) (gid + nv)) - range.y
//                range @> }
//
//
//
////// CTA mergesort support
//let ctaBlocksortPass (NT:int) (VT:int) (compOp:IComp<'TV>) =
//    let mergePath = (mergePath MgpuBoundsLower compOp).DMergePath
//    let serialMerge = serialMerge VT true compOp
//    let comp = compOp.Device
//    <@ fun  (keys_shared    :deviceptr<'TV>) 
//            (tid            :int) 
//            (count          :int) 
//            (coop           :int) 
//            (keys           :deviceptr<'TV>) 
//            (indices        :deviceptr<int>) 
//            ->
//
//        let mergePath = %mergePath
//        let serialMerge = %serialMerge
//
//        let list = ~~~(coop - 1) &&& tid
//        let diag = min count (VT * ((coop - 1) &&& tid))
//        let start = VT * list
//        let a0 = min count start
//        let b0 = min count (start + VT * (coop / 2))
//        let b1 = min count (start + VT * coop)
//                
//        let p = mergePath (keys_shared + a0) (b0 - a0) (keys_shared + b0) (b1 - b0) diag
//        serialMerge keys_shared (a0 + p) b0 (b0 + diag - p) b1 keys indices 
//    @>
//
//
//let ctaBlocksortLoop (NT:int) (VT:int) (hasValues:bool) (compOp:IComp<'TV>) =
//    let ctaBlocksortPass = ctaBlocksortPass NT VT compOp
//    let deviceThreadToShared = deviceThreadToShared VT
//    let deviceGather = deviceGather NT VT
//
//    <@ fun  (threadValues   :deviceptr<'TV>) 
//            (keys_shared    :deviceptr<'TV>) 
//            (values_shared  :deviceptr<'TV>) 
//            (tid            :int) 
//            (count          :int) 
//            ->
//
//        let ctaBlocksortPass = %ctaBlocksortPass
//        let deviceThreadToShared = %deviceThreadToShared
//        let deviceGather = %deviceGather
//
//        let mutable coop = 2
//        while coop <= NT do
//            
//            let indices = __local__.Array<int>(VT) |> __array_to_ptr
//            let keys = __local__.Array<'TV>(VT) |> __array_to_ptr
//            ctaBlocksortPass keys_shared tid count coop keys indices
//            
//            if hasValues then
//                deviceThreadToShared threadValues tid values_shared true
//                deviceGather (NT * VT) values_shared indices tid threadValues true
//
//            deviceThreadToShared keys tid keys_shared true
//            coop <- coop * 2
//    @>
//
//
//////////////////////////////////////////////////////////////////////////////////
//// CTAMergesort
//// Caller provides the keys in shared memory. This functions sorts the first
//// count elements.
//let ctaMergesort (NT:int) (VT:int) (hasValues:bool) (compOp:IComp<'TV>) =
//    
//    let deviceThreadToShared = deviceThreadToShared VT
//    let ctaBlocksortLoop = ctaBlocksortLoop NT VT hasValues compOp
//    let oddEvenTransposeSort = oddEvenTransposeSort VT compOp
//    let comp = compOp.Device
//    <@ fun (threadKeys      :deviceptr<'TV>) 
//           (threadValues    :deviceptr<'TV>) 
//           (keys_shared     :deviceptr<'TV>) 
//           (values_shared   :deviceptr<'TV>) 
//           (count           :int) 
//           (tid             :int) 
//           ->
//
//        let comp = %comp
//        let deviceThreadToShared = %deviceThreadToShared
//        let ctaBlocksortLoop = %ctaBlocksortLoop
//        let oddEvenTransposeSort = %oddEvenTransposeSort
//        if VT * tid < count then
//            //oddEvenTransposeSort threadKeys threadValues            
//            let mutable level = 0
//            while level < VT do
//                let mutable i = (1 &&& level)
//                while i < VT - 1 do                    
//                    if ( comp threadKeys.[i + 1] threadKeys.[i] ) then
//                        let a = threadKeys.[i] 
//                        let b = threadKeys.[i + 1]
//                        threadKeys.[i] <- b
//                        threadKeys.[i + 1] <- a
//                        let a = threadValues.[i]
//                        let b = threadValues.[i + 1]
//                        threadValues.[i] <- b
//                        threadValues.[i + 1] <- a
//                    i <- i + 2  
//                level <- level + 1 
//            
//        deviceThreadToShared threadKeys tid keys_shared true
//        ctaBlocksortLoop threadValues keys_shared values_shared tid count 
//        @>
//
//// works
//let deviceMergeKeysIndices (NT:int) (VT:int) (compOp:IComp<'TV>) =
//    let deviceLoad2ToShared = deviceLoad2ToSharedB NT VT VT
//    let mergePath = (mergePath MgpuBoundsLower compOp).DMergePath
//    let serialMerge = serialMerge VT true compOp
//
//    <@ fun  (a_global   :deviceptr<'TV>) 
//            (b_global   :deviceptr<'TV>) 
//            (range      :int4)
//            (tid        :int) 
//            (keys_shared:deviceptr<'TV>) 
//            (results    :deviceptr<'TV>) 
//            (indices    :deviceptr<int>) 
//            ->
//
//        let deviceLoad2ToShared = %deviceLoad2ToShared
//        let mergePath = %mergePath
//        let serialMerge = %serialMerge
//
//        let a0 = range.x
//        let a1 = range.y
//        let b0 = range.z
//        let b1 = range.w
//        let aCount = a1 - a0
//        let bCount = b1 - b0
//
//        deviceLoad2ToShared (a_global + a0) aCount (b_global + b0) bCount tid keys_shared true
//
//        let diag = VT * tid        
//        let mp = mergePath keys_shared aCount (keys_shared + aCount) bCount diag
//
//        let a0tid = mp
//        let a1tid = aCount
//        let b0tid = aCount + diag - mp
//        let b1tid = aCount + bCount
//
//        serialMerge keys_shared a0tid a1tid b0tid b1tid results indices
//        @>
//
//// works
//let deviceMerge (NT:int) (VT:int) (hasValues:bool) (compOp:IComp<'TV>) =
//    let deviceMergeKeysIndices = deviceMergeKeysIndices NT VT compOp
//    let deviceThreadToSharedG = deviceThreadToShared VT
//    let deviceThreadToSharedI = deviceThreadToShared VT
//    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
//    let deviceTransferMergeValues = deviceTransferMergeValuesA NT VT
//
//    <@ fun  (aKeys_global   :deviceptr<'TV>) 
//            (aVals_global   :deviceptr<'TV>) 
//            (bKeys_global   :deviceptr<'TV>) 
//            (bVals_global   :deviceptr<'TV>) 
//            (tid            :int) 
//            (block          :int) 
//            (range          :int4) 
//            (keys_shared    :deviceptr<'TV>) 
//            (indices_shared :deviceptr<int>) 
//            (keys_global    :deviceptr<'TV>) 
//            (vals_global    :deviceptr<'TV>) 
//            ->
//
//        let deviceMergeKeysIndices = %deviceMergeKeysIndices
//        let deviceThreadToSharedG = %deviceThreadToSharedG
//        let deviceThreadToSharedI = %deviceThreadToSharedI
//        let deviceSharedToGlobal = %deviceSharedToGlobal
//        let deviceTransferMergeValues = %deviceTransferMergeValues
//
//        let results = __local__.Array<'TV>(VT) |> __array_to_ptr
//        let indices = __local__.Array<int>(VT) |> __array_to_ptr
//                
//        deviceMergeKeysIndices 
//            aKeys_global 
//            bKeys_global 
//            range 
//            tid 
//            keys_shared 
//            results 
//            indices
//        
//        deviceThreadToSharedG results tid keys_shared true
//
//        let aCount = range.y - range.x
//        let bCount = range.w - range.z
//        deviceSharedToGlobal (aCount + bCount) keys_shared tid (keys_global + NT * VT * block) true
//
//        if hasValues then
//            deviceThreadToSharedI indices tid indices_shared true
//            
//            deviceTransferMergeValues 
//                (aCount + bCount) 
//                (aVals_global + range.x) 
//                (bVals_global + range.z) 
//                aCount 
//                indices_shared 
//                tid 
//                (vals_global + NT * VT * block) 
//                true
//        @>