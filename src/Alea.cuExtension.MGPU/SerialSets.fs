module Alea.cuExtension.MGPU.SerialSets
//// NOT IMPLEMENTED YET
//open System.Runtime.InteropServices
//open Microsoft.FSharp.Collections
//open Alea.CUDA
//open Alea.cuExtension
////open Alea.cuExtension.Util
//open Alea.cuExtension.MGPU
////open Alea.cuExtension.MGPU.QuotationUtil
//open Alea.cuExtension.MGPU.DeviceUtil
//open Alea.cuExtension.MGPU.LoadStore
//open Alea.cuExtension.MGPU.CTAScan
//
//
//
//
//
////////////////////////////////////////////////////////////////////////////////////
////// SerialSetIntersection
////// Emit A if A and B are in range and equal.
////
//let serialSetIntersection (VT:int) (rangeCheck:bool) (compOp:IComp<int>) =
//    let comp = compOp.Device
//    <@ fun  (data       :deviceptr<int>) 
//            (aBegin     :int) 
//            (aEnd       :int) 
//            (bBegin     :int) 
//            (bEnd       :int) 
//            (end'       :int) 
//            (results    :deviceptr<int>) 
//            (indices    :deviceptr<int>) 
//            ->
//            let comp = %comp
//
//            let mutable aBegin = aBegin
//            let mutable bBegin = bBegin
//
//            let minIterations = VT / 2
//            let mutable commit = 0
//
//            for i = 0 to VT - 1 do
//                let mutable test = if rangeCheck then ((aBegin + bBegin) < end') && ((aBegin < aEnd) && (bBegin < bEnd)) else ((i < minIterations) || ((aBegin + bBegin) < end'))
//
//                if test then
//                    let aKey = data.[aBegin]
//                    let bKey = data.[bBegin]
//
//                    let pA = comp aKey bKey
//                    let pB = comp bKey aKey
//
//                    results.[i] <- aKey
//                    indices.[i] <- aBegin
//
//                    if not pB then aBegin <- aBegin + 1
//                    if not pA then bBegin <- bBegin + 1
//
//                    if pA = pB then commit <- commit ||| 1 <<< i
//            commit
//    @>
//
//
////////////////////////////////////////////////////////////////////////////////////
////// SerialSetUnion
////// Emit A if A <= B. Emit B if B < A.
////
////template<int VT, bool RangeCheck, typename T, typename Comp>
////MGPU_DEVICE int SerialSetUnion(const T* data, int aBegin, int aEnd,
////	int bBegin, int bEnd, int end, T* results, int* indices, Comp comp) {
//let serialSetUnion (VT:int) (rangeCheck:bool) (compOp:IComp<int>) =
//    let comp = compOp.Device
//    <@ fun  (data       :deviceptr<int>) 
//            (aBegin     :int) 
//            (aEnd       :int) 
//            (bBegin     :int) 
//            (bEnd       :int) 
//            (end'       :int) 
//            (results    :deviceptr<int>) 
//            (indices    :deviceptr<int>) 
//            ->
//        let comp = %comp
//
//        let mutable aBegin = aBegin
//        let mutable bBegin = bBegin
//
//        let minIterations = VT / 2
//        let mutable commit = 0
//
//        for i = 0 to VT - 1 do
//            let mutable test = if rangeCheck then ((aBegin + bBegin) < end') else ((i < minIterations) || ((aBegin + bBegin) < end'))
//
//            if test then
//                let aKey = data.[aBegin]
//                let bKey = data.[bBegin]
//
//                let mutable pA = false
//                let mutable pB = false
//
//                if rangeCheck && (aBegin >= aEnd) then 
//                    pB <- true
//                elif rangeCheck && (bBegin >= bEnd) then
//                    pA <- true
//                else
//                    pA <- comp aKey bKey
//                    pB <- comp bKey aKey
//
//                results.[i] <- if pB then bKey else aKey
//                indices.[i] <- if pB then bBegin else aBegin
//
//                if not pB then aBegin <- aBegin + 1
//                if not pA then bBegin <- bBegin + 1
//
//                commit <- commit ||| 1 <<< i
//        commit
//    @>
//
//
////////////////////////////////////////////////////////////////////////////////////
////// SerialSetDifference
////// Emit A if A < B.
//// 
//let serialSetDifference (VT:int) (rangeCheck:bool) (compOp:IComp<int>) =
//    let comp = compOp.Device
//    <@ fun  (data       :deviceptr<int>) 
//            (aBegin     :int) 
//            (aEnd       :int) 
//            (bBegin     :int) 
//            (bEnd       :int) 
//            (end'       :int) 
//            (results    :deviceptr<int>) 
//            (indices    :deviceptr<int>) 
//            ->
//        let comp = %comp
//
//        let mutable aBegin = aBegin
//        let mutable bBegin = bBegin
//
//        let minIterations = VT / 2
//        let mutable commit = 0
//
//        for i = 0 to VT - 1 do
//            let mutable test = if rangeCheck then ((aBegin + bBegin) < end') else ((i < minIterations) || ((aBegin + bBegin) < end'))
//
//            if test then
//                let aKey = data.[aBegin]
//                let bKey = data.[bBegin]
//
//                let mutable pA = false
//                let mutable pB = false
//
//                if rangeCheck && (aBegin >= aEnd) then
//                    pB <- true
//                elif rangeCheck && (bBegin >= bEnd) then
//                    pA <- true
//                else
//                    pA <- comp aKey bKey
//                    pB <- comp bKey aKey
//
//                results.[i] <- aKey
//                indices.[i] <- aBegin
//                if not pB then aBegin <- aBegin + 1
//                if not pA then bBegin <- bBegin + 1
//                if pA then commit <- commit ||| (1 <<< i)
//
//        commit        
//        @>
//
//
////////////////////////////////////////////////////////////////////////////////////
////// SerialSetSymDiff
////// Emit A if A < B and emit B if B < A.
////
//let serialSetSymDiff (VT:int) (rangeCheck:bool) (compOp:IComp<int>) =
//    let comp = compOp.Device
//    <@ fun  (data       :deviceptr<int>) 
//            (aBegin     :int) 
//            (aEnd       :int) 
//            (bBegin     :int) 
//            (bEnd       :int) 
//            (end'       :int) 
//            (results    :deviceptr<int>) 
//            (indices    :deviceptr<int>) 
//            ->
//        let comp = %comp
//
//        let mutable aBegin = aBegin
//        let mutable bBegin = bBegin
//
//        let minIterations = VT / 2
//        let mutable commit = 0
//
//
//        for i = 0 to VT - 1 do
//            let test = if rangeCheck then ((aBegin + bBegin) < end') else ((i < minIterations) || ((aBegin + bBegin) < end'))
//
//            if test then
//                let aKey = data.[aBegin]
//                let bKey = data.[bBegin]
//
//                let mutable pA = false
//                let mutable pB = false
//
//                if (rangeCheck && (bBegin >= aEnd)) then
//                    pA <- true
//                elif (rangeCheck && (aBegin >= aEnd)) then
//                    pB <- true
//                else
//                    pA <- comp aKey bKey
//                    pB <- comp bKey aKey
//
//                results.[i] <- if pA then aKey else bKey
//                indices.[i] <- if pA then aBegin else bBegin
//
//                if not pA then bBegin <- bBegin + 1
//                if not pB then aBegin <- aBegin + 1
//                if (pA <> pB) then commit <- commit ||| 1 <<< i
//        commit
//        @>
//
//
//
////////////////////////////////////////////////////////////////////////////////////
////// SerialSetOp
////// Uses the MgpuSetOp enum to statically select one of the four serial ops
////// above.
////
//let serialSetOp (VT:int) (rangeCheck:bool) (setOp:MgpuSetOp) (compOp:IComp<int>) =
//    let comp = compOp.Device
//    let serialSetIntersection = serialSetIntersection VT rangeCheck compOp
//    let serialSetUnion = serialSetUnion VT rangeCheck compOp
//    let serialSetDifference = serialSetDifference VT rangeCheck compOp
//    let serialSetSymDiff = serialSetSymDiff VT rangeCheck compOp
//
//    <@ fun  (data:deviceptr<int>) 
//            (aBegin:int) 
//            (aEnd:int) 
//            (bBegin:int) 
//            (bEnd:int) 
//            (star:int) 
//            (results:deviceptr<int>) 
//            (indices:deviceptr<int>) 
//            ->
//        let comp = %comp
//        let serialSetIntersection = %serialSetIntersection
//        let serialSetUnion = %serialSetUnion
//        let serialSetDifference = %serialSetDifference
//        let serialSetSymDiff = %serialSetSymDiff
//
//        let mutable end' = aBegin + bBegin + VT - star
//        if rangeCheck then end' <- min end' (aEnd + bEnd)
//        let mutable commit = 0
//
//        match setOp with
//        | MgpuSetOpIntersection -> commit <- serialSetIntersection data aBegin aEnd bBegin bEnd end' results indices
//        | MgpuSetOpUnion -> commit <- serialSetUnion data aBegin aEnd bBegin bEnd end' results indices
//        | MgpuSetOpDiff -> commit <- serialSetDifference data aBegin aEnd bBegin bEnd end' results indices
//        | MgpuSetOpSymDiff -> commit <- serialSetSymDiff data aBegin aEnd bBegin bEnd end' results indices
//        __syncthreads()
//
//    @>