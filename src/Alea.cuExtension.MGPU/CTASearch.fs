[<AutoOpen>]
module Alea.cuExtension.MGPU.CTASearch

// this file maps to ctasearch.cuh

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.cuExtension
//open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTAScan
//
//
//// @COMMENTS@ : I checked the C++ code, the type IT means iterator type, and here in our case,
//// iterator could be either 'TC[] or deviceptr<'TC> for host and device. So it is not neccessary
//// to have two types.
//type IBinarySearchIt<'TC> =
//    abstract HBinarySearchIt : ('TC[] -> int ref -> int ref -> 'TC -> int -> unit)
//    abstract DBinarySearchIt : Expr<deviceptr<'TC> -> int ref -> int ref -> 'TC -> int -> unit>    
//    
//type IBiasedBinarySearch<'TC> =
//    abstract HBiasedBinarySearch : ('TC[] -> int -> 'TC -> int -> int)
//    abstract DBiasedBinarySearch : Expr<deviceptr<'TC> -> int -> 'TC -> int -> int>
//    
//type IBinarySearch<'TC> =
//    abstract HBinarySearch : ('TC[] -> int -> 'TC -> int)
//    abstract DBinarySearch : Expr<deviceptr<'TC> -> int -> 'TC -> int>
//    
//type IMergePath<'TC> =
//    abstract HMergePath : ('TC[] -> int -> 'TC[] -> int -> int -> int)
//    abstract DMergePath : Expr<deviceptr<'TC> -> int -> deviceptr<'TC> -> int -> int -> int>
//    abstract DMergePathInt : Expr<deviceptr<int> -> int -> deviceptr<int> -> int -> int -> int>
//
//type ISegmentedMergePath<'TC> =
//    abstract HSegmentedMergePath : ('TC[] -> int -> int -> int -> int -> int -> int -> int -> int)
//    abstract DSegmentedMergePath : Expr<deviceptr<'TC> -> int -> int -> int -> int -> int -> int -> int -> int>
//
//type IBalancedPathSearch<'TC> =
//    abstract HBalancedPath : ('TC[] -> int -> 'TC[] -> int -> int -> int -> int2)
//    abstract DBalancedPath : Expr<deviceptr<'TC> -> int -> deviceptr<'TC> -> int -> int -> int -> int2>
//
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////                                                                                                      //
////      Binary Search It                                                                                //
////                                                                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//let binarySearchIt (bounds:int) (compOp:IComp<'TC>) =
//    { new IBinarySearchIt<'TC> with
//        member this.HBinarySearchIt =
//            let comp a b = compOp.Host a b
//            fun (data:intC[]) (begin':int ref) (end':int ref) (key:intC) (shift:int) ->
//                let scale = (1 <<< shift) - 1
//                let mid = (!begin' + scale * !end') >>> shift
//                let key2 = data.[mid]
//                let pred = 
//                    match bounds = MgpuBoundsUpper with
//                    | true -> not (comp key key2)
//                    | false -> comp key2 key
//
//                if pred then begin' := mid + 1
//                else end' := mid                    
//                        
//        member this.DBinarySearchIt = 
//            let comp = compOp.Device
//            <@ fun (data:deviceptr<'TC>) (begin':int ref) (end':int ref) (key:intC) (shift:int) ->
//                let comp = %comp
//                
////                match typeof<'TInt> with
////                    | x when x = typeof<int>
////                    | x when x = typeof<int64>
////                    | _ -> ()
//                
//                let scale = (1 <<< shift) - 1
//                let mid = ((!begin' + scale * !end') >>> shift)
//                let key2 = data.[mid]
//                let pred =
//                    match bounds = MgpuBoundsUpper with
//                    | true -> not (comp key key2)
//                    | false -> comp key2 key
//
//                if pred then begin' := mid + 1
//                else end' := mid @> }
//
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////                                                                                                      //
////      Binary Search                                                                                   //
////                                                                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//let binarySearch (bounds:int) (compOp:IComp<'TC>)  =
//    let binarySearchIt = binarySearchIt bounds compOp
//    { new IBinarySearch<'TC> with               
//       member this.HBinarySearch = 
//            let binarySearchIt = binarySearchIt.HBinarySearchIt
//            fun (data:intC[]) (count:int) (key:intC) ->
//                let begin' = ref 0
//                let end' = ref count
//                while begin' < end' do
//                    binarySearchIt data begin' end' key 1
//                !begin'
//
//        member this.DBinarySearch = 
//            let binarySearchIt = binarySearchIt.DBinarySearchIt
//            <@ fun (data:deviceptr<'TC>) (count:int) (key:intC) ->
//                let binarySearchIt = %binarySearchIt
//                    
//                let begin' = __local__.Variable(0)
//                let end' = __local__.Variable(0)                
//                    
//                while !begin' < !end' do
//                    binarySearchIt data begin' end' key 1
//                !begin' @> }
//
//
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////                                                                                                      //
////      Biased Binary Search                                                                            //
////                                                                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//let biasedBinarySearch (bounds:int) (compOp:IComp<'TC>) =
//    let binarySearchIt = binarySearchIt bounds compOp
//    { new IBiasedBinarySearch<'TC> with
//        member this.HBiasedBinarySearch =
//            let binarySearchIt = binarySearchIt.HBinarySearchIt
//            let comp a b = compOp.Host a b
//            fun (data:intC[]) (count:int) (key:intC) (levels:int) ->
//                let begin' = ref 0
//                let end' = ref count
//
//                if levels >= 4 && begin' < end' then binarySearchIt data begin' end' key 9
//                if levels >= 3 && begin' < end' then binarySearchIt data begin' end' key 7
//                if levels >= 2 && begin' < end' then binarySearchIt data begin' end' key 5
//                if levels >= 1 && begin' < end' then binarySearchIt data begin' end' key 4
//
//                while begin' < end' do
//                    binarySearchIt data begin' end' key 1
//                begin'.contents
//
//        member this.DBiasedBinarySearch = 
//            let comp = compOp.Device
//            let binarySearchIt = binarySearchIt.DBinarySearchIt
//            <@ fun (data:deviceptr<'TC>) (count:int) (key:intC) (levels:int) ->
//                let comp = %comp
//                let binarySearchIt = %binarySearchIt
//                                
//                let begin' = __local__.Variable<int>(0)
//                let end' = __local__.Variable(count)
//
//                if levels >= 4 && begin' < end' then binarySearchIt data begin' end' key 9
//                if levels >= 3 && begin' < end' then binarySearchIt data begin' end' key 7
//                if levels >= 2 && begin' < end' then binarySearchIt data begin' end' key 5
//                if levels >= 1 && begin' < end' then binarySearchIt data begin' end' key 4
//
//                while begin' < end' do
//                    binarySearchIt data begin' end' key 1
//                !begin' @> }
//
//
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////                                                                                                      //
////      Merge Path Search                                                                               //
////                                                                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//let mergePath (bounds:int) (compOp:IComp<'TC>) =
//    { new IMergePath<'TC> with
//        member this.HMergePath =
//                let comp a b = compOp.Host a b
//                fun (a:intC[]) (aCount:int) (b:intC[]) (bCount:int) (diag:int) ->
//                    let mutable begin' = max 0 (diag - bCount)
//                    let mutable end' = min diag aCount
//
//                    while begin' < end' do
//                        let mid = (begin' + end') >>> 1
//                        let aKey = a.[mid]
//                        let bKey = b.[diag - 1 - mid]
//
//                        let pred = if bounds = MgpuBoundsUpper then comp aKey bKey else not (comp bKey aKey)
//                                
//                        if pred then 
//                            begin' <- mid + 1
//                        else
//                            end' <- mid
//                    begin'
//
//            member this.DMergePath =
//                let comp = compOp.Device
//                <@ fun (a:deviceptr<'TC>) (aCount:int) (b:deviceptr<'TC>) (bCount:int) (diag:int) ->
//                    let comp = %comp
//                    let mutable begin' = max 0 (diag - bCount)
//                    let mutable end' = min diag aCount
//
//                    while begin' < end' do
//                        let mid = (begin' + end') >>> 1
//                        let aKey = a.[mid]
//                        let bKey = b.[diag - 1 - mid]
//
//                        let pred = if bounds = MgpuBoundsUpper then comp aKey bKey else not (comp bKey aKey)
//                        //let pred = (comp aKey bKey)
//                        if pred then 
//                            begin' <- mid + 1
//                        else
//                            end' <- mid
//                    begin' @>
//                    
//            member this.DMergePathInt =                    
//                    <@ fun (a:deviceptr<int>) (aCount:int) (b:deviceptr<int>) (bCount:int) (diag:int) ->                        
//                        let mutable begin' = max 0 (diag - bCount)
//                        let mutable end' = min diag aCount
//
//                        while begin' < end' do
//                            let mid = (begin' + end') >>> 1
//                            let aKey = a.[mid]
//                            let bKey = b.[diag - 1 - mid]
//
//                            let pred = if bounds = MgpuBoundsUpper then aKey < bKey else not (bKey < aKey)
//                            
//                            if pred then 
//                                begin' <- mid + 1
//                            else
//                                end' <- mid
//                        begin' @> }
//
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////                                                                                                      //
////      Balanced Path Search                                                                            //
////                                                                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//let balancedPathSearch (duplicates:bool) (intType:intInt) (compOp:IComp<'TC>) =
//        let binarySearch = binarySearch MgpuBoundsLower compOp
//        let biasedBinarySearch = biasedBinarySearch MgpuBoundsLower compOp
//        let mergePath = mergePath MgpuBoundsLower compOp
//        { new IBalancedPathSearch<'TC> with
//            member this.HBalancedPath =
//                let comp a b = compOp.Host a b
//                let binarySearch = binarySearch.HBinarySearch
//                let biasedBinarySearch = biasedBinarySearch.HBiasedBinarySearch
//                let mergePath = mergePath.HMergePath
//
//                fun (a:intC[]) (aCount:int) (b:intC[]) (bCount:int) (diag:int) (levels:int) ->
//                    let p = mergePath a aCount b bCount diag
//                    let mutable aIndex = p
//                    let bIndex = diag - p
//
//                    let mutable star = 0
//                    if bIndex < bCount then
//                        if duplicates then
//                            let x = b.[bIndex]
//
//                            let aStart = biasedBinarySearch a aIndex x levels
//                            let bStart = biasedBinarySearch b bIndex x levels
//
//                            let aRun = aIndex - aStart
//                            let mutable bRun = bIndex - bStart
//                            let xCount = aRun + bRun
//
//                            let mutable bAdvance = max (xCount >>> 1) (xCount - aRun)
//                            let bEnd = min bCount (bStart + bAdvance + 1)
//                            let bRunEnd =
//                                (binarySearch (Array.sub b 0 bIndex) (bEnd - bIndex) x) + bIndex
//                            bRun <- bRunEnd - bStart
//
//                            bAdvance <- min bAdvance bRun
//                            let aAdvance = xCount - bAdvance
//
//                            let roundUp = (aAdvance = (bAdvance + 1)) && (bAdvance < bRun)
//                            aIndex <- aStart + aAdvance
//
//                            if roundUp then star <- 1
//                    else
//                        if (aIndex > 0) && (aCount > 0) then
//                            let aKey = a.[aIndex - 1]
//                            let bKey = b.[bIndex]
//
//                            if not ( comp aKey bKey ) then star <- 1
//                                                
//                    let result = int2(aIndex,  star)
//                    result
//
//            member this.DBalancedPath =
//                let comp = compOp.Device
//                let mp = mergePath.DMergePath
//                let aStart = biasedBinarySearch.DBiasedBinarySearch
//                let bStart = biasedBinarySearch.DBiasedBinarySearch
//                let bRunEnd = binarySearch.DBinarySearch
//                <@ fun (a:deviceptr<'TC>) (aCount:int) (b:deviceptr<'TC>) (bCount:int) (diag:int) (levels:int) ->
//                    let comp = %comp
//                    let mp = %mp
//                    let aStart = %aStart
//                    let bStart = %bStart
//                    let bRunEnd = %bRunEnd
//                    let p = mp a aCount b bCount diag
//                    let mutable aIndex = p
//                    let mutable bIndex = diag - p
//
////                    match typeof<'TInt> with
////                    | x when x = typeof<int>
////                    | x when x = typeof<int64>
////                    | _ -> ()
//
//                    let mutable star = 0
//                    if bIndex < bCount then
//                        if duplicates then
//                            let x = b.[bIndex]
//                                                        
//                            let aStart = aStart a aIndex x levels
//                            let bStart = bStart b bIndex x levels
//
//                            let aRun = aIndex - aStart
//                            let mutable bRun = bIndex - bStart
//                            let xCount = aRun + bRun
//
//                            let mutable bAdvance = max (xCount >>> 1) (xCount - aRun)
//                            let bEnd = min bCount (bStart + bAdvance + 1)
//                            let bRunEnd = bRunEnd (b + bIndex) (bEnd - bIndex) x
//                            let bRunEnd = bRunEnd + bIndex
//                            
//                            bRun <- bRunEnd - bStart
//
//                            bAdvance <- min bAdvance bRun
//                            let aAdvance = xCount - bAdvance
//
//                            let roundUp = (aAdvance = (bAdvance + 1)) && (bAdvance < bRun)
//                            aIndex <- aStart + aAdvance
//
//                            if roundUp then star <- 1
//                    else
//                        if (aIndex > 0) && (aCount > 0) then
//                            let aKey = a.[aIndex - 1]
//                            let bKey = b.[bIndex]
//
//                            if not ( comp aKey bKey ) then star <- 1
//
//                    let result = int2(aIndex,  star)
//                    result @> }
//
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////                                                                                                      //
////      Segmented Merge Path Search                                                                     //
////                                                                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//let segmentedMergePath (compOp:IComp<'TC>) =        
//        { new ISegmentedMergePath<'TC> with
//            member this.HSegmentedMergePath =
//                let comp a b = compOp.Host a b
//                fun (keys:intC[]) (aOffset:int) (aCount:int) (bOffset:int) (bCount:int) (leftEnd:int) (rightStart:int) (diag:int) ->
//                    let mutable result = 0
//                    let test = 
//                        if (aOffset + diag) <= leftEnd then 
//                            result <- diag
//                        if (aOffset + diag) >= rightStart then 
//                            result <- aCount
//
//                        let bCount = min bCount (rightStart - bOffset)
//                        let mutable begin' = max (max (leftEnd - aOffset) 0) (diag - bCount)
//                        let mutable end' = min diag aCount
//
//                        while begin' < end' do
//                            let mid = (begin' + end') >>> 1
//                            let ai = aOffset + mid
//                            let bi = bOffset + diag - 1 - mid
//
//                            let pred = not ( comp keys.[bi] keys.[ai] )
//                            if pred then begin' <- mid + 1
//                            else end' <- mid
//                        result <- begin'                                                
//                    result
//
//            member this.DSegmentedMergePath =
//                let comp = compOp.Device
//                <@ fun (keys:deviceptr<'TC>) (aOffset:int) (aCount:int) (bOffset:int) (bCount:int) (leftEnd:int) (rightStart:int) (diag:int) ->
//                    let comp = %comp
//                    let mutable result = 0
//                    let test = 
//                        if (aOffset + diag) <= leftEnd then 
//                            result <- diag
//                        if (aOffset + diag) >= rightStart then 
//                            result <- aCount
//
//                        let bCount = min bCount (rightStart - bOffset)
//                        let mutable begin' = max (max (leftEnd - aOffset) 0) (diag - bCount)
//                        let mutable end' = min diag aCount
//
//                        while begin' < end' do
//                            let mid = (begin' + end') >>> 1
//                            let ai = aOffset + mid
//                            let bi = bOffset + diag - 1 - mid
//
//                            let pred = not (comp keys.[bi] keys.[ai])
//                            if pred then begin' <- mid + 1
//                            else end' <- mid
//                        result <- begin'
//                                                
//                    result @> }
