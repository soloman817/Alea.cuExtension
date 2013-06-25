module Alea.CUDA.Extension.MGPU.CTASearch

// this file maps to ctasearch.cuh

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan


type IBinarySearch<'TI, 'TK> =
    abstract Identity : 'TI
    abstract HBinarySearchIt : ('TI[] -> int ref -> int ref -> 'TK -> int -> unit)
    abstract DBinarySearchIt : Expr<DevicePtr<'TI> -> RWPtr<int> -> RWPtr<int> -> 'TK -> int -> unit>    
    abstract HBiasedBinarySearch : ('TI[] -> int -> 'T -> int -> int)
    abstract DBiasedBinarySearch : Expr<DevicePtr<'TI> -> int -> 'T -> int -> int>
    abstract HBinarySearch : ('TI[] -> int -> 'TK -> int)
    abstract DBinarySearch : Expr<DevicePtr<'TI> -> int -> 'TK -> int>
    
type IMergeSearch<'TI, 'TK> =
    abstract HMergePath : ('TI[] -> int -> 'TI[] -> int -> int -> int)
    abstract DMergePath : Expr<DevicePtr<'TI> -> int -> DevicePtr<'TI> -> int -> int -> int>
    abstract HSegmentedMergePath : ('TI[] -> int -> int -> int -> int -> int -> int -> int -> int)
    abstract DSegmentedMergePath : Expr<DevicePtr<'TI> -> int -> int -> int -> int -> int -> int -> int -> int>

type IBalancedPathSearch<'TI, 'TK> =
    abstract HBalancedPath : ('TI[] -> int -> 'TI[] -> int -> int -> int -> int2)
    abstract DBalancedPath : Expr<DevicePtr<'TI> -> int -> DevicePtr<'TI> -> int -> int -> int -> int2>


type SearchOpType =
    | SearchOpTypeBinary
    | SearchOpTypeMerge
    | SearchOpTypeBalanced


[<ReflectedDefinition>] 
let MgpuBoundsLower = 0
[<ReflectedDefinition>] 
let MgpuBoundsUpper = 1




let inline binarySearchFun (bounds:int) (compOp:IComp<'T>)  =
        { new IBinarySearch<'T, 'T> with
            member this.Identity = compOp.Identity
            member this.HBinarySearchIt =
                let comp a b = compOp.Host a b
                fun (data:'T[]) (begin':int ref) (end':int ref) (key:'T) (shift:int) ->
                    let scale = (1 <<< shift) - 1
                    let mid = ((!begin' + scale * !end') >>> shift)
                    let key2 = data.[mid]
                    let pred = 
                        if bounds = MgpuBoundsUpper then
                            if (comp key key2) = 0 then 1 else 0
                        else
                            comp key2 key

                    if pred = 1 then 
                        begin' := mid + 1
                    else
                        end' := mid                    
                        
            member this.DBinarySearchIt = 
                let comp = compOp.Device
                <@ fun (data:DevicePtr<'T>) (begin':RWPtr<int>) (end':RWPtr<int>) (key:'T) (shift:int) ->
                    let comp = %comp
                    let scale = (1 <<< shift) - 1
                    let mid = ((begin'.[0] + scale * end'.[0]) >>> shift)
                    let key2 = data.[mid]
                    let pred =
                        if bounds = MgpuBoundsUpper then
                            if (comp key key2) = 0 then 1 else 0
                        else
                            comp key2 key

                    if pred = 1 then 
                        begin'.[0] <- mid + 1
                    else
                        end'.[0] <- mid @>

            member this.HBiasedBinarySearch = 
                let comp a b = compOp.Host a b
                fun (data:'T[]) (count:int) (key:'T) (levels:int) ->
                    let begin' = ref 0
                    let end' = ref count

                    if levels >= 4 && begin' < end' then this.HBinarySearchIt data begin' end' key 9 //comp
                    if levels >= 3 && begin' < end' then this.HBinarySearchIt data begin' end' key 7 //comp
                    if levels >= 2 && begin' < end' then this.HBinarySearchIt data begin' end' key 5 //comp
                    if levels >= 1 && begin' < end' then this.HBinarySearchIt data begin' end' key 4 //comp

                    while begin' < end' do
                        this.HBinarySearchIt data begin' end' key 1 //comp
                    begin'.contents

            member this.DBiasedBinarySearch = 
                let comp = compOp.Device
                <@ fun (data:DevicePtr<'T>) (count:int) (key:'T) (levels:int) ->
                    let comp = %comp
                    let begin' = 0
                    let end' = count

                    let dbs = %(this.DBinarySearchIt)

                    if levels >= 4 && begin' < end' then dbs data (RWPtr(int64(begin'))) (RWPtr(int64(end'))) key 9 //comp
                    if levels >= 3 && begin' < end' then dbs data (RWPtr(int64(begin'))) (RWPtr(int64(end'))) key 7 //comp
                    if levels >= 2 && begin' < end' then dbs data (RWPtr(int64(begin'))) (RWPtr(int64(end'))) key 5 //comp
                    if levels >= 1 && begin' < end' then dbs data (RWPtr(int64(begin'))) (RWPtr(int64(end'))) key 4 //comp

                    while begin' < end' do
                        dbs data (RWPtr(int64(begin'))) (RWPtr(int64(end'))) key 1 //comp
                    begin' @>

            member this.HBinarySearch = 
                fun (data:'T[]) (count:int) (key:'T) ->
                    let begin' = ref 0
                    let end' = ref count
                    while begin' < end' do
                        this.HBinarySearchIt data begin' end' key 1
                    !begin'

            member this.DBinarySearch = 
                let dbs = this.DBinarySearchIt                
                <@ fun (data:DevicePtr<'T>) (count:int) (key:'T) ->
                    let dbs = %dbs
                    
                    let begin' = __local__<int>(1).Ptr(0)
                    begin'.[0] <- 0

                    let end' = __local__<int>(1).Ptr(0)
                    end'.[0] <- count
                    
                    while begin'.[0] < end'.[0] do
                        dbs data begin' end' key 1
                    begin'.[0] @> }


let inline mergeSearch (bounds:int) (compOp:IComp<'TC>) =
        { new IMergeSearch<'TI, 'TC> with
            member this.HMergePath =
                let comp a b = compOp.Host a b
                fun (a:'TI[]) (aCount:int) (b:'TI[]) (bCount:int) (diag:int) ->
                    let mutable begin' = max 0 (diag - bCount)
                    let mutable end' = min diag aCount

                    while begin' < end' do
                        let mid = (begin' + end') >>> 1
                        let aKey = a.[mid]
                        let bKey = b.[diag - 1 - mid]

                        let pred = if bounds = MgpuBoundsUpper then comp aKey bKey else (comp bKey aKey) - 1
                                
                        if pred = 1 then 
                            begin' <- mid + 1
                        else
                            end' <- mid
                    begin'

            member this.DMergePath =
                let comp = compOp.Device
                <@ fun (a:DevicePtr<'TI>) (aCount:int) (b:DevicePtr<'TI>) (bCount:int) (diag:int) ->
                    let comp = %comp
                    let mutable begin' = max 0 (diag - bCount)
                    let mutable end' = min diag aCount

                    while begin' < end' do
                        let mid = (begin' + end') >>> 1
                        let aKey = a.[mid]
                        let bKey = b.[diag - 1 - mid]

                        let pred = if bounds = MgpuBoundsUpper then comp aKey bKey else (comp bKey aKey) - 1
                        if pred = 1 then 
                            begin' <- mid + 1
                        else
                            end' <- mid
                    begin' @>

            member this.HSegmentedMergePath =
                let comp a b = compOp.Host a b
                fun (keys:'TI[]) (aOffset:int) (aCount:int) (bOffset:int) (bCount:int) (leftEnd:int) (rightStart:int) (diag:int) ->
                    let mutable result = 0
                    let test = 
                        if (aOffset + diag) <= leftEnd then 
                            result <- diag
                        if (aOffset + diag) >= rightStart then 
                            result <- aCount

                        let bCount = min bCount (rightStart - bOffset)
                        let mutable begin' = max (max (leftEnd - aOffset) 0) (diag - bCount)
                        let mutable end' = min diag aCount

                        while begin' < end' do
                            let mid = (begin' + end') >>> 1
                            let ai = aOffset + mid
                            let bi = bOffset + diag - 1 - mid

                            let pred = -( comp keys.[bi] keys.[ai] )
                            if pred = 1 then begin' <- mid + 1
                            else end' <- mid
                        result <- begin'
                                                
                    result

            member this.DSegmentedMergePath =
                let comp = compOp.Device
                <@ fun (keys:DevicePtr<'TI>) (aOffset:int) (aCount:int) (bOffset:int) (bCount:int) (leftEnd:int) (rightStart:int) (diag:int) ->
                    let comp = %comp
                    let mutable result = 0
                    let test = 
                        if (aOffset + diag) <= leftEnd then 
                            result <- diag
                        if (aOffset + diag) >= rightStart then 
                            result <- aCount

                        let bCount = min bCount (rightStart - bOffset)
                        let mutable begin' = max (max (leftEnd - aOffset) 0) (diag - bCount)
                        let mutable end' = min diag aCount

                        while begin' < end' do
                            let mid = (begin' + end') >>> 1
                            let ai = aOffset + mid
                            let bi = bOffset + diag - 1 - mid

                            let pred = if comp keys.[bi] keys.[ai] = 1 then 0 else 1
                            if pred = 1 then begin' <- mid + 1
                            else end' <- mid
                        result <- begin'
                                                
                    result @> }

let inline balancedPathSearch (duplicates:int) (bounds:int) (compOp:IComp<'TC>) =
        { new IBalancedPathSearch<'TI, 'TC> with
            member this.HBalancedPath =
                let comp a b = compOp.Host a b
                fun (a:'TI[]) (aCount:int) (b:'TI[]) (bCount:int) (diag:int) (levels:int) ->
                    let p = (mergeSearch MgpuBoundsLower compOp).HMergePath a aCount b bCount diag //comp
                    let mutable aIndex = p
                    let bIndex = diag - p

                    let mutable star = 0
                    if bIndex < bCount then
                        if duplicates <> 0 then
                            let x = b.[bIndex]

                            let aStart = (binarySearchFun MgpuBoundsLower compOp).HBiasedBinarySearch a aIndex x levels //comp
                            let bStart = (binarySearchFun MgpuBoundsLower compOp).HBiasedBinarySearch b bIndex x levels //comp

                            let aRun = aIndex - aStart
                            let mutable bRun = bIndex - bStart
                            let xCount = aRun + bRun

                            let mutable bAdvance = max (xCount >>> 1) (xCount - aRun)
                            let bEnd = min bCount (bStart + bAdvance + 1)
                            let bRunEnd =
                                (((binarySearchFun MgpuBoundsLower compOp).HBinarySearch) (Array.sub b 0 bIndex) (bEnd - bIndex) x) + bIndex
                            bRun <- bRunEnd - bStart

                            bAdvance <- min bAdvance bRun
                            let aAdvance = xCount - bAdvance

                            let roundUp = (aAdvance = (bAdvance + 1)) && (bAdvance < bRun)
                            aIndex <- aStart + aAdvance

                            if roundUp then star <- 1
                    else
                        if (aIndex > 0) && (aCount > 0) then
                            let aKey = a.[aIndex - 1]
                            let bKey = b.[bIndex]

                            if ( comp aKey bKey = 0 ) then star <- 1
                                                
                    let result = int2(aIndex,  star)
                    result

            member this.DBalancedPath =
                let comp = compOp.Device
                let p = (mergeSearch MgpuBoundsLower compOp).DMergePath
                let aStart = (binarySearchFun MgpuBoundsLower compOp).DBiasedBinarySearch
                let bStart = (binarySearchFun MgpuBoundsLower compOp).DBiasedBinarySearch
                let bRunEnd = (binarySearchFun MgpuBoundsLower compOp).DBinarySearch
                <@ fun (a:DevicePtr<'TI>) (aCount:int) (b:DevicePtr<'TI>) (bCount:int) (diag:int) (levels:int) ->
                    let comp = %comp
                    let p = %p
                    let aStart = %aStart
                    let bStart = %bStart
                    let bRunEnd = %bRunEnd

                    let p = p a aCount b bCount diag
                    let mutable aIndex = p
                    let bIndex = diag - p

                    let mutable star = 0
                    if bIndex < bCount then
                        if duplicates <> 0 then
                            let x = b.[bIndex]
                                                        
                            let aStart = aStart a aIndex x levels
                            let bStart = bStart b bIndex x levels

                            let aRun = aIndex - aStart
                            let mutable bRun = bIndex - bStart
                            let xCount = aRun + bRun

                            let mutable bAdvance = max (xCount >>> 1) (xCount - aRun)
                            let bEnd = min bCount (bStart + bAdvance + 1)
                            let bRunEnd = bRunEnd (b.Ptr(0) + bIndex) (bEnd - bIndex) x
                            let bRunEnd = bRunEnd + bIndex
                            
                            bRun <- bRunEnd - bStart

                            bAdvance <- min bAdvance bRun
                            let aAdvance = xCount - bAdvance

                            let roundUp = (aAdvance = (bAdvance + 1)) && (bAdvance < bRun)
                            aIndex <- aStart + aAdvance

                            if roundUp then star <- 1
                    else
                        if (aIndex > 0) && (aCount > 0) then
                            let aKey = a.[aIndex - 1]
                            let bKey = b.[bIndex]

                            if ( comp aKey bKey = 0 ) then star <- 1

                    let result = int2(aIndex,  star)
                    result @> }

                    
