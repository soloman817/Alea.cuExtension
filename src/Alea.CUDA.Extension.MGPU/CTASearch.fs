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


type IBinarySearch<'TN, 'TI, 'T, 'TC> =
    abstract HBinarySearchIt : ('TI[] -> RPtr<int> -> RPtr<int> -> 'T -> int -> 'TC -> unit)
    abstract DBinarySearchIt : Expr<DevicePtr<'TI> -> RPtr<int> -> RPtr<int> -> 'T -> int -> 'TC -> unit>
    abstract HBiasedBinarySearch : ('TI[] -> int -> 'T -> int -> 'TC -> int)
    abstract DBiasedBinarySearch : Expr<DevicePtr<'TI> -> int -> 'T -> int -> 'TC -> int>
    abstract HBinarySearch : ('TI[] -> int -> 'T -> 'TC -> int)
    abstract DBinarySearch : Expr<DevicePtr<'TI> -> int -> 'T -> 'TC -> int>
    
type IMergeSearch<'TI, 'TC> =
    abstract HMergePath : ('TI[] -> int -> 'TI[] -> int -> int -> 'TC -> int)
    abstract DMergePath : Expr<DevicePtr<'TI> -> int -> DevicePtr<'TI> -> int -> int -> 'TC -> int>
    abstract HSegmentedMergePath : ('TI[] -> int -> int -> int -> int -> int -> int -> int -> 'TC -> int)
    abstract DSegmentedMergePath : Expr<DevicePtr<'TI> -> int -> int -> int -> int -> int -> int -> int -> 'TC -> int>

type IBalancedPathSearch<'TI, 'TC> =
    abstract HBalancedPath : ('TI[] -> int -> 'TI[] -> int -> int -> int -> 'TC -> int2)
    abstract DBalancedPath : Expr<DevicePtr<'TI> -> int -> DevicePtr<'TI> -> int -> int -> int -> 'TC -> int2>


type SearchOpType =
    | SearchOpTypeBinary
    | SearchOpTypeMerge
    | SearchOpTypeBalanced

let inline BinarySearch (bounds:MgpuBounds) =
        { new IBinarySearch<'TN, 'TI, 'T, 'TC> with
            member this.HBinarySearchIt =
                fun (data:'TI[]) (begin':RPtr<int>) (end':RPtr<int>) (key:'T) (shift:int) (comp:'TC) ->
                    let scale : 'TN = (1 <<< shift) - 1
                    let mid = ((begin'.[0] + scale * end'.[0]) >>> shift)
                    let key2 = data.[mid]
                    let mutable pred = false
                    match bounds with
                    | MgpuBoundsUpper -> pred <- comp key key2
                    | _ -> pred <- comp key2 key
                    if pred then 
                        begin'.Ref(0) := mid + 1
                    else
                        end'.Ref(0) := mid

            member this.DBinarySearchIt = 
                <@ fun (data:DevicePtr<'TI>) (begin':RPtr<int>) (end':RPtr<int>) (key:'T) (shift:int) (comp:'TC) ->
                    let scale : 'TN = (1 <<< shift) - 1
                    let mid = ((begin'.[0] + scale * end'.[0]) >>> shift)
                    let key2 = data.[mid]
                    let mutable pred = false
                    match bounds with
                    | MgpuBoundsUpper -> pred <- comp key key2
                    | _ -> pred <- comp key2 key
                    if pred then 
                        begin'.Ref(0) := mid + 1
                    else
                        end'.Ref(0) := mid @>

            member this.HBiasedBinarySearch = 
                fun (data:'TI[]) (count:int) (key:'T) (levels:int) (comp:'TC) ->
                    let begin' = 0
                    let end' = count

                    if levels >= 4 && begin' < end' then this.HBinarySearchIt data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 9 comp
                    if levels >= 3 && begin' < end' then this.HBinarySearchIt data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 7 comp
                    if levels >= 2 && begin' < end' then this.HBinarySearchIt data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 5 comp
                    if levels >= 1 && begin' < end' then this.HBinarySearchIt data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 4 comp

                    while begin' < end' do
                        this.HBinarySearchIt data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 1 comp

                    begin'

            member this.DBiasedBinarySearch = 
                <@ fun (data:DevicePtr<'TI>) (count:int) (key:'T) (levels:int) (comp:'TC) ->
                    let begin' = 0
                    let end' = count

                    let dbs = %(this.DBinarySearchIt)

                    if levels >= 4 && begin' < end' then dbs data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 9 comp
                    if levels >= 3 && begin' < end' then dbs data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 7 comp
                    if levels >= 2 && begin' < end' then dbs data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 5 comp
                    if levels >= 1 && begin' < end' then dbs data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 4 comp

                    while begin' < end' do
                        dbs data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 1 comp

                    begin' @>

            member this.HBinarySearch = 
                fun (data:'TI[]) (count:int) (key:'T) (comp:'TC) ->
                    let begin' = 0
                    let end' = count
                    while begin' < end' do
                        this.HBinarySearchIt data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 1 comp
                    begin'

            member this.DBinarySearch = 
                <@ fun (data:DevicePtr<'TI>) (count:int) (key:'T) (comp:'TC) ->
                    let begin' = 0
                    let end' = count
                    let dbs = %(this.DBinarySearchIt)

                    while begin' < end' do
                        dbs data (RPtr(int64(begin'))) (RPtr(int64(end'))) key 1 comp
                    begin' @> }

let inline MergeSearch (bounds:MgpuBounds) =
        { new IMergeSearch<'TI, 'TC> with
            member this.HMergePath =
                fun (a:'TI[]) (aCount:int) (b:'TI[]) (bCount:int) (diag:int) (comp:'TC) ->
                    let mutable begin' = max 0 (diag - bCount)
                    let mutable end' = min diag aCount

                    while begin' < end' do
                        let mid = (begin' + end') >>> 1
                        let aKey = a.[mid]
                        let bKey = b.[diag - 1 - mid]

                        let mutable pred = false
                        match bounds with
                        | MgpuBoundsUpper -> pred <- comp aKey bKey
                        | _ -> pred <- comp bKey aKey
                        if pred then 
                            begin' <- mid + 1
                        else
                            end' <- mid
                    begin'

            member this.DMergePath =
                <@ fun (a:DevicePtr<'TI>) (aCount:int) (b:DevicePtr<'TI>) (bCount:int) (diag:int) (comp:'TC) ->
                    let mutable begin' = max 0 (diag - bCount)
                    let mutable end' = min diag aCount

                    while begin' < end' do
                        let mid = (begin' + end') >>> 1
                        let aKey = a.[mid]
                        let bKey = b.[diag - 1 - mid]

                        let mutable pred = false
                        match bounds with
                        | MgpuBoundsUpper -> pred <- comp aKey bKey
                        | _ -> pred <- comp bKey aKey
                        if pred then 
                            begin' <- mid + 1
                        else
                            end' <- mid
                    begin' @>

            member this.HSegmentedMergePath =
                fun (keys:'TI[]) (aOffset:int) (aCount:int) (bOffset:int) (bCount:int) (leftEnd:int) (rightStart:int) (diag:int) (comp:'TC) ->
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

                            let pred : bool = not ( comp keys.[bi] keys.[ai] )
                            if pred then begin' <- mid + 1
                            else end' <- mid
                        result <- begin'
                                                
                    result

            member this.DSegmentedMergePath =
                <@ fun (keys:DevicePtr<'TI>) (aOffset:int) (aCount:int) (bOffset:int) (bCount:int) (leftEnd:int) (rightStart:int) (diag:int) (comp:'TC) ->
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

                            let pred : bool = not ( comp keys.[bi] keys.[ai] )
                            if pred then begin' <- mid + 1
                            else end' <- mid
                        result <- begin'
                                                
                    result @> }

let inline BalancedPathSearch (duplicates:bool) (bounds:MgpuBounds) =
        { new IBalancedPathSearch<'TI, 'TC> with
            member this.HBalancedPath =
                fun (a:'TI[]) (aCount:int) (b:'TI[]) (bCount:int) (diag:int) (levels:int) (comp:'TC) ->
                    let p = (MergeSearch MgpuBounds.MgpuBoundsLower).HMergePath a aCount b bCount diag comp
                    let mutable aIndex = p
                    let bIndex = diag - p

                    let mutable star = false
                    if bIndex < bCount then
                        if duplicates then
                            let x = b.[bIndex]

                            let aStart = (BinarySearch MgpuBounds.MgpuBoundsLower).HBiasedBinarySearch a aIndex x levels comp
                            let bStart = (BinarySearch MgpuBounds.MgpuBoundsLower).HBiasedBinarySearch b bIndex x levels comp

                            let aRun = aIndex - aStart
                            let mutable bRun = bIndex - bStart
                            let xCount = aRun + bRun

                            let mutable bAdvance = max (xCount >>> 1) (xCount - aRun)
                            let bEnd = min bCount (bStart + bAdvance + 1)
                            let bRunEnd = ((BinarySearch MgpuBounds.MgpuBoundsLower).HBinarySearch) 
                            let passB = b.[0] + bIndex
                            let passB = Array.sub(b) passB (b.Length - 1)
                            let bRunEnd =  bRunEnd passB (bEnd - bIndex) x comp + bIndex
                            bRun <- bRunEnd - bStart

                            bAdvance <- min bAdvance bRun
                            let aAdvance = xCount - bAdvance

                            let roundUp = (aAdvance = (bAdvance + 1)) && (bAdvance < bRun)
                            aIndex <- aStart + aAdvance

                            if roundUp then star <- true
                    else
                        if (aIndex > 0) && (aCount > 0) then
                            let aKey = a.[aIndex - 1]
                            let bKey = b.[bIndex]

                            if ( not( comp aKey bKey ) ) then star <- true

                    let str = 
                        if star then 1
                        else 0
                    let result = int2(aIndex,  str)
                    result

            member this.DBalancedPath = 
                <@ fun (a:DevicePtr<'TI>) (aCount:int) (b:DevicePtr<'TI>) (bCount:int) (diag:int) (levels:int) (comp:'TC) ->
                    let p = %((MergeSearch MgpuBounds.MgpuBoundsLower).DMergePath) 
                    let p = p a aCount b bCount diag comp
                    let mutable aIndex = p
                    let bIndex = diag - p

                    let mutable star = false
                    if bIndex < bCount then
                        if duplicates then
                            let x = b.[bIndex]

                            let aStart = %((BinarySearch MgpuBounds.MgpuBoundsLower).DBiasedBinarySearch)
                            let aStart = aStart a aIndex x levels comp
                            let bStart = %((BinarySearch MgpuBounds.MgpuBoundsLower).DBiasedBinarySearch)
                            let bStart = bStart b bIndex x levels comp

                            let aRun = aIndex - aStart
                            let mutable bRun = bIndex - bStart
                            let xCount = aRun + bRun

                            let mutable bAdvance = max (xCount >>> 1) (xCount - aRun)
                            let bEnd = min bCount (bStart + bAdvance + 1)
                            let bRunEnd = %((BinarySearch MgpuBounds.MgpuBoundsLower).DBinarySearch)
                            let bRunEnd = bRunEnd (b.Ptr(0) + bIndex) (bEnd - bIndex) x comp 
                            let bRunEnd = bRunEnd + bIndex
                            
                            bRun <- bRunEnd - bStart

                            bAdvance <- min bAdvance bRun
                            let aAdvance = xCount - bAdvance

                            let roundUp = (aAdvance = (bAdvance + 1)) && (bAdvance < bRun)
                            aIndex <- aStart + aAdvance

                            if roundUp then star <- true
                    else
                        if (aIndex > 0) && (aCount > 0) then
                            let aKey = a.[aIndex - 1]
                            let bKey = b.[bIndex]

                            if ( not( comp aKey bKey ) ) then star <- true

                    let str = 
                        if star then 1
                        else 0
                    let result = int2(aIndex,  str)
                    result @> }