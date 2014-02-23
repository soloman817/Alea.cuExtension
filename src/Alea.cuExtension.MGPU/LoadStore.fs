[<AutoOpen>]
module Alea.cuExtension.MGPU.LoadStore

// this file maps to loadstore.cuh. which acturally did the different matrix
// order changing. Please check http://www.moderngpu.com/scan/globalscan.html#Scan
// and check the concept of transposeValues. 

open Alea.CUDA


// @COMMENT@ sync can use bool here, bool type cannot be used in kernel arugment, but here
// it is just a lambda function, so it is ok.
// Cooperative load functions
let deviceSharedToReg (NT:int) (VT:int) =
    <@ fun (count:int) (data:deviceptr<int>) (tid:int) (reg:deviceptr<int>) (sync:bool) ->
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                reg.[i] <- data.[NT * i + tid]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then reg.[i] <- data.[index]
        if sync then __syncthreads() @>

// Cooperative store functions
let deviceRegToShared (NT:int) (VT:int) =
    <@ fun (count:int) (reg:deviceptr<int>) (tid:int) (dest:deviceptr<int>) (sync:bool) ->
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                dest.[NT * i + tid] <- reg.[i]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then
                    dest.[index] <- reg.[i]
        if sync then __syncthreads() @>

let deviceRegToGlobal (NT:int) (VT:int) =
    <@ fun (count:int) (reg:deviceptr<int>) (tid:int) (dest:deviceptr<int>) (sync:bool) ->
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < count then
                dest.[index] <- reg.[i]
        if sync then __syncthreads() @>

//let deviceGlobalToReg (NT:int) (VT:int) = deviceSharedToReg NT VT
let deviceGlobalToReg (NT:int) (VT:int) =
    <@ fun (count:int) (data:deviceptr<int>) (tid:int) (reg:deviceptr<int>) (sync:bool) ->
            if count >= NT * VT then
                for i = 0 to VT - 1 do
                    reg.[i] <- data.[NT * i + tid]
            else
                for i = 0 to VT - 1 do
                    let index = NT * i + tid
                    if index < count then reg.[i] <- data.[index]
            if sync then __syncthreads() @>


// DeviceMemToMemLoop
// Transfer from shared memory to global, or glbal to shared, for transfers that are smaller than NT * VT in the average case.
// The goal is to reduce unnecessary comparison logic
let deviceMemToMem4 (NT:int) (VT:int) =
    <@ fun (count:int) (source:deviceptr<int>) (tid:int) (dest:deviceptr<int>) (sync:bool) ->
        let x = __local__.Array<int>(VT)
        let count' = if VT < 4 then VT else 4
        if (count >= NT * VT) then
            for i = 0 to count' - 1 do
                x.[i] <- source.[NT * i + tid]
            for i = 0 to count' - 1 do
                dest.[NT * i + tid] <- x.[i]
        else
            for i = 0 to count' - 1 do
                let index = NT * i + tid
                if index < count then
                    x.[i] <- source.[NT * i + tid]
            for i = 0 to count' - 1 do
                let index = NT * i + tid
                if index < count then
                    dest.[index] <- x.[i]
        if sync then __syncthreads() @>

let deviceMemToMemLoop (NT:int) =
    let deviceMemToMem4 = deviceMemToMem4 NT 4
    <@ fun (count:int) (source:deviceptr<int>) (tid:int) (dest:deviceptr<int>) (sync:bool) ->
        let deviceMemToMem4 = %deviceMemToMem4
        let mutable i = 0
        while i < count do
            deviceMemToMem4 (count - i) (source + i) tid (dest + i) false
            i <- i + 4 * NT
        if sync then __syncthreads() @>


// Functions to copy between shared and global memory where the average case is to transfer NT * VT elements.
let deviceSharedToGlobal (NT:int) (VT:int) =
    <@ fun (count:int) (source:deviceptr<int>) (tid:int) (dest:deviceptr<int>) (sync:bool) ->
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < count then
                dest.[NT * i + tid] <- source.[NT * i + tid]
        if sync then __syncthreads() @>

let deviceGlobalToShared (NT:int) (VT:int)  =
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceRegToShared = deviceRegToShared NT VT
    <@ fun (count:int) (source:deviceptr<int>) (tid:int) (dest:deviceptr<int>) (sync:bool) ->
        let reg = __local__.Array<int>(VT) |> __array_to_ptr
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceRegToShared = %deviceRegToShared
        deviceGlobalToReg count source tid reg false
        deviceRegToShared (NT * VT) reg tid dest sync @>

let deviceGlobalToGlobal (NT:int) (VT:int) =
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    <@ fun (count:int) (source:deviceptr<int>) (tid:int) (dest:deviceptr<int>) (sync:bool) ->
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceRegToGlobal = %deviceRegToGlobal

        let values = __local__.Array<int>(VT) |> __array_to_ptr
        deviceGlobalToReg count source tid values false
        deviceRegToGlobal count values tid dest sync @>


// Gather/scatter functions
let deviceGather (NT:int) (VT:int) =
    <@ fun (count:int) (data:deviceptr<int>) (indices:deviceptr<int>) (tid:int) (reg:deviceptr<int>) (sync:bool) ->
        if count >= (NT * VT) then
            for i = 0 to VT - 1 do
                reg.[i] <- data.[indices.[i]]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then
                    reg.[i] <- data.[indices.[i]]
        if sync then __syncthreads() @>

let deviceScatter (NT:int) (VT:int) =
    <@ fun (count:int) (reg:deviceptr<int>) (tid:int) (indices:deviceptr<int>) (data:deviceptr<int>) (sync:bool) ->
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                data.[indices.[i]] <- reg.[i]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then
                    data.[indices.[i]] <- reg.[i]
        if sync then __syncthreads() @>

// Cooperative transpose functions (strided to thread order)
let deviceThreadToShared (VT:int) =
    <@ fun (threadReg:deviceptr<int>) (tid:int) (shared:deviceptr<int>) (sync:bool) ->
        for i = 0 to VT - 1 do
            shared.[VT * tid + i] <- threadReg.[i]
        if sync then __syncthreads() @>

let deviceSharedToThread (VT:int) =
    <@ fun (shared:deviceptr<int>) (tid:int) (threadReg:deviceptr<int>) (sync:bool) ->
        for i = 0 to VT - 1 do
            threadReg.[i] <- shared.[VT * tid + i]
        if sync then __syncthreads() @>

///////////
let deviceLoad2ToSharedA (NT:int) (VT0:int) (VT1:int) =
    let deviceRegToShared = deviceRegToShared NT VT1
    <@ fun (a_global:deviceptr<int>) (aCount:int) (b_global:deviceptr<int>) (bCount:int) (tid:int) (shared:deviceptr<int>) (sync:bool) ->
        let deviceRegToShared = %deviceRegToShared
        let b0 = (int(b_global - a_global) / sizeof<int>) - aCount
        
        let total = aCount + bCount
        let reg = __local__.Array<int>(VT1)
        if total >= NT * VT0 then
            for i = 0 to VT0 - 1 do
                let index = NT * i + tid
                let x = if index >= aCount then b0 else 0
                reg.[i] <- a_global.[index + x]
        else
            for i = 0 to VT0 - 1 do
                let index = NT * i + tid
                if index < total then
                    let x = if index >= aCount then b0 else 0
                    reg.[i] <- a_global.[index + x]

        for i = VT0 to VT1 - 1 do
            let index = NT * i + tid
            if index < total then
                let x = if index >= aCount then b0 else 0
                reg.[i] <- a_global.[index + x]
        let reg = reg |> __array_to_ptr
        deviceRegToShared (NT * VT1) reg tid shared sync @>


let deviceLoad2ToSharedB (NT:int) (VT0:int) (VT1:int) =
    let deviceRegToShared = deviceRegToShared NT VT1
    <@ fun (a_global:deviceptr<int>) (aCount:int) (b_global:deviceptr<int>) (bCount:int) (tid:int) (shared:deviceptr<int>) (sync:bool) ->
        let deviceRegToShared = %deviceRegToShared
        //let mutable b_global = b_global
        
        let b_global = b_global - aCount
        let total = aCount + bCount

        let reg = __local__.Array<int>(VT1)
        
        if total >= NT * VT0 then
            for i = 0 to VT0 - 1 do
                let index = NT * i + tid
                reg.[i] <- if index < aCount then a_global.[index] else b_global.[index]                
        else
            for i = 0 to VT0 - 1 do
                let index = NT * i + tid
                if index < aCount then
                    reg.[i] <- a_global.[index]
                else if index < total then
                    reg.[i] <- b_global.[index]
                else
                    ()

        for i = VT0 to VT1 - 1 do
            let index = NT * i + tid
            if index < aCount then
                reg.[i] <- a_global.[index]
            else if index < total then
                reg.[i] <- b_global.[index]
        let reg = reg |> __array_to_ptr
        deviceRegToShared (NT * VT1) reg tid shared sync @>


// DeviceGatherGlobalToGlobal
let deviceGatherGlobalToGlobal (NT:int) (VT:int) =
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    <@ fun (count:int) (data_global:deviceptr<int>) (indices_shared:deviceptr<int>) (tid:int) (dest_global:deviceptr<int>) (sync:bool) ->
        let deviceRegToGlobal = %deviceRegToGlobal
        
        let values = __local__.Array<int>(VT)
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < count then
                let gather = indices_shared.[index]
                values.[i] <- data_global.[gather]
        if sync then __syncthreads()

        let values = values |> __array_to_ptr
        deviceRegToGlobal count values tid dest_global false @>


// DeviceTransferMergeValues
// Gather in a merge-like value from two input arrays and store to a single output.
// Like DeviceGatherGlobalToGlobal, but for two arrays at once.
let deviceTransferMergeValuesA (NT:int) (VT:int) =
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    <@ fun (count:int) (a_global:deviceptr<int>) (b_global:deviceptr<int>) (bStart:int) (indices_shared:deviceptr<int>) (tid:int) (dest_global:deviceptr<int>) (sync:bool) ->
        let deviceRegToGlobal = %deviceRegToGlobal
                     
        let values = __local__.Array<int>(VT)
        let b_global = b_global - bStart
        
        if count >= ( NT * VT ) then
            for i = 0 to VT - 1 do
                let gather = indices_shared.[NT * i + tid]
                values.[i] <- if gather < bStart then a_global.[gather] else b_global.[gather]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                let gather = indices_shared.[index]
                if index < count then
                    values.[i] <- if gather < bStart then a_global.[gather] else b_global.[gather]

        if sync then __syncthreads()

        let values = values |> __array_to_ptr
        deviceRegToGlobal count values tid dest_global false @>

let deviceTransferMergeValuesB (NT:int) (VT:int) =
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    <@ fun (count:int) (a_global:deviceptr<int>) (b_global:deviceptr<int>) (bStart:int) (indices_shared:deviceptr<int>) (tid:int) (dest_global:deviceptr<int>) (sync:bool) ->
        let deviceRegToGlobal = %deviceRegToGlobal
        
        let values = __local__.Array<int>(VT)
        let bOffset = (int(b_global - a_global) / sizeof<int>) - bStart
        
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                let mutable gather = indices_shared.[NT * i + tid]
                if gather >= bStart then gather <- gather + bOffset
                values.[i] <- a_global.[gather]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                let mutable gather = indices_shared.[index]
                if gather >= bStart then gather <- gather + bOffset
                if index < count then values.[i] <- a_global.[gather]

        if sync then __syncthreads()

        let values = values |> __array_to_ptr
        deviceRegToGlobal count values tid dest_global false @>