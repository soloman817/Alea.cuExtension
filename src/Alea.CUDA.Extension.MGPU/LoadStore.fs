module Alea.CUDA.Extension.MGPU.LoadStore

// this file maps to loadstore.cuh. which acturally did the different matrix
// order changing. Please check http://www.moderngpu.com/scan/globalscan.html#Scan
// and check the concept of transposeValues. 

open Alea.CUDA

let [<ReflectedDefinition>] doSync = 1
let [<ReflectedDefinition>] dontSync = 0

let deviceSharedToReg (NT:int) (VT:int) =
    <@ fun (count:int) (data:RWPtr<'T>) (tid:int) (reg:RWPtr<'T>) (sync:int) ->
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                reg.[i] <- data.[NT * i + tid]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then reg.[i] <- data.[index]
        if sync <> 0 then __syncthreads() @>

let deviceGlobalToReg (NT:int) (VT:int) = deviceSharedToReg NT VT


let deviceRegToShared (NT:int) (VT:int) =
    <@ fun (count:int) (reg:RWPtr<'T>) (tid:int) (dest:RWPtr<'T>) (sync:int) ->
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                dest.[NT * i + tid] <- reg.[i]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then
                    dest.[index] <- reg.[i]
        if sync <> 0 then __syncthreads() @>


let deviceSharedToGlobal (NT:int) (VT:int) =
    <@ fun (count:int) (source:RWPtr<'T>) (tid:int) (dest:RWPtr<'T>) (sync:int) ->
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < count then
                dest.[NT * i + tid] <- source.[NT * i + tid]
        if sync <> 0 then __syncthreads() @>

let deviceGlobalToShared (NT:int) (VT:int)  =
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceRegToShared = deviceRegToShared NT VT
    <@ fun (count:int) (source:RWPtr<'T>) (tid:int) (dest:RWPtr<'T>) (sync:int) ->
        let reg = __local__<'T>(VT).Ptr(0)
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceRegToShared = %deviceRegToShared
        deviceGlobalToReg count source tid reg 0
        deviceRegToShared (NT*VT) reg tid dest sync @>

let deviceRegToGlobal (NT:int) (VT:int) =
    <@ fun (count:int) (reg:RWPtr<'T>) (tid:int) (dest:RWPtr<'T>) (sync:int) ->
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < count then
                dest.[index] <- reg.[i]
        if sync <> 0 then __syncthreads() @>

let deviceGather (NT:int) (VT:int) =
    <@ fun (count:int) (data:RWPtr<'T>) (indices:RWPtr<'T>) (tid:int) (reg:RWPtr<'T>) (sync:int) ->
        if count >= (NT * VT) then
            for i = 0 to VT - 1 do
                reg.[i] <- data.[indices.[i]]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then
                    reg.[i] <- data.[indices.[i]]
        if sync <> 0 then __syncthreads() @>
