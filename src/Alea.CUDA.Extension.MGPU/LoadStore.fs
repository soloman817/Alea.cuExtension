module Alea.CUDA.Extension.MGPU.LoadStore

// this file maps to loadstore.cuh. which acturally did the different matrix
// order changing. Please check http://www.moderngpu.com/scan/globalscan.html#Scan
// and check the concept of transposeValues. 

open Alea.CUDA

let deviceSharedToReg (NT:int) (VT:int) =
    <@ fun (count:int) (data:RPtr<'T>) (tid:int) (reg:RWPtr<'T>) (sync:bool) ->
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                reg.[i] <- data.[NT * i + tid]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then reg.[i] <- data.[index]
        if sync then __syncthreads() @>

let deviceGlobalToReg (NT:int) (VT:int) = deviceSharedToReg NT VT


let deviceRegToShared (NT:int) (VT:int) =
    <@ fun (count:int) (reg:RWPtr<'T>) (tid:int) (dest:RWPtr<'T>) (sync:bool) ->
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                dest.[NT * i + tid] <- reg.[i]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then
                    dest.[index] <- reg.[i]
        if sync then __syncthreads() @>


let deviceSharedToGlobal (NT:int) (VT:int) =
    <@ fun (count:int) (source:RPtr<'T>) (tid:int) (dest:RWPtr<'T>) (sync:bool) ->
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < count then
                dest.[NT * i + tid] <- source.[NT * i + tid]
        if sync then __syncthreads() @>

let deviceGlobalToShared (NT:int) (VT:int)  =
//T reg[VT];
//DeviceGlobalToReg<NT, VT>(count, source, tid, reg, false);
//DeviceRegToShared<NT, VT>(NT * VT, reg, tid, dest, sync);
    <@ fun (count:int) (data:RPtr<'T>) (tid:int) (dest:RWPtr<'T>) (sync:bool) ->
        let reg = __local__<'T>(VT).Ptr(0)
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                reg.[i] <- data.[NT * i + tid]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then reg.[i] <- data.[index]
        if sync then __syncthreads()

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
    <@ fun (count:int) (reg:RWPtr<'T>) (tid:int) (dest:RWPtr<'T>) (sync:bool) =
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < count then
                dest.[index] <- reg.[i]
        if sync then __syncthreads() @>

let deviceGather (NT:int) (VT:int) =
    <@ fun (count:int) (data:RPtr<'T>) (indices:RPtr<'T>) (tid:int) (reg:RWPtr<'T>) (sync:bool) ->
        if count >= (NT * VT) then
            for i = 0 to VT - 1 do
                reg.[i] <- data.[indices.[i]]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then
                    reg.[i] <- data.[indices.[i]]
        if sync then __syncthreads() @>
