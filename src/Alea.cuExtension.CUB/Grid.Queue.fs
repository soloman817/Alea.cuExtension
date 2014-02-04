[<AutoOpen>]
module Alea.cuExtension.CUB.Grid.Queue

open System
open Microsoft.FSharp.NativeInterop
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

let FILL = 0
let DRAIN = 1


let fillAndResetDrain fill_size stream =
    fun (d_counters:deviceptr<int>) ->
        d_counters.[FILL] <- fill_size
        d_counters.[DRAIN] <- 0




[<Record>]
type GridQueue =
    {
        d_counters : deviceptr<Offset>
    }

    
    member inline this.FillAndResetDrain(fill_size, ?stream:CUstream) =
        let stream = if stream.IsSome then stream.Value else 0n
        fun (locale:Locale) ->
            match locale with
            | Locale.Device ->
                this.d_counters.[FILL] <- fill_size
                this.d_counters.[DRAIN] <- 0
//                Alea.CUDA.CUDAInterop.cudaError_enum.CUDA_SUCCESS
            | Locale.Host ->
                let counters = Array.zeroCreate 2
                counters |> Array.set <| FILL <| fill_size
                counters |> Array.set <| DRAIN <| 0
//                Alea.CUDA.CUDAInterop.cuMemcpyHtoDAsync( this.d_counters.Handle, &&(counters) |> NativeInterop.NativePtr.toNativeInt, sizeof<Offset> * 2, stream)

    member inline this.ResetDrain(?stream:CUstream) =
        let stream = if stream.IsSome then stream.Value else 0n
        fun (locale:Locale) ->
            match locale with
            | Locale.Device -> this.d_counters.[DRAIN] <- 0
            | Locale.Host -> this.FillAndResetDrain(0,stream) <| Locale.Host

    member inline this.ResetFill() =
        fun (locale:Locale) ->
            match locale with
            | Locale.Device -> this.d_counters.[FILL] <- 0
            | Locale.Host -> ()

    member inline this.FillSize(fill_size:Offset ref, ?stream:CUstream) =
        let stream = if stream.IsSome then stream.Value else 0n
        fun (locale:Locale) ->
            match locale with
            | Locale.Device -> fill_size := this.d_counters.[FILL]
            | Locale.Host -> ()

    member inline this.Drain(num_items:Offset) = () //atomicAdd(d_counters + FILL, num_items)
    member inline this.Fill(num_items:Offset) = () //atomicAdd(d_counters + FILL, num_items)
    
    static member AllocationSize() = sizeof<Offset> * 2

    static member Create() =
        {
            d_counters = __null()
        }

    static member Create(d_storage:deviceptr<_>) = 
        {
            d_counters = d_storage
        } 


let FillAndResetDrainKernel =
    <@ fun (grid_queue:GridQueue) (num_items:Offset) ->
        grid_queue.FillAndResetDrain(num_items)
    @>