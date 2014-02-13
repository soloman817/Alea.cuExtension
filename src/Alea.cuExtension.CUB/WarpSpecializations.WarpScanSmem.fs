[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
    

/// The number of warp scan steps
let STEPS =
    fun logical_warp_threads ->
        logical_warp_threads |> log2

/// The number of threads in half a warp
let HALF_WARP_THREADS =
    fun logical_warp_threads ->
        let STEPS = logical_warp_threads |> STEPS
        1 <<< (STEPS - 1)

/// The number of shared memory elements per warp
let WARP_SMEM_ELEMENTS =
    fun logical_warp_threads ->
        logical_warp_threads + (logical_warp_threads |> HALF_WARP_THREADS)

let inline _TempStorage<'T>() =
    fun logical_warps warp_smem_elements ->
        __shared__.Array2D(logical_warps, warp_smem_elements)

[<Record>]
type ThreadFields<'T> =
    {
        temp_storage : 'T[,]
        warp_id : int
        lane_id : int
    }

    static member Init(temp_storage, warp_id, lane_id) =
        {
            temp_storage = temp_storage
            warp_id = warp_id
            lane_id = lane_id
        }

let initIdentity logical_warps logical_warp_threads =
    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
    fun (has_identity:bool) ->
        let temp_storage = (logical_warps, WARP_SMEM_ELEMENTS) ||> _TempStorage<'T>()
        let store = STORE_VOLATILE |> threadStore<'T>()
        fun warp_id lane_id ->
            match has_identity with
            | true ->
                let identity = ZeroInitialize<'T>() |> __ptr_to_obj
                (temp_storage.[warp_id, lane_id] |> __array_to_ptr, identity) ||> store
            | false ->
                ()

let scanStep logical_warps logical_warp_threads =
    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
    let STEPS = logical_warp_threads |> STEPS
    fun (temp_storage:'T[,]) warp_id lane_id ->
        let load = LOAD_VOLATILE |> threadLoad<'T>()
        let store = STORE_VOLATILE |> threadStore<'T>()
        fun has_identity step ->
            let step = ref 0
            fun (partial:Ref<'T>) (scan_op:('T -> 'T -> 'T)) ->
                while !step < STEPS do
                    let OFFSET = 1 <<< !step
                    
                    //(temp_storage |> __array_to_ptr, !partial) ||> store

                    if has_identity || (lane_id >= OFFSET) then
                        let addend = ((temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id)]), Some(partial |> __ref_to_ptr)) ||> load
                        partial := (addend.Value, !partial) ||> scan_op
                        
                    step := !step + 1


let broadcast =
    fun (temp_storage:deviceptr<'T>) warp_id lane_id ->
        let load = LOAD_VOLATILE |> threadLoad<'T>()
        let store = STORE_VOLATILE |> threadStore<'T>()
        fun input src_lane ->
            if lane_id = src_lane then (temp_storage.[warp_id], input) ||> store
            (temp_storage.[warp_id], None) ||> load
            |> Option.get


let inline basicScan logical_warps logical_warp_threads = 
    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
    let scanStep = (logical_warps, logical_warp_threads) ||> scanStep
    fun has_identity share_final ->
        fun (temp_storage:'T[,]) warp_id lane_id ->
            let store = STORE_VOLATILE |> threadStore<'T>()
            fun (partial:'T) (scan_op:('T -> 'T -> 'T)) ->
                let partial = partial |> __obj_to_ref
                scanStep
                <|||    (temp_storage, warp_id, lane_id)
                <||     (has_identity, 0)
                <||     (partial, scan_op)
                if share_final then (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id)], !partial) ||> store
                !partial


let inline inclusiveSum logical_warps logical_warp_threads =
    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
    let initIdentity = (logical_warps, logical_warp_threads) ||> initIdentity
    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan

    fun (temp_storage:'T[,]) warp_id lane_id ->
        let load = LOAD_VOLATILE |> threadLoad<'T>()
            
        fun (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T> option) ->
            match warp_aggregate with
            | None ->
                let HAS_IDENTITY = true // Traits<'T>::PRIMITIVE
                initIdentity
                <|  HAS_IDENTITY 
                <|| (warp_id, lane_id)
                
                output :=
                    basicScan
                    <||     (HAS_IDENTITY, false)
                    <|||    (temp_storage, warp_id, lane_id) 
                    <||     (input, ( + ))

            | Some warp_aggregate ->
                let HAS_IDENTITY = true // Traits<'T>::PRIMITIVE
                initIdentity
                <|  HAS_IDENTITY
                <|| (warp_id, lane_id)

                output :=
                    basicScan
                    <||     (HAS_IDENTITY, true)
                    <|||    (temp_storage, warp_id, lane_id)
                    <||     (input, ( + ))

                warp_aggregate :=
                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)], None) 
                    ||> load
                    |> Option.get


let inline inclusiveScan logical_warps logical_warp_threads =
    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan

    fun (temp_storage:'T[,]) warp_id lane_id ->
        let load = LOAD_VOLATILE |> threadLoad<'T>()

        fun (input:'T) (output:Ref<'T>) (scan_op:('T -> 'T -> 'T)) (warp_aggregate:Ref<'T> option) ->
            match warp_aggregate with
            | None ->
                output :=
                    basicScan
                    <||     (false, false)
                    <|||    (temp_storage, warp_id, lane_id)
                    <||     (input, scan_op)

            | Some warp_aggregate ->
                output :=
                    basicScan
                    <||     (false, true)
                    <|||    (temp_storage, warp_id, lane_id)
                    <||     (input, scan_op)

                warp_aggregate :=
                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)], None) 
                    ||> load
                    |> Option.get 

    
let inline exclusiveScan logical_warps logical_warp_threads =
    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan

    fun (temp_storage:'T[,]) warp_id lane_id ->
        let load = LOAD_VOLATILE |> threadLoad<'T>()
        let store = STORE_VOLATILE |> threadStore<'T>()

        fun (input:'T) (output:Ref<'T>) (scan_op:('T -> 'T -> 'T)) (identity:'T option) (warp_aggregate:Ref<'T> option) ->
            match identity, warp_aggregate with
            | Some identity, None ->
                (temp_storage.[warp_id, lane_id], identity) ||> store
                let inclusive =
                    basicScan
                    <||     (true, true)
                    <|||    (temp_storage, warp_id, lane_id)
                    <||     (input, scan_op)

                output :=
                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)], None) 
                    ||> load
                    |> Option.get

            | Some identity, Some warp_aggregate ->
                (temp_storage.[warp_id, lane_id], identity) ||> store
                let inclusive =
                    basicScan
                    <||     (true, true)
                    <|||    (temp_storage, warp_id, lane_id)
                    <||     (input, scan_op)

                output :=
                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)], None) 
                    ||> load
                    |> Option.get

                warp_aggregate :=
                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)], None)
                    ||> load
                    |> Option.get

            | None, None ->
                let inclusive =
                    basicScan
                    <||     (false, true)
                    <|||    (temp_storage, warp_id, lane_id)
                    <||     (input, scan_op)

                output :=
                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)], None)
                    ||> load
                    |> Option.get

            | None, Some warp_aggregate ->
                let inclusive =
                    basicScan
                    <||     (false, true)
                    <|||    (temp_storage, warp_id, lane_id)
                    <||     (input, scan_op)

                output :=
                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)], None)
                    ||> load
                    |> Option.get

                warp_aggregate :=
                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)], None)
                    ||> load
                    |> Option.get
                    
            
[<Record>]
type WarpScanSmem<'T> =
    {
        temp_storage : deviceptr<'T>
        warp_id : uint32
        lane_id : uint32
    }

        
    member this.ScanStep(partial:Ref<'T>) =
        fun has_identity step -> ()

    member this.ScanStep(partial:Ref<'T>, scan_op:('T -> 'T -> 'T), step:bool) = 
        fun has_identity step -> ()

    member this.Broadcast(input:'T, src_lane:uint32) = ()
        
    member this.BasicScan(partial:'T, scan_op:('T -> 'T -> 'T)) = ()

    member this.InclusiveSum(input:'T, output:Ref<'T>) = ()
    member this.InclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>) = ()

    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), warp_aggregate:Ref<'T>) = ()
    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T)) = ()

    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T)) = ()
    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T), warp_aggregate:Ref<'T>) = ()