[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.Specializations

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

open Alea.cuExtension
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
    
module ReduceShfl =
    let f() = "reduce shfl"

module ReduceSmem =
    let f() = "reduce smem"

module ScanShfl =
    
    let STEPS = 
        fun logical_warp_threads ->
            logical_warp_threads |> log2

    let SHFL_C =
        fun logical_warp_threads ->
            let STEPS = logical_warp_threads |> STEPS
            ((-1 <<< STEPS) &&& 31) <<< 8

    let broadcast =
        fun logical_warp_threads ->
            fun input src_lane ->
                (input, src_lane, logical_warp_threads) |||> ShuffleBroadcast

    
    let inclusiveScan w x y z = ()

    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
    type InclusiveSumPtxAttribute() =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, irParams with
                | None, temp :: shlStep :: shfl_c :: [] ->
                    let clrType = info.GetGenericArguments().[0]
                    let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
                    let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<uint32 -> int -> int-> uint32>)
                    let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                    IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                        "{
                            .reg .u32 r0;
                            .reg .pred p;
                            shfl.up.b32 r0|p, $1, $2, $3;
                            @p add.u32 r0, r0, %4;
                            mov.u32 %0, r0;
                        }", "=r,r,r,r,r", temp :: shlStep :: shfl_c :: []) |> Some
                | _ -> None

    let [<InclusiveSumPtx>] inclusiveSumPtx (temp:uint32) (shlStep:int) (shfl_c:int) : uint32 = failwith ""
    let inclusiveSum logical_warps logical_warp_threads =
        let STEPS = logical_warp_threads |> STEPS
        let SHFL_C = logical_warp_threads |> SHFL_C
        let broadcast = logical_warp_threads |> broadcast

        fun (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (single_shfl:bool option) ->
            match single_shfl with
            | Some single_shfl ->
                if single_shfl then
                    let temp = ref input
                    for STEP = 0 to (STEPS - 1) do
                        temp := (!temp, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx
                    output := !temp

                    warp_aggregate := (!output, (logical_warp_threads - 1)) ||> broadcast
//                else
//                    (input, output, ( + ), warp_aggregate) |||> inclusiveScan
            


    [<Record>]
    type ThreadFields<'T> =
        {
            warp_id : int
            lane_id : int
        }

        static member Create(warp_id, lane_id) =
            {
                warp_id = warp_id
                lane_id = lane_id
            }


    [<Record>]
    type WarpScanShfl =
        {
            LOGICAL_WARPS : int
            LOGICAL_WARP_THREADS : int
        }

        static member Create(logical_warps, logical_warp_threads) =
            {
                LOGICAL_WARPS = logical_warps
                LOGICAL_WARP_THREADS = logical_warp_threads
            }

module ScanSmem =
    
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
                        (temp_storage |> __array2d_to_ptr, !partial) ||> store

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
            ThreadFields : ThreadFields<'T>
        }

        static member Create(temp_storage, warp_id, lane_id) =
            {
                ThreadFields = ThreadFields<'T>.Init(temp_storage, warp_id, lane_id)
            }

        member this.ScanStep(partial, scan_op, step) = ()
        member this.ScanStep() = ()