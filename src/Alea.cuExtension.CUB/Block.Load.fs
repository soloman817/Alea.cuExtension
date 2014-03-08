﻿[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Load
    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Vector

type BlockLoadAlgorithm =
    | BLOCK_LOAD_DIRECT         = 0  
    | BLOCK_LOAD_VECTORIZE      = 1
    | BLOCK_LOAD_TRANSPOSE      = 2
    | BLOCK_LOAD_WARP_TRANSPOSE = 3

module Template =
    module Host =
        module Params =
            [<Record>]
            type API =
                {
                    BLOCK_THREADS       : int
                    ITEMS_PER_THREAD    : int
                    ALGORITHM           : BlockLoadAlgorithm
                    WARP_TIME_SLICING   : bool
                }

                static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
                    {
                        BLOCK_THREADS       = block_threads
                        ITEMS_PER_THREAD    = items_per_thread
                        ALGORITHM           = algorithm
                        WARP_TIME_SLICING   = warp_time_slicing
                    }

        [<Record>]
        type API =
            {
                Params              : Params.API
                SharedMemoryLength  : int 
            }

            static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
                let p = Params.API.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
                let sml = Exchange.Template.Host.API.Init(block_threads, items_per_thread, warp_time_slicing).SharedMemoryLength
                {
                    Params              = p
                    SharedMemoryLength  = sml
                }

    module Device =
        module TempStorage =
            type [<Record>] API<'T> = Alea.cuExtension.CUB.Block.Exchange.Template._TempStorage<'T>

        module ThreadFields =
            
            [<Record>]
            type API<'T> =
                {
                    temp_storage : TempStorage.API<'T>
                }

                [<ReflectedDefinition>] static member Init(temp_storage:TempStorage.API<'T>) = { temp_storage = temp_storage }
                [<ReflectedDefinition>] static member Init(length) = { temp_storage = TempStorage.API<'T>.Uninitialized(length) }
                
        [<Record>]
        type API<'T> =
            {
                mutable Params          : Host.Params.API
                mutable ThreadFields    : ThreadFields.API<'T>
            }

            [<ReflectedDefinition>]
            static member Init(h:Host.API) : API<'T> =
                let f = ThreadFields.API<'T>.Init(h.SharedMemoryLength)
                let p = h.Params                
                { Params = p; ThreadFields = f}

            
    //type _TemplateParams    = Host.Params.API
    type [<Record>] _TempStorage<'T>   = Device.TempStorage.API<'T>
    type [<Record>] _ThreadFields<'T>  = Device.ThreadFields.API<'T>

    [<Record>]
    type API<'T> =
        {
            Host    : Host.API
        }

        static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
            let h = Host.API.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
            {
                Host    = h
            }

        [<ReflectedDefinition>] member inline this.DeviceAPI = Device.API<'T>.Init(this.Host)
    

type [<Record>] _Template<'T> = Template.API<'T>


module LoadDirectBlocked =
//    type _Template<'T> = Template.Device.API<'T>

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = template.Host.Params
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]
        
    let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = template.Host.Params
        let bounds = valid_items - (linear_tid * p.ITEMS_PER_THREAD)
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]
        
    let [<ReflectedDefinition>] inline GuardedWithOOB (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
        let p = template.Host.Params
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
        let bounds = valid_items - (linear_tid * p.ITEMS_PER_THREAD)
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]


module LoadDirectBlockedVectorized =
    //type _Template<'T> = Template.Device.API<'T>

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = template.Host.Params
        let MAX_VEC_SIZE = CUB_MIN 4 p.ITEMS_PER_THREAD
        let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((p.ITEMS_PER_THREAD % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
        let VECTORS_PER_THREAD = p.ITEMS_PER_THREAD / VEC_SIZE
        let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret

        let vec_items = __local__.Array<'T>(VECTORS_PER_THREAD) |> __array_to_ptr

        for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM] //|> __ptr_to_obj
        

    let private Guarded x = <@ fun x y z w -> () @>
    let private GuardedWithOOB x = <@ fun x y z w u -> () @>


module LoadDirectStriped =
    //type _Template<'T> = Template.Device.API<'T>

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = template.Host.Params
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]

    let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = template.Host.Params
        let bounds = valid_items - linear_tid
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
            if (ITEM * p.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]

    let [<ReflectedDefinition>] inline GuardedWithOOB (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
        let p = template.Host.Params
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
        let bounds = valid_items - linear_tid
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
            if (ITEM * p.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]

    
module LoadDirectWarpStriped =
    //type _Template<'T> = Template.Device.API<'T>

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = template.Host.Params
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]

    let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = template.Host.Params
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
        let bounds = valid_items - warp_offset - tid

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
            if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
        
    let [<ReflectedDefinition>] inline GuardedWithOOB (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
        let p = template.Host.Params
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
        let bounds = valid_items - warp_offset - tid

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
            if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]


module LoadInternal =
    //type _Template<'T> = Template.Device.API<'T>
    
    module BlockLoadDirect =
        let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
            (linear_tid:int) 
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            LoadDirectBlocked.Default template linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
            (linear_tid:int) 
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            LoadDirectBlocked.Guarded template linear_tid block_ptr items valid_items
            
        let [<ReflectedDefinition>] inline GuardedWithOOB (template:_Template<'T>) 
            (linear_tid:int) 
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            LoadDirectBlocked.GuardedWithOOB template linear_tid block_ptr items valid_items oob_default            

    module BlockLoadVectorized =
        let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
            (linear_tid:int) 
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            LoadDirectBlockedVectorized.Default template linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
            (linear_tid:int) 
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            LoadDirectBlocked.Guarded template linear_tid block_ptr items valid_items
            
        let [<ReflectedDefinition>] inline GuardedWithOOB (template:_Template<'T>)
            (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            LoadDirectBlocked.GuardedWithOOB template linear_tid block_ptr items valid_items oob_default
            

    module BlockLoadTranspose =

        let [<ReflectedDefinition>] inline StripedToBlocked (template:_Template<'T>)
            (linear_tid:int) =
            let p = template.DeviceAPI.Params
            let f = template.DeviceAPI.ThreadFields
            if p.WARP_TIME_SLICING then 
                BlockExchange.API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING, f.temp_storage).StripedToBlocked.WithTimeslicing
            else 
                BlockExchange.API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING, f.temp_storage).StripedToBlocked.Default
                
        let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                LoadDirectStriped.Default template linear_tid block_ptr items
                StripedToBlocked template linear_tid items
                
        let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                LoadDirectStriped.Guarded template linear_tid block_ptr items valid_items
                StripedToBlocked template linear_tid items

        let [<ReflectedDefinition>] inline GuardedWithOOB (template:_Template<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                LoadDirectStriped.GuardedWithOOB template linear_tid block_ptr items valid_items oob_default
                StripedToBlocked template linear_tid items


    module BlockLoadWarpTranspose =
        let [<ReflectedDefinition>] inline WARP_THREADS block_threads =
            ((block_threads % CUB_PTX_WARP_THREADS) = 0) |> function
            | false -> failwith "BLOCK_THREADS must be a multiple of WARP_THREADS"
            | true -> CUB_PTX_WARP_THREADS

        let [<ReflectedDefinition>] inline WarpStripedToBlocked (template:_Template<'T>)
            (linear_tid:int) =
            let p = template.DeviceAPI.Params
            let f = template.DeviceAPI.ThreadFields
            if p.WARP_TIME_SLICING then
                BlockExchange.API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING, f.temp_storage).WarpStripedToBlocked.WithTimeslicing
            else
                BlockExchange.API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING, f.temp_storage).WarpStripedToBlocked.Default

        let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                LoadDirectWarpStriped.Default template linear_tid block_ptr items
                WarpStripedToBlocked template linear_tid items
                
        let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                LoadDirectWarpStriped.Guarded template linear_tid block_ptr items valid_items
                WarpStripedToBlocked template linear_tid items

        let [<ReflectedDefinition>] inline GuardedWithOOB (template:_Template<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                LoadDirectWarpStriped.GuardedWithOOB template linear_tid block_ptr items valid_items oob_default
                WarpStripedToBlocked template linear_tid items

    [<Record>]
    type API<'T> =
        {
            Default         : _Template<'T> -> int -> deviceptr<'T> -> deviceptr<'T> -> unit
            Guarded         : _Template<'T> -> int -> deviceptr<'T> -> deviceptr<'T> -> int -> unit
            GuardedWithOOB  : _Template<'T> -> int -> deviceptr<'T> -> deviceptr<'T> -> int -> 'T -> unit
        }

//    [<Record>]
//    type API<'T> =
//        {
//            Default         : _Template<'T> -> int -> deviceptr<'T> -> deviceptr<'T> -> unit
//            Guarded         : _Template<'T> -> int -> deviceptr<'T> -> deviceptr<'T> -> int -> unit
//            GuardedWithOOB  : _Template<'T> -> int -> deviceptr<'T> -> deviceptr<'T> -> int -> 'T -> unit
//        }

        [<ReflectedDefinition>]
        static member Init(algorithm:BlockLoadAlgorithm) =
            let _Default = algorithm |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.Default
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.Default
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.Default
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.Default
                | _ -> failwith ""
            let _Guarded = algorithm |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.Guarded 
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.Guarded
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.Guarded
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.Guarded
                | _ -> failwith ""
            let _GuardedWithOOB = algorithm |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.GuardedWithOOB
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.GuardedWithOOB
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.GuardedWithOOB
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.GuardedWithOOB
                | _ -> failwith ""
            { Default = _Default; Guarded = _Guarded; GuardedWithOOB = _GuardedWithOOB }

        

module BlockLoad =

    type API<'T> =
        {
            Default         : Function<int -> deviceptr<'T> -> deviceptr<'T> -> unit>
            Guarded         : Function<int -> deviceptr<'T> -> deviceptr<'T> -> int -> unit>
            GuardedWithOOB  : Function<int -> deviceptr<'T> -> deviceptr<'T> -> int -> 'T -> unit>
        }

//    [<Record>]
//    type API<'T> =
//        {
//            template    : Template.API<'T>
//            Load        : LoadInternal.API<'T>
//        }
//
//
//        static member Init(block_threads:int, items_per_thread:int, algorithm:BlockLoadAlgorithm, warp_time_slicing:bool) =
//            let LoadInternal = LoadInternal.API<'T>.Init(algorithm)
//            let h = Template.API<'T>.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
//            { template = h; Load = LoadInternal }
//
//        static member Init(block_threads, items_per_thread) = API<'T>.Init(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT, false)
//
//        [<ReflectedDefinition>] member inline this.Default = this.Load.Default this.template
//            let _Default =
//                    let template = _Template<'T>.Init(h, algorithm)
//                    (%Default) template linear_tid block_ptr items
//                
//        
//            let! _Guarded =
//                let Guarded = LoadInternal.Guarded
//                <@ fun (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) ->
//                    let template = _Template<'T>.Init(h, algorithm)
//                    (%Guarded) template linear_tid block_ptr items valid_items
//                @> |> Compiler.DefineFunction
//        
//            let! _GuardedWithOOB =
//                let GuardedWithOOB = LoadInternal.GuardedWithOOB
//                <@ fun (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) ->
//                    let template = _Template<'T>.Init(h, algorithm)
//                    (%GuardedWithOOB) template linear_tid block_ptr items valid_items oob_default
//                @> |> Compiler.DefineFunction
//        
//            
//                { 
//                    Default         = _Default
//                    Guarded         = _Guarded
//                    GuardedWithOOB  = _GuardedWithOOB
//                }

    let inline template<'T> (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) = cuda {
        let LoadInternal = LoadInternal.API<'T>.Init(algorithm)
        let template = Template.API<'T>.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
        let! _Default =
            let Default = LoadInternal.Default
            <@ fun (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) ->
                Default template linear_tid block_ptr items
            @> |> Compiler.DefineFunction
        
        let! _Guarded =
            let Guarded = LoadInternal.Guarded
            <@ fun (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) ->
                Guarded template linear_tid block_ptr items valid_items
            @> |> Compiler.DefineFunction
        
        let! _GuardedWithOOB =
            let GuardedWithOOB = LoadInternal.GuardedWithOOB
            <@ fun (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) ->
                GuardedWithOOB template linear_tid block_ptr items valid_items oob_default
            @> |> Compiler.DefineFunction
        
        return
            { 
                Default         = _Default
                Guarded         = _Guarded
                GuardedWithOOB  = _GuardedWithOOB
            }
        }


//    let [<ReflectedDefinition>] api block_threads items_per_thread algorithm (warp_time_slicing:bool option) =
//        let warp_time_slicing = if warp_time_slicing.IsNone then false else warp_time_slicing.Value
//        fun temp_storage linear_tid ->
//            let _Default, _Guarded, _GuardedWithOOB =
//                algorithm |> function
//                | BLOCK_LOAD_DIRECT -> 
//                    (   
//                        LoadInternal.BlockLoadDirect.Default
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),
//                        LoadInternal.BlockLoadDirect.Guarded
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),
//                        LoadInternal.BlockLoadDirect.GuardedWithOOB
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid)
//                    )
//
//                | BLOCK_LOAD_VECTORIZE ->
//                    (   
//                        LoadInternal.BlockLoadVectorized.Default
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                        
//                        LoadInternal.BlockLoadVectorized.Guarded
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                        
//                        LoadInternal.BlockLoadVectorized.GuardedWithOOB
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid)
//                    )
//
//                | BLOCK_LOAD_TRANSPOSE ->
//                    (   
//                        LoadInternal.BlockLoadTranspose.Default
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                        
//                        LoadInternal.BlockLoadTranspose.Guarded
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                        
//                        LoadInternal.BlockLoadTranspose.GuardedWithOOB
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid)
//                    )
//
//                | BLOCK_LOAD_WARP_TRANSPOSE ->
//                    (   
//                        LoadInternal.BlockLoadWarpTranspose.Default
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                    
//                        LoadInternal.BlockLoadWarpTranspose.Guarded
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                    
//                        LoadInternal.BlockLoadWarpTranspose.GuardedWithOOB
//                        <|||    (block_threads, items_per_thread, warp_time_slicing)
//                        <||     (temp_storage, linear_tid)                    
//                    )
//
//            {Default = _Default; Guarded = _Guarded; GuardedWithOOB = _GuardedWithOOB}
                
        
//    let api block_threads items_per_thread warp_time_slicing =
//        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
//            {
//                BlockLoadDirect         =   BlockLoadDirect.api
//                                            <||| (block_threads, items_per_thread, warp_time_slicing)
//
//                BlockLoadVectorized     =   BlockLoadVectorized.api
//                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
//
//                BlockLoadTranspose      =   BlockLoadTranspose.api
//                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
//                                            <||     (temp_storage, linear_tid)
//                                            <|||    (warp_lane, warp_id, warp_offset)
//
//                BlockLoadWarpTranspose  =   BlockLoadWarpTranspose.api
//                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
//                                            <||     (temp_storage, linear_tid)
//                                            <|||    (warp_lane, warp_id, warp_offset)
//            }                    

//
//module BlockLoad =
//    open Internal
//
//    type API =
//        {
//            BlockLoadDirect         : LoadInternal.BlockLoadDirect.API
//            BlockLoadVectorized     : LoadInternal.BlockLoadVectorized.API
//            BlockLoadTranspose      : LoadInternal.BlockLoadTranspose.API
//            BlockLoadWarpTranspose  : LoadInternal.BlockLoadWarpTranspose.API
//        }
//
//    let private (|BlockLoadDirect|_|) threadFields templateParams =
//        let block_threads, items_per_thread, algorithm, warp_time_slicing = templateParams
//        let temp_storage, linear_tid, warp_lane, warp_id, warp_offset = threadFields
//        if algorithm = BlockLoadDirect then
//            (   LoadInternal.api
//                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                <||     (temp_storage, linear_tid)
//                <|||    (warp_lane, warp_id, warp_offset)
//            ).BlockLoadDirect
//            |>      Some
//        else
//            None
//
//    let private (|BlockLoadVectorized|_|) threadFields templateParams =
//        let block_threads, items_per_thread, algorithm, warp_time_slicing = templateParams
//        let temp_storage, linear_tid, warp_lane, warp_id, warp_offset = threadFields
//        if algorithm = BlockLoadVectorized then
//            (   LoadInternal.api
//                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                <||     (temp_storage, linear_tid)
//                <|||    (warp_lane, warp_id, warp_offset)
//            ).BlockLoadVectorized
//            |>      Some
//        else
//            None
//
//    let private (|BlockLoadTranspose|_|) threadFields templateParams =
//        let block_threads, items_per_thread, algorithm, warp_time_slicing = templateParams
//        let temp_storage, linear_tid, warp_lane, warp_id, warp_offset = threadFields
//        if algorithm = BlockLoadTranspose then
//            (   LoadInternal.api
//                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                <||     (temp_storage, linear_tid)
//                <|||    (warp_lane, warp_id, warp_offset)
//            ).BlockLoadTranspose
//            |>      Some
//        else
//            None
//
//    let private (|BlockLoadWarpTranspose|_|) threadFields templateParams =
//        let block_threads, items_per_thread, algorithm, warp_time_slicing = templateParams
//        let temp_storage, linear_tid, warp_lane, warp_id, warp_offset = threadFields
//        if algorithm = BlockLoadWarpTranspose then
//            (   LoadInternal.api
//                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                <||     (temp_storage, linear_tid)
//                <|||    (warp_lane, warp_id, warp_offset)
//            ).BlockLoadWarpTranspose
//            |>      Some
//        else
//            None
//
//
//    let private Default block_threads items_per_thread algorithm warp_time_slicing = 
//        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
//            let templateParams = (block_threads, items_per_thread, algorithm, warp_time_slicing)
//            let threadFields = (temp_storage, linear_tid, warp_lane, warp_id, warp_offset)
//            let InternalLoad =
//                templateParams |> function
//                | BlockLoadDirect threadFields bld ->
//                    bld.LoadDirectBlocked.Default
//                | BlockLoadVectorized threadFields blv ->
//                    blv.LoadDirectBlockedVectorized.Default
//                | BlockLoadTranspose threadFields blt ->
//                    blt.LoadDirectStriped.Default
//                | BlockLoadWarpTranspose threadFields blwt ->
//                    blwt.LoadDirectWarpStriped.Default
//                | _ -> failwith "Invalid Template Parameters"
//
//
//
//
//            <@ fun _ -> () @>
//
//    let private Guarded block_threads items_per_thread algorithm warp_time_slicing =
//        <@ fun _ -> () @>
//
//    let private GuardedWithOOB block_threads items_per_thread algorithm warp_time_slicing =
//        <@ fun _ -> () @>

//module LoadInternal =
//    open Internal
//
//    module BlockLoadDirect =
//        type API =
//            {
//                LoadDirectBlocked : LoadDirectBlocked.API
//            }
//
//        let api _ items_per_thread _ = 
//            {
//                LoadDirectBlocked   =   LoadDirectBlocked.api
//                                        <|| (None, items_per_thread)
//            }
//
//    module BlockLoadVectorize =
//        type API =
//            {
//                LoadDirectBlockedVectorized : LoadDirectBlockedVectorized.API
//            }
//
//        let api _ items_per_thread _ =
//            {
//                LoadDirectBlockedVectorized =   LoadDirectBlockedVectorized.api
//                                                <|| (None, items_per_thread)
//            }
//
//    module BlockLoadTranspose =
//        type API =
//            {
//                LoadDirectStriped   : LoadDirectStriped.API
//                BlockExchange       : BlockExchange.API
//            }
//
//        let api block_threads items_per_thread warp_time_slicing = 
//            fun temp_storage linear_tid warp_lane warp_id warp_offset ->
//                {
//                    LoadDirectStriped   =   LoadDirectStriped.api
//                                            <|| (block_threads, items_per_thread)
//
//                    BlockExchange       =   BlockExchange.api
//                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
//                                            <||     (temp_storage, linear_tid)
//                                            <|||    (warp_lane, warp_id, warp_offset)
//                }
//
//    module BlockLoadWarpTranspose =
//        type API =
//            {
//                LoadDirectWarpStriped   : LoadDirectWarpStriped.API
//                BlockExchange           : BlockExchange.API                
//            }
//
//        let api block_threads items_per_thread warp_time_slicing = 
//            fun temp_storage linear_tid warp_lane warp_id warp_offset ->
//                {
//                    LoadDirectWarpStriped   =   LoadDirectWarpStriped.api
//                                                <|| (block_threads, items_per_thread)
//
//                    BlockExchange           =   BlockExchange.api
//                                                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                                                <||     (temp_storage, linear_tid)
//                                                <|||    (warp_lane, warp_id, warp_offset)
//                }
//                 
    //let load (block_threads:int) (items_per_thread:int) =
//    let private internalAPI = 
//            {   LoadDirectBlocked           = LoadDirectBlocked.api;
//                LoadDirectBlockedVectorized = LoadDirectBlockedVectorized.api
//                LoadDirectStriped           = LoadDirectStriped.api
//                LoadDirectWarpStriped       = LoadDirectWarpStriped.api }

//    let api block_threads items_per_thread algorithm = 
//            algorithm |> function
//            | BLOCK_LOAD_DIRECT ->           (block_threads, items_per_thread) ||> internalAPI.LoadDirectBlocked
//            | BLOCK_LOAD_VECTORIZE ->        (block_threads, items_per_thread) ||> internalAPI.LoadDirectBlockedVectorized
//            | BLOCK_LOAD_TRANSPOSE ->        (block_threads, items_per_thread) ||> internalAPI.LoadDirectStriped
//            | BLOCK_LOAD_WARP_TRANSPOSE ->   (block_threads, items_per_thread) ||> internalAPI.LoadDirectWarpStriped
//
//    let api (block_threads:int) (items_per_thread:int) = 
//            {   LoadDirectBlocked           = (block_threads, items_per_thread) ||> LoadDirectBlocked.api;
//                LoadDirectBlockedVectorized = (block_threads, items_per_thread) ||> LoadDirectBlockedVectorized.api
//                LoadDirectStriped           = (block_threads, items_per_thread) ||> LoadDirectStriped.api
//                LoadDirectWarpStriped       = (block_threads, items_per_thread) ||> LoadDirectWarpStriped.api }


//    let load (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) = //cuda {//(valid_items:int option) (oob_default:int option) =
//        let api = (block_threads, items_per_thread) ||> api
//
//        fun (valid_items:int option) (oob_default:int option) -> 
//            let Option = (valid_items, oob_default)
//            let (|DefaultOption|GuardedOption|GuardedWithOOBOption|) x = 
//                x |> function
//                | (None, None) -> DefaultOption
//                | (Some valid_items, None) -> GuardedOption
//                | (Some valid_items, Some oob_default) -> GuardedWithOOBOption
//                | _,_ -> DefaultOption
//
//            (algorithm, Option) |> function
//            | BLOCK_LOAD_DIRECT, DefaultOption ->           Load.Default(api.LoadDirectBlocked.Default)
//            | BLOCK_LOAD_VECTORIZE, DefaultOption ->        Load.Default(api.LoadDirectBlockedVectorized.Default)
//            | BLOCK_LOAD_TRANSPOSE, DefaultOption ->        Load.Default(api.LoadDirectStriped.Default)
//            | BLOCK_LOAD_WARP_TRANSPOSE, DefaultOption ->   Load.Default(api.LoadDirectWarpStriped.Default)
//
//            | BLOCK_LOAD_DIRECT, GuardedOption ->           Load.Guarded(api.LoadDirectBlocked.Guarded)
//            | BLOCK_LOAD_VECTORIZE, GuardedOption ->        Load.Guarded(api.LoadDirectBlockedVectorized.Guarded)
//            | BLOCK_LOAD_TRANSPOSE, GuardedOption ->        Load.Guarded(api.LoadDirectStriped.Guarded)
//            | BLOCK_LOAD_WARP_TRANSPOSE, GuardedOption ->   Load.Guarded(api.LoadDirectWarpStriped.Guarded)
//
//            | BLOCK_LOAD_DIRECT, GuardedWithOOBOption ->           Load.GuardedWithOOB(api.LoadDirectBlocked.GuardedWithOOB)
//            | BLOCK_LOAD_VECTORIZE, GuardedWithOOBOption ->        Load.GuardedWithOOB(api.LoadDirectBlockedVectorized.GuardedWithOOB)
//            | BLOCK_LOAD_TRANSPOSE, GuardedWithOOBOption ->        Load.GuardedWithOOB(api.LoadDirectStriped.GuardedWithOOB)
//            | BLOCK_LOAD_WARP_TRANSPOSE, GuardedWithOOBOption ->   Load.GuardedWithOOB(api.LoadDirectWarpStriped.GuardedWithOOB)
//    let load (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) = //cuda {//(valid_items:int option) (oob_default:int option) =
//        let api = (block_threads, items_per_thread) ||> api
//        algorithm |> function
//        | BLOCK_LOAD_DIRECT ->          api.LoadDirectBlocked
//        | BLOCK_LOAD_VECTORIZE ->       api.LoadDirectBlockedVectorized
//        | BLOCK_LOAD_TRANSPOSE ->       api.LoadDirectStriped
//        | BLOCK_LOAD_WARP_TRANSPOSE ->  api.LoadDirectWarpStriped
//
//
//let inline blockLoad (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) = 
//    let loadInternal = (block_threads, items_per_thread, algorithm) |||> InternalLoad.load
//    { new BlockLoadAPI with
//        member this.Default =           cuda { return! loadInternal.Default         |> Compiler.DefineFunction}
//        member this.Guarded =           cuda { return! loadInternal.Guarded         |> Compiler.DefineFunction}
//        member this.GuardedWithOOB =    cuda { return! loadInternal.GuardedWithOOB  |> Compiler.DefineFunction}
//    }
//    
//
////let blockLoad (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) = cuda {
//    //let loadInternal = (block_threads, items_per_thread, algorithm) |||> InternalLoad.load
////        algorithm |> function
////        | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
////            ()
////
////        | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->
////            ()
////
////        | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->
////            let stripedToBlocked = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.stripedToBlocked
////            ()
////
////        | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->
////            let warpStripedToBlocked = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.warpStripedToBlocked
////            ()
//        //}
////let PrivateStorage() = __null()
////
////[<Record>]
////type ThreadFields =
////    {
////        mutable temp_storage : deviceptr<int>
////        mutable linear_tid : int
////    }
////
////    [<ReflectedDefinition>]
////    member this.Get() = (this.temp_storage, this.linear_tid)
////    
////    [<ReflectedDefinition>]
////    static member Init(temp_storage:deviceptr<int>, linear_tid:int) =
////        {
////            temp_storage = temp_storage
////            linear_tid = linear_tid
////        }
////
////    [<ReflectedDefinition>]
////    static member Default() =
////        {
////            temp_storage = __null()
////            linear_tid = 0
////        }
//
//
//
//[<Record>]
//type BlockLoad =
//    {
//        BLOCK_THREADS       : int
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] ALGORITHM           : BlockLoadAlgorithm
//        WARP_TIME_SLICING   : bool
//        ThreadFields        : ThreadFields
//    }
//
//
//    [<ReflectedDefinition>]
//    member this.Initialize() =
//        this.ThreadFields.temp_storage <- PrivateStorage()
//        this.ThreadFields.linear_tid <- threadIdx.x
//        this
//    
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<int>) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- threadIdx.x
//        this
//    
//    [<ReflectedDefinition>]
//    member this.Initialize(linear_tid:int) =
//        this.ThreadFields.temp_storage <- PrivateStorage()
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid:int) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Load(block_itr:deviceptr<int>, items:deviceptr<int>) = 
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (None, None)
//
//    [<ReflectedDefinition>]
//    member this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (Some valid_items, None)
//
//    [<ReflectedDefinition>]
//    member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|| (Some valid_items, Some oob_default)
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread, algorithm, warp_time_slicing) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = warp_time_slicing
//            ThreadFields        = ThreadFields.Default()
//        }
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread, algorithm) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread) = 
//        {
//            
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = BlockLoadAlgorithm.BLOCK_LOAD_DIRECT
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }


//[<ReflectedDefinition>]
//type BlockLoadAlgorithm =
//    | BLOCK_LOAD_DIRECT
//    | BLOCK_LOAD_VECTORIZE
//    | BLOCK_LOAD_TRANSPOSE
//    | BLOCK_LOAD_WARP_TRANSPOSE
//
//
//let loadDirectBlocked (block_threads:int) (items_per_thread:int) = 
//    fun (valid_items:int option) (oob_default:int option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                let bounds = valid_items - (linear_tid * items_per_thread)
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
//                let bounds = valid_items - (linear_tid * items_per_thread)
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//
//        | _, _ ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) -> ()
//
//
//let loadDirectBlockedVectorized (block_threads:int) (items_per_thread:int) =
//    fun _ _ ->
//        fun (linear_tid:int) (block_ptr:deviceptr<'Vector>) (items:deviceptr<'Vector>) ->
//            let MAX_VEC_SIZE = CUB_MIN 4 items_per_thread
//            let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((items_per_thread % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
//            let VECTORS_PER_THREAD = items_per_thread / VEC_SIZE
//            let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret
//
//            let vec_items = __local__.Array<'Vector>(VECTORS_PER_THREAD) |> __array_to_ptr
//
//            for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
//            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- vec_items.[ITEM] //|> __ptr_to_obj
//
//
//let loadDirectStriped (block_threads:int) (items_per_thread:int) = 
//    fun (valid_items:int option) (oob_default:int option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//               for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                let bounds = valid_items - linear_tid
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if (ITEM * block_threads < bounds) then items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
//                let bounds = valid_items - linear_tid
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if (ITEM * block_threads < bounds) then items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//
//        | _, _ ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) -> ()
//
//
//let loadDirectWarpStriped (block_threads:int) (items_per_thread:int) = 
//    fun (valid_items:int option) (oob_default:int option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//                let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//                let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//                let bounds = valid_items - warp_offset - tid
//
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
//                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//                let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//                let bounds = valid_items - warp_offset - tid
//
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//        | _, _ ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) -> ()
//
//
//let loadInternal (algorithm:BlockLoadAlgorithm) =
//    fun (block_threads:int) (items_per_thread:int) ->
//        algorithm |> function
//        | BLOCK_LOAD_DIRECT ->           (block_threads, items_per_thread) ||> loadDirectBlocked
//        | BLOCK_LOAD_VECTORIZE ->        (block_threads, items_per_thread) ||> loadDirectBlockedVectorized
//        | BLOCK_LOAD_TRANSPOSE ->        (block_threads, items_per_thread) ||> loadDirectStriped
//        | BLOCK_LOAD_WARP_TRANSPOSE ->   (block_threads, items_per_thread) ||> loadDirectWarpStriped
//
//
//let blockLoad (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) =
//    algorithm |> function
//    | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
//        fun _ linear_tid ->
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//
//    | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->
//        fun _ linear_tid ->    
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//
//    | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->
//        let stripedToBlocked = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.stripedToBlocked
//        
//        fun temp_storage linear_tid ->
//            let stripedToBlocked = (temp_storage, linear_tid) ||> stripedToBlocked
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//                items |> stripedToBlocked
//
//    | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->
//        let warpStripedToBlocked = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.warpStripedToBlocked
//        
//        fun temp_storage linear_tid ->
//            let warpStripedToBlocked = (temp_storage, linear_tid) ||> warpStripedToBlocked
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//                items |> warpStripedToBlocked
//
//let PrivateStorage() = __null()
//
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage : deviceptr<int>
//        mutable linear_tid : int
//    }
//
//    [<ReflectedDefinition>]
//    member this.Get() = (this.temp_storage, this.linear_tid)
//    
//    [<ReflectedDefinition>]
//    static member Init(temp_storage:deviceptr<int>, linear_tid:int) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//        }
//
//    [<ReflectedDefinition>]
//    static member Default() =
//        {
//            temp_storage = __null()
//            linear_tid = 0
//        }
//
//
//
//[<Record>]
//type BlockLoad =
//    {
//        BLOCK_THREADS       : int
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] ALGORITHM           : BlockLoadAlgorithm
//        WARP_TIME_SLICING   : bool
//        ThreadFields        : ThreadFields
//    }
//
//
//    [<ReflectedDefinition>]
//    member this.Initialize() =
//        this.ThreadFields.temp_storage <- PrivateStorage()
//        this.ThreadFields.linear_tid <- threadIdx.x
//        this
//    
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<int>) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- threadIdx.x
//        this
//    
//    [<ReflectedDefinition>]
//    member this.Initialize(linear_tid:int) =
//        this.ThreadFields.temp_storage <- PrivateStorage()
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid:int) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Load(block_itr:deviceptr<int>, items:deviceptr<int>) = 
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (None, None)
//
//    [<ReflectedDefinition>]
//    member this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (Some valid_items, None)
//
//    [<ReflectedDefinition>]
//    member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|| (Some valid_items, Some oob_default)
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread, algorithm, warp_time_slicing) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = warp_time_slicing
//            ThreadFields        = ThreadFields.Default()
//        }
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread, algorithm) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread) = 
//        {
//            
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = BlockLoadAlgorithm.BLOCK_LOAD_DIRECT
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }


//
//let vars (temp_storage:deviceptr<int> option) (linear_tid:int option) =
//    match temp_storage, linear_tid with
//    | Some temp_storage, Some linear_tid -> temp_storage,       linear_tid
//    | None,              Some linear_tid -> privateStorage(),   linear_tid
//    | Some temp_storage, None ->            temp_storage,       threadIdx.x
//    | None,              None ->            privateStorage(),   threadIdx.x
//
//
//
//    [<Record>]
//    type LoadInternal =
//        {
//            mutable real : RealTraits<int>
//            mutable ITEMS_PER_THREAD : int option
//            mutable BLOCK_THREADS : int option
//            mutable ALGORITHM : BlockLoadAlgorithm
//            mutable temp_storage : deviceptr<int> option
//            mutable linear_tid : int option
//            mutable LoadDirectBlocked : LoadDirectBlocked<int> option
//            mutable LoadDirectBlockedVectorized : LoadDirectBlockedVectorized<int> option
//            mutable LoadDirectStriped : LoadDirectStriped<int> option
//        }
//
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
//                this.LoadDirectBlocked <- LoadDirectBlocked.Create(this.real, this.ITEMS_PER_THREAD.Value) |> Some
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> 
//                this.LoadDirectBlockedVectorized <- LoadDirectBlockedVectorized.Create(this.real, this.ITEMS_PER_THREAD.Value) |> Some
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        static member inline Create(real:RealTraits<int>, _ALGORITHM:BlockLoadAlgorithm, linear_tid:int) =
//            {   real = real;
//                ITEMS_PER_THREAD = None;
//                BLOCK_THREADS = None;
//                ALGORITHM = _ALGORITHM;
//                temp_storage = None;
//                linear_tid = linear_tid |> Some;
//                LoadDirectBlocked = None;
//                LoadDirectStriped = None}
//
//        [<ReflectedDefinition>]
//        static member inline Create(real:RealTraits<int>, _ALGORITHM:BlockLoadAlgorithm) =
//            {   real = real;
//                ITEMS_PER_THREAD = None;
//                BLOCK_THREADS = None;
//                ALGORITHM = _ALGORITHM;
//                temp_storage = None;
//                linear_tid = None;
//                LoadDirectBlocked = None;
//                LoadDirectStriped = None}
//
//
//    [<Record>]
//    type BlockLoad =
//        {
//            real : RealTraits<int>
//            mutable BLOCK_THREADS      : int
//            mutable ITEMS_PER_THREAD   : int
//            mutable ALGORITHM          : BlockLoadAlgorithm
//            mutable WARP_TIME_SLICING  : bool
//            TempStorage : Expr<unit -> deviceptr<int>> option
//            LoadInternal : LoadInternal<int> option
//        }
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        static member Create(real:RealTraits<int>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int, _ALGORITHM:BlockLoadAlgorithm, _WARP_TIME_SLICING:bool) =
//            {   real = real;
//                BLOCK_THREADS = _BLOCK_THREADS;
//                ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//                ALGORITHM = _ALGORITHM;
//                WARP_TIME_SLICING = _WARP_TIME_SLICING;
//                TempStorage = None;
//                LoadInternal = LoadInternal.Create(real, _ALGORITHM) |> Some}
//
//
//    type BlockLoad(?_ITEMS_PER_THREAD_:int,?BLOCK_THREADS:int) =
//        abstract LoadDirectBlocked : (int * deviceptr<int> * deviceptr<int>) -> unit
//        abstract LoadDirectBlocked : (int * deviceptr<int> * deviceptr<int> * int) -> unit
//        abstract LoadDirectBlocked : (int * deviceptr<int> * deviceptr<int> * int * 'T) -> unit
//        abstract LoadDirectBlockedVectorized : (int * deviceptr<int> * deviceptr<int>) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<int> * deviceptr<int>) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<int> * deviceptr<int> * int) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<int> * deviceptr<int> * int * 'T) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<int> * deviceptr<int>) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<int> * deviceptr<int> * int) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<int> * deviceptr<int> * int * 'T) -> unit
//
//[<Record>]
//type LoadDirectBlocked =
//    {
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] real : RealTraits<int>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        let bounds = valid_items - (linear_tid * this.ITEMS_PER_THREAD)
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectBlocked(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<int>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<int>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }
//
//
//[<Record>]
//type LoadDirectBlockedVectorized =
//    {
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] real : RealTraits<int>
//    }
//
//        
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlockedVectorized(linear_tid:int, block_ptr:deviceptr<int>, items:deviceptr<int>) =
//        let MAX_VEC_SIZE = CUB_MIN 4 this.ITEMS_PER_THREAD
//        let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((this.ITEMS_PER_THREAD % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
//        let VECTORS_PER_THREAD = this.ITEMS_PER_THREAD / VEC_SIZE
//        let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret
//
//        let vec_items = __local__.Array<CubVector<int>>(VECTORS_PER_THREAD) |> __array_to_ptr
//
//        for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM].Ptr |> __ptr_to_obj
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<int>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<int>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }
//
//[<Record>]
//type LoadDirectStriped =
//    {
//        BLOCK_THREADS : int
//        ITEMS_PER_THREAD : int
//        [<RecordExcludedField>] real : RealTraits<int>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        let bounds = valid_items - linear_tid
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
//            if (ITEM * this.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectStriped(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<int>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int) =
//        {   BLOCK_THREADS = _BLOCK_THREADS
//            ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<int>) =
//        {   BLOCK_THREADS = 128;
//            ITEMS_PER_THREAD = 128;
//            real = real }
//
//
//[<Record>]
//type LoadDirectWarpStriped =
//    {
//        mutable ITEMS_PER_THREAD : int
//        [<RecordExcludedField>] real : RealTraits<int>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>) =
//        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//        let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD
//
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//        let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD
//        let bounds = valid_items - warp_offset - tid
//
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
//            if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<int>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<int>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }