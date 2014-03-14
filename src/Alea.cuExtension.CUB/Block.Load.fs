[<AutoOpen>]
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

                
        module Constants =
            type API =
                { MAX_VEC_SIZE : int; VEC_SIZE : int; VECTORS_PER_THREAD : int }

                static member Init(p:Params.API) =
                    let max_vec_size = CUB_MIN 4 p.ITEMS_PER_THREAD
                    let vec_size = if ((((max_vec_size - 1) &&& max_vec_size) = 0) && ((p.ITEMS_PER_THREAD % max_vec_size) = 0)) then max_vec_size else 1
                    let vectors_per_thread = p.ITEMS_PER_THREAD / vec_size
                    { MAX_VEC_SIZE = max_vec_size; VEC_SIZE = vec_size; VECTORS_PER_THREAD = vectors_per_thread }

        
        type API =
            {
                BlockExchangeHostApi: BlockExchange.HostApi
                Params              : Params.API
                Constants           : Constants.API
                SharedMemoryLength  : int 
            }

            static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
                let p = Params.API.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
                let c = Constants.API.Init(p)
                let bex_hApi = BlockExchange.HostApi.Init(block_threads, items_per_thread, warp_time_slicing)
                { BlockExchangeHostApi = bex_hApi; Params = p; Constants = c; SharedMemoryLength = bex_hApi.SharedMemoryLength }

            static member Init(block_threads, items_per_thread) = API.Init(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT, false)

    module Device =
        module TempStorage = type [<Record>] API<'T> = BlockExchange.TempStorage<'T>

        let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:Host.API) = 
            __shared__.Array<'T>(h.SharedMemoryLength) |> __array_to_ptr

        [<Record>]
        type API<'T> =
            { mutable temp_storage : TempStorage.API<'T>; mutable linear_tid : int }

//            [<ReflectedDefinition>] 
//            static member Init(h:Host.API) =
//                { temp_storage = PrivateStorage<'T>(h); linear_tid = threadIdx.x }
//
//            [<ReflectedDefinition>] 
//            static member Init(h:Host.API, temp_storage:TempStorage.API<'T>) =
//                { temp_storage = temp_storage; linear_tid = threadIdx.x }
//
//            [<ReflectedDefinition>] 
//            static member Init(h:Host.API, linear_tid:int) =
//                { temp_storage = PrivateStorage<'T>(h); linear_tid = linear_tid }

            [<ReflectedDefinition>] 
            static member Init(h:Host.API, temp_storage:TempStorage.API<'T>, linear_tid:int) =
                { temp_storage = temp_storage; linear_tid = linear_tid }
                           
            
    type _TemplateParams    = Host.Params.API
    type _HostApi           = Host.API

    type _TempStorage<'T>   = Device.TempStorage.API<'T>
    type _DeviceApi<'T>     = Device.API<'T>

    type _FunctionApi<'T> =
        {
            Default         : Function<_DeviceApi<'T> -> deviceptr<'T> -> deviceptr<'T> -> unit>
            Guarded         : Function<_DeviceApi<'T> -> deviceptr<'T> -> deviceptr<'T> -> int -> unit>
            GuardedWithOOB  : Function<_DeviceApi<'T> -> deviceptr<'T> -> deviceptr<'T> -> int -> 'T -> unit>
        }


module LoadDirectBlocked =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = h.Params
            
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]

        
    let [<ReflectedDefinition>] inline Guarded (h:_HostApi) 
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = h.Params
            

        let bounds = valid_items - (linear_tid * p.ITEMS_PER_THREAD)
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]

        
    let [<ReflectedDefinition>] inline GuardedWithOOB (h:_HostApi)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
        let p = h.Params
            

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
        let bounds = valid_items - (linear_tid * p.ITEMS_PER_THREAD)
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]


//    let [<ReflectedDefinition>] inline api (h:_HostApi) (linear_tid:int) = (Default h, Guarded h, GuardedWithOOB h)


module LoadDirectBlockedVectorized =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = h.Params
        let c = h.Constants            

        let ptr = (block_ptr + (linear_tid * c.VEC_SIZE * c.VECTORS_PER_THREAD)) |> __ptr_reinterpret

        let vec_items = __local__.Array<'T>(c.VECTORS_PER_THREAD) |> __array_to_ptr

        for ITEM = 0 to (c.VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM] //|> __ptr_to_obj
        
    //let [<ReflectedDefinition>] inline api (h:_HostApi) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) = Default h d block_ptr items

module LoadDirectStriped =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (linear_tid:int)
        
                (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            let p = h.Params
            
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


    let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (linear_tid:int)
        
                (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            let p = h.Params
            

            let bounds = valid_items - linear_tid
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                if (ITEM * p.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


    let [<ReflectedDefinition>] inline GuardedWithOOB (h:_HostApi) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            let p = h.Params
            

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            let bounds = valid_items - linear_tid
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                if (ITEM * p.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


//    let [<ReflectedDefinition>] inline api (h:_HostApi) (linear_tid:int) = (Default h d, Guarded h d, GuardedWithOOB d)
    

module LoadDirectWarpStriped =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (linear_tid:int)
        
                (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            let p = h.Params
            
            
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]


    let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (linear_tid:int)
        
                (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            let p = h.Params
            

            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            let bounds = valid_items - warp_offset - tid

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]

        
    let [<ReflectedDefinition>] inline GuardedWithOOB (h:_HostApi) (linear_tid:int)
        
                (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            let p = h.Params
            
            
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            let bounds = valid_items - warp_offset - tid

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]


//    let [<ReflectedDefinition>] inline api (h:_HostApi) (linear_tid:int) = (Default h d, Guarded h d, GuardedWithOOB h d)

module private InternalLoad =
    open Template

    module BlockLoadDirect =
        
        let [<ReflectedDefinition>] inline Default (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            LoadDirectBlocked.Default h d.linear_tid block_ptr items
    

        let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            LoadDirectBlocked.Guarded h d.linear_tid block_ptr items valid_items
    

        let [<ReflectedDefinition>] inline GuardedWithOOB (h:_HostApi) (d:_DeviceApi<'T>) 
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            LoadDirectBlocked.GuardedWithOOB h d.linear_tid block_ptr items valid_items oob_default
    

//        let [<ReflectedDefinition>] inline api (h:_HostApi) (d:_DeviceApi<'T>) = (Default h d, Guarded h d, GuardedWithOOB d)


    module BlockLoadVectorized =
        let [<ReflectedDefinition>] inline Default (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            LoadDirectBlockedVectorized.Default h d.linear_tid block_ptr items
    

        let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            LoadDirectBlocked.Guarded h d.linear_tid block_ptr items valid_items
    
            
        let [<ReflectedDefinition>] inline GuardedWithOOB (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            LoadDirectBlocked.GuardedWithOOB h d.linear_tid block_ptr items valid_items oob_default
    

//        let [<ReflectedDefinition>] inline api (h:_HostApi) (d:_DeviceApi<'T>) = (Default h d, Guarded h d, GuardedWithOOB d)


    module BlockLoadTranspose =
    

        let [<ReflectedDefinition>] inline Default (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            LoadDirectStriped.Default h d.linear_tid block_ptr items
            BlockExchange.API<'T>.Create(h.BlockExchangeHostApi, d.temp_storage, d.linear_tid).StripedToBlocked(h.BlockExchangeHostApi, items)
    

        let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            LoadDirectStriped.Guarded h d.linear_tid block_ptr items valid_items
            BlockExchange.API<'T>.Create(h.BlockExchangeHostApi, d.temp_storage, d.linear_tid).StripedToBlocked(h.BlockExchangeHostApi, items)
    

        let [<ReflectedDefinition>] inline GuardedWithOOB (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            LoadDirectStriped.GuardedWithOOB h d.linear_tid block_ptr items valid_items oob_default
            BlockExchange.API<'T>.Create(h.BlockExchangeHostApi, d.temp_storage, d.linear_tid).StripedToBlocked(h.BlockExchangeHostApi, items)
    

//        let [<ReflectedDefinition>] inline api (h:_HostApi) (d:_DeviceApi<'T>) = (Default h d, Guarded h d, GuardedWithOOB d)


    module BlockLoadWarpTranspose =
    
        let [<ReflectedDefinition>] inline Default (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            let WARP_THREADS = CUB_PTX_WARP_THREADS
            if (h.Params.BLOCK_THREADS % WARP_THREADS) <> 0 then
                ()
            else
                LoadDirectWarpStriped.Default h d.linear_tid block_ptr items
                BlockExchange.API<'T>.Create(h.BlockExchangeHostApi, d.temp_storage, d.linear_tid).WarpStripedToBlocked(h.BlockExchangeHostApi, items)
    

        let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            let WARP_THREADS = CUB_PTX_WARP_THREADS
            if (h.Params.BLOCK_THREADS % WARP_THREADS) <> 0 then
                ()
            else
                LoadDirectWarpStriped.Guarded h d.linear_tid block_ptr items valid_items
                BlockExchange.API<'T>.Create(h.BlockExchangeHostApi, d.temp_storage, d.linear_tid).WarpStripedToBlocked(h.BlockExchangeHostApi, items)
    

        let [<ReflectedDefinition>] inline GuardedWithOOB (h:_HostApi) (d:_DeviceApi<'T>)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            let WARP_THREADS = CUB_PTX_WARP_THREADS
            if (h.Params.BLOCK_THREADS % WARP_THREADS) <> 0 then
                ()
            else
                LoadDirectWarpStriped.GuardedWithOOB h d.linear_tid block_ptr items valid_items oob_default
                BlockExchange.API<'T>.Create(h.BlockExchangeHostApi, d.temp_storage, d.linear_tid).WarpStripedToBlocked(h.BlockExchangeHostApi, items)
    
    
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (d:_DeviceApi<'T>) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            h.Params.ALGORITHM |> function
            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.Default h d block_ptr items
            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.Default h d block_ptr items
            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.Default h d block_ptr items
            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.Default h d block_ptr items
            | _ -> BlockLoadDirect.Default h d block_ptr items

    let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (d:_DeviceApi<'T>) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            h.Params.ALGORITHM |> function
            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.Guarded h d block_ptr items valid_items
            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.Guarded h d block_ptr items valid_items
            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.Guarded h d block_ptr items valid_items
            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.Guarded h d block_ptr items valid_items
            | _ -> BlockLoadDirect.Guarded h d block_ptr items valid_items

    let [<ReflectedDefinition>] inline GuardedWithOOB (h:_HostApi) (d:_DeviceApi<'T>) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            h.Params.ALGORITHM |> function
            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.GuardedWithOOB h d block_ptr items valid_items oob_default
            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.GuardedWithOOB h d block_ptr items valid_items oob_default
            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.GuardedWithOOB h d block_ptr items valid_items oob_default
            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.GuardedWithOOB h d block_ptr items valid_items oob_default
            | _ -> BlockLoadDirect.GuardedWithOOB h d block_ptr items valid_items oob_default

//    type API<'T> =
//        {
//            Default         : Function<_DeviceApi<'T> -> deviceptr<'T> -> deviceptr<'T> -> unit>
//            Guarded         : Function<_DeviceApi<'T> -> deviceptr<'T> -> deviceptr<'T> -> int -> unit>
//            GuardedWithOOB  : Function<_DeviceApi<'T> -> deviceptr<'T> -> deviceptr<'T> -> int -> 'T -> unit>
//        }
//
//    let inline api<'T>(h:_HostApi) : Template<API<'T>> = cuda {
//        let p = h.Params
//
//        let _Default, _Guardeh d, _GuardedWithOOB = 
//            p.ALGORITHM |> function
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.api h
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.api h
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.api h
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.api h
//            | _ -> failwith ""
//
//        let! _Default = _Default |> Compiler.DefineFunction
//        let! _Guarded = _Guarded |> Compiler.DefineFunction
//        let! _GuardedWithOOB = _GuardedWithOOB |> Compiler.DefineFunction
//        return { Default = _Default; Guarded = _Guarded; GuardedWithOOB = _GuardedWithOOB }}
        

module BlockLoad =
    
    type TemplateParams     = Template._TemplateParams
    type TempStorage<'T>    = Template._TempStorage<'T>

    type HostApi            = Template._HostApi
    type DeviceApi<'T>      = Template._DeviceApi<'T>
    type FunctionApi<'T>    = Template._FunctionApi<'T>

    let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:HostApi) = Template.Device.PrivateStorage<'T>(h)

    [<Record>]
    type API<'T> =
        {
            mutable DeviceApi  : DeviceApi<'T>
        }

        [<ReflectedDefinition>] static member Create(h:HostApi)                                                 = { DeviceApi = DeviceApi<'T>.Init(h, PrivateStorage<'T>(h), threadIdx.x) }
        [<ReflectedDefinition>] static member Create(h:HostApi, temp_storage:TempStorage<'T>)                   = { DeviceApi = DeviceApi<'T>.Init(h, temp_storage, threadIdx.x) }
        [<ReflectedDefinition>] static member Create(h:HostApi, linear_tid:int)                                 = { DeviceApi = DeviceApi<'T>.Init(h, PrivateStorage<'T>(h), linear_tid) }
        [<ReflectedDefinition>] static member Create(h:HostApi, temp_storage:TempStorage<'T>, linear_tid:int)   = { DeviceApi = DeviceApi<'T>.Init(h, temp_storage, linear_tid) }
            
        [<ReflectedDefinition>] member this.Load(h, block_ptr, items)                              = InternalLoad.Default h this.DeviceApi block_ptr items
        [<ReflectedDefinition>] member this.Load(h, block_ptr, items, valid_items)                 = InternalLoad.Guarded h this.DeviceApi block_ptr items valid_items
        [<ReflectedDefinition>] member this.Load(h, block_ptr, items, valid_items, oob_default)    = InternalLoad.GuardedWithOOB h this.DeviceApi block_ptr items valid_items oob_default


//    let template<'T> (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) 
//        : Template<HostApi*FunctionApi<'T>> = cuda {
//        let h = HostApi.Init(block_threads, items_per_threah d, algorithm, warp_time_slicing)
//        
//        let! _LoadInternal = h |> LoadInternal.api<'T>
//        return h, { 
//            Default         = _LoadInternal.Default
//            Guarded         = _LoadInternal.Guarded
//            GuardedWithOOB  = _LoadInternal.GuardedWithOOB
//        }}