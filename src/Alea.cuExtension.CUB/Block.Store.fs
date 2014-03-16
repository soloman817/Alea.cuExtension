[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Store

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

type BlockStoreAlgorithm =
    | BLOCK_STORE_DIRECT            = 0
    | BLOCK_STORE_VECTORIZE         = 1
    | BLOCK_STORE_TRANSPOSE         = 2
    | BLOCK_STORE_WARP_TRANSPOSE    = 3

module Template =
    module Host =
        module Params =
            type API =
                {
                    BLOCK_THREADS       : int
                    ITEMS_PER_THREAD    : int
                    ALGORITHM           : BlockStoreAlgorithm
                    WARP_TIME_SLICING   : bool
                }

                
                static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
                    {
                        BLOCK_THREADS       = block_threads
                        ITEMS_PER_THREAD    = items_per_thread
                        ALGORITHM           = algorithm
                        WARP_TIME_SLICING   = warp_time_slicing
                    }

                
                static member Init(block_threads, items_per_thread) = API.Init(block_threads, items_per_thread, BlockStoreAlgorithm.BLOCK_STORE_DIRECT, false)
        

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

            static member Init(block_threads, items_per_thread) = API.Init(block_threads, items_per_thread, BlockStoreAlgorithm.BLOCK_STORE_DIRECT, false)

    module Device =
        module TempStorage =
            type [<Record>] API<'T> = BlockExchange.TempStorage<'T>

        let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:Host.API) = __shared__.Array<'T>(h.SharedMemoryLength) |> __array_to_ptr

        [<Record>]
        type API<'T> =
            { mutable temp_storage : TempStorage.API<'T>; mutable linear_tid : int }

//            [<ReflectedDefinition>] 
//            static member Init(h:Host.API) =
//                { temp_storage  = PrivateStorage<'T>(h); linear_tid = threadIdx.x }
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
        }


module StoreDirectBlocked =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (linear_tid:int)
        (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = h.Params

        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]
        
        
    let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (linear_tid:int)
        (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = h.Params
            
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
            if ITEM + (linear_tid * p.ITEMS_PER_THREAD) < valid_items then
                block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]
        

    //let api (h:_HostApi) = (Default h, Guarded h)



module StoreDirectBlockedVectorized =
    open Template
            
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (linear_tid:int)
        (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        ()
       
    

module StoreDirectStriped =
    open Template
            
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (linear_tid:int)
        (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = h.Params

        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid] <- items.[ITEM]
        

    let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (linear_tid:int)
        (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = h.Params
        
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
            if ((ITEM * p.BLOCK_THREADS) + linear_tid) < valid_items then
                block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid] <- items.[ITEM]
        

//    let api (h:_HostApi) = (Default h, Guarded h)
    


module StoreDirectWarpStriped =
    open Template
            
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (linear_tid:int)
        (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = h.Params
        
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
        
    
    let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (linear_tid:int)
        (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = h.Params
                    
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
            if (warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS) < valid_items) then
                block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
        

module InternalStore =
    open Template

    module BlockStoreDirect =
        
        let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            StoreDirectBlocked.Default h linear_tid block_ptr items
            

        let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            StoreDirectBlocked.Guarded h linear_tid block_ptr items valid_items
  
//        let api (h:_HostApi) = (Default h, Guarded h)


    module BlockStoreVectorized =
        let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            StoreDirectBlockedVectorized.Default h linear_tid block_ptr items
            

        let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            StoreDirectBlocked.Guarded h linear_tid block_ptr items valid_items
    
//        let api (h:_HostApi) = (Default h, Guarded h)


    module BlockStoreTranspose =
        
        let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            StoreDirectStriped.Default h linear_tid block_ptr items
            BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).StripedToBlocked(h.BlockExchangeHostApi, items)
            

        let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            StoreDirectStriped.Guarded h linear_tid block_ptr items valid_items
            BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).StripedToBlocked(h.BlockExchangeHostApi, items)
            

//        let api (h:_HostApi) = (Default h, Guarded h)


    module BlockStoreWarpTranspose =
        
        let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            let WARP_THREADS = CUB_PTX_WARP_THREADS
            if (h.Params.BLOCK_THREADS % WARP_THREADS) <> 0 then 
                ()
            else
                StoreDirectWarpStriped.Default h linear_tid block_ptr items
                BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).WarpStripedToBlocked(h.BlockExchangeHostApi, items)
            

        let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            let WARP_THREADS = CUB_PTX_WARP_THREADS
            if (h.Params.BLOCK_THREADS % WARP_THREADS) <> 0 then
                ()
            else
                StoreDirectWarpStriped.Guarded h linear_tid block_ptr items valid_items
                BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).WarpStripedToBlocked(h.BlockExchangeHostApi, items)
            

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            h.Params.ALGORITHM |> function
            | BlockStoreAlgorithm.BLOCK_STORE_DIRECT ->          BlockStoreDirect.Default h temp_storage linear_tid block_ptr items
            | BlockStoreAlgorithm.BLOCK_STORE_VECTORIZE ->       BlockStoreVectorized.Default h temp_storage linear_tid block_ptr items
            | BlockStoreAlgorithm.BLOCK_STORE_TRANSPOSE ->       BlockStoreTranspose.Default h temp_storage linear_tid block_ptr items
            | BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE ->  BlockStoreWarpTranspose.Default h temp_storage linear_tid block_ptr items
            | _ -> BlockStoreDirect.Default h temp_storage linear_tid block_ptr items

    let [<ReflectedDefinition>] inline Guarded (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            h.Params.ALGORITHM |> function
            | BlockStoreAlgorithm.BLOCK_STORE_DIRECT ->          BlockStoreDirect.Guarded h temp_storage linear_tid block_ptr items valid_items
            | BlockStoreAlgorithm.BLOCK_STORE_VECTORIZE ->       BlockStoreVectorized.Guarded h temp_storage linear_tid block_ptr items valid_items
            | BlockStoreAlgorithm.BLOCK_STORE_TRANSPOSE ->       BlockStoreTranspose.Guarded h temp_storage linear_tid block_ptr items valid_items
            | BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE ->  BlockStoreWarpTranspose.Guarded h temp_storage linear_tid block_ptr items valid_items
            | _ -> BlockStoreDirect.Guarded h temp_storage linear_tid block_ptr items valid_items

   
module BlockStore =
    type TemplateParams     = Template._TemplateParams
    type TempStorage<'T>    = Template._TempStorage<'T>
    
    type HostApi            = Template._HostApi
    type DeviceApi<'T>      = Template._DeviceApi<'T>
    type FunctionApi<'T>    = Template._FunctionApi<'T>

    let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:HostApi) = BlockExchange.PrivateStorage<'T>(h.BlockExchangeHostApi)

    [<Record>]
    type API<'T> =
        { mutable temp_storage : TempStorage<'T>; mutable linear_tid : int }

        [<ReflectedDefinition>] static member Init(h:HostApi)                                                 = { temp_storage = PrivateStorage<'T>(h); linear_tid = threadIdx.x }
        [<ReflectedDefinition>] static member Init(h:HostApi, temp_storage:TempStorage<'T>)                   = { temp_storage = temp_storage; linear_tid = threadIdx.x }
        [<ReflectedDefinition>] static member Init(h:HostApi, linear_tid:int)                                 = { temp_storage = PrivateStorage<'T>(h); linear_tid = linear_tid }
        [<ReflectedDefinition>] static member Init(h:HostApi, temp_storage:TempStorage<'T>, linear_tid:int)   = { temp_storage = temp_storage; linear_tid = linear_tid }
            
        [<ReflectedDefinition>] member this.Store(h, block_ptr, items)                              = InternalStore.Default h this.temp_storage this.linear_tid block_ptr items
        [<ReflectedDefinition>] member this.Store(h, block_ptr, items, valid_items)                 = InternalStore.Guarded h this.temp_storage this.linear_tid block_ptr items valid_items
        


//    let template<'T> (block_threads:int) (items_per_thread:int) (algorithm:BlockStoreAlgorithm) (warp_time_slicing:bool) 
//        : Template<HostApi*FunctionApi<'T>> = cuda {
//        let h = HostApi.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
//        
//        let! _StoreInternal = h |> StoreInternal.api<'T>
//        return h, { Default = _StoreInternal.Default; Guarded = _StoreInternal.Guarded }}
