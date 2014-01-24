module Alea.cuExtension.CUB.Block.Exchange

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Ptx


let LOG_WARP_THREADS = CUB_PTX_LOG_WARP_THREADS
let WARP_THREADS = 1 <<< LOG_WARP_THREADS
let WARPS = fun block_threads -> (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
let LOG_SMEM_BANKS = CUB_PTX_LOG_SMEM_BANKS
let SMEM_BANKS = 1 <<< LOG_SMEM_BANKS
let TILE_ITEMS = fun block_threads items_per_thread -> block_threads * items_per_thread
let TIME_SLICES = 
    fun block_threads warp_time_slicing -> if warp_time_slicing then WARPS block_threads else 1
let TIME_SLICED_THREADS = 
    fun block_threads warp_time_slicing -> if warp_time_slicing then (block_threads, WARP_THREADS) ||> CUB_MIN else block_threads
let TIME_SLICED_ITEMS = fun block_threads items_per_thread -> (block_threads, items_per_thread) ||> TIME_SLICED_THREADS * items_per_thread
let WARP_TIME_SLICED_THREADS = fun block_threads -> (block_threads, WARP_THREADS) ||> CUB_MIN
let WARP_TIME_SLICED_ITEMS = fun block_threads items_per_thread -> (block_threads |> WARP_TIME_SLICED_THREADS) * items_per_thread
let INSERT_PADDING = fun items_per_thread -> ((items_per_thread &&& (items_per_thread - 1)) = 0)
let PADDING_ITEMS = fun block_threads items_per_thread -> if items_per_thread |> INSERT_PADDING then (block_threads, items_per_thread) ||> TIME_SLICED_ITEMS >>> LOG_SMEM_BANKS) else 0


//let inline privateStorage() = cuda { return! <@ fun (n:int) -> __shared__.Array<'T>(n) |> __array_to_ptr @> |> Compiler.DefineFunction }

let warp_id = fun linear_tid -> linear_tid >>> LOG_WARP_THREADS
let linear_tid = fun tidx -> tidx
let warp_lane = fun linear_tid -> linear_tid &&& (WARP_THREADS - 1)
let warp_offset = fun linear_tid items_per_thread block_threads -> (linear_tid |> warp_id) * (block_threads, items_per_thread) ||> WARP_TIME_SLICED_ITEMS


 




[<Record>]
type BlockExchangeVars<'T> =
    {
        temp_storage : deviceptr<'T>
        linear_tid : int
        warp_id : int
        warp_lane : int
        warp_offset : int
    }

    [<ReflectedDefinition>]
    static member Init(props:BlockExchangeProps) =
        let linear_tid = threadIdx.x
        let warp_id = linear_tid >>> props.LOG_WARP_THREADS
        {   temp_storage = __null()
            linear_tid = linear_tid
            warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
            warp_id = linear_tid >>> props.LOG_WARP_THREADS
            warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }

    [<ReflectedDefinition>]
    static member Init(props:BlockExchangeProps, temp_storage:deviceptr<'T>) =
        let linear_tid = threadIdx.x
        let warp_id = linear_tid >>> props.LOG_WARP_THREADS
        {   temp_storage = temp_storage
            linear_tid = linear_tid
            warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
            warp_id = linear_tid >>> props.LOG_WARP_THREADS
            warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }            
        
    [<ReflectedDefinition>]
    static member Init(props:BlockExchangeProps, linear_tid:int) =
        let warp_id = linear_tid >>> props.LOG_WARP_THREADS
        {   temp_storage = __null()
            linear_tid = linear_tid
            warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
            warp_id = linear_tid >>> props.LOG_WARP_THREADS
            warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }

    [<ReflectedDefinition>]
    static member Init(props:BlockExchangeProps, temp_storage:deviceptr<'T>, linear_tid:int) =
        let warp_id = linear_tid >>> props.LOG_WARP_THREADS
        {   temp_storage = temp_storage
            linear_tid = linear_tid
            warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
            warp_id = linear_tid >>> props.LOG_WARP_THREADS
            warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }    

let blockExchangeVars 

    fun (temp_storage:deviceptr<'T> option) (linear_tid:int option) ->
        match temp_storage, linear_tid with
        | None, None ->
            fun _ linear_tid ->
                let warp_id = linear_tid |> warp_id
                let temp_storage = __null()
                let warp_lane = warp_lane |> linear_tid
                let warp_offset = linear_tid |> warp_offset

[<Record>]
type BlockExchange<'T> =
    {
        BLOCK_THREADS : int
        ITEMS_PER_THREAD : int
        WARP_TIME_SLICING : bool
    }

    [<ReflectedDefinition>]
    member this.BlockToStriped(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
        let props = BlockExchangeProps.Init(this.BLOCK_THREADS,this.ITEMS_PER_THREAD,this.WARP_TIME_SLICING)
        let vars = BlockExchangeVars<'T>.Init(props)
        let linear_tid = vars.linear_tid
        let warp_id = vars.warp_id
        let warp_lane = vars.warp_lane

        match this.WARP_TIME_SLICING with
        | false ->
            fun (temp_storage:deviceptr<'T>) ->
                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ITEM * this.BLOCK_THREADS + linear_tid
                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]
        | true ->
            fun (temp_storage:deviceptr<'T>) ->
                let temp_items = __local__.Array<'T>(this.ITEMS_PER_THREAD)
                
                for SLICE = 0 to (props.TIME_SLICES - 1) do
                    let SLICE_OFFSET = SLICE * props.TIME_SLICED_ITEMS
                    let SLICE_OOB = SLICE_OFFSET + props.TIME_SLICED_ITEMS

                    __syncthreads()

                    if warp_id = SLICE then
                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                            temp_storage.[item_offset] <- items.[ITEM]

                    __syncthreads()

                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS

                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                            if (item_offset >= 0) && (item_offset < props.TIME_SLICED_ITEMS) then
                                if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                                temp_items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    items.[ITEM] <- temp_items.[ITEM]
        
    [<ReflectedDefinition>]
    member this.BlockTOWarpStriped(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
        let vars = BlockExchangeVars<'T>.Init(props)
        let warp_lane = vars.warp_lane
        let warp_offset = vars.warp_offset
        let warp_id = vars.warp_id
                        
        match this.WARP_TIME_SLICING with
        | false ->
            fun (temp_storage:deviceptr<'T>) ->
                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.ITEMS_PER_THREAD)
                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = warp_offset + (ITEM * props.WARP_TIME_SLICED_THREADS) + warp_lane
                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]

        | true ->
            fun (temp_storage:deviceptr<'T>) ->
                for SLICE = 0 to (props.TIME_SLICES - 1) do
                    __syncthreads()
                        
                    if warp_id = SLICE then
                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                            let mutable item_offset = ITEM + (warp_lane * this.ITEMS_PER_THREAD)
                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                            temp_storage.[item_offset] <- items.[ITEM]

                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                            let mutable item_offset = (ITEM * props.WARP_TIME_SLICED_THREADS) + warp_lane
                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                            items.[ITEM] <- temp_storage.[item_offset]
        
    [<ReflectedDefinition>]
    member this.StripedToBlocked(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
        let vars = BlockExchangeVars<'T>.Init(props)
        let linear_tid = vars.linear_tid
        let warp_id = vars.warp_id
        let warp_lane = vars.warp_lane

        match this.WARP_TIME_SLICING with
        | false ->
            fun (temp_storage:deviceptr<'T>) ->
                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]

        | true ->
            fun (temp_storage:deviceptr<'T>) ->
                let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
                let temp_items = __local__.Array<'T>(this.ITEMS_PER_THREAD)

                for SLICE = 0 to (props.TIME_SLICES - 1) do
                    let SLICE_OFFSET = SLICE * props.TIME_SLICED_ITEMS
                    let SLICE_OOB = SLICE_OFFSET + props.TIME_SLICED_ITEMS

                    __syncthreads()

                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
                            
                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                            if (item_offset >= 0) && (item_offset < props.TIME_SLICED_ITEMS) then
                                if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                                temp_storage.[item_offset] <- items.[ITEM]

                    __syncthreads()

                    if warp_id = SLICE then
                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                            temp_items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    items.[ITEM] <- temp_items.[ITEM]
        
    [<ReflectedDefinition>]
    member this.WarpStripedToBlocked(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
        let vars = BlockExchangeVars<'T>.Init(props)
        let warp_offset = vars.warp_offset
        let warp_lane = vars.warp_lane
        let warp_id = vars.warp_id

        match this.WARP_TIME_SLICING with
        | false ->
            fun (temp_storage:deviceptr<'T>) ->
                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = warp_offset + (ITEM * props.WARP_TIME_SLICED_THREADS) + warp_lane
                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.ITEMS_PER_THREAD)
                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]
        | true ->
            fun (temp_storage:deviceptr<'T>) ->
                for SLICE = 0 to (props.TIME_SLICES - 1) do
                    __syncthreads()

                    if warp_id = SLICE then
                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                            let mutable item_offset = (ITEM * props.WARP_TIME_SLICED_THREADS) + warp_lane
                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                            temp_storage.[item_offset] <- items.[ITEM]

                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                            let mutable item_offset = ITEM + (warp_lane * this.ITEMS_PER_THREAD)
                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                            items.[ITEM] <- temp_storage.[item_offset]
        
    [<ReflectedDefinition>]
    member this.ScatterToBlocked(items:deviceptr<'T>, ranks:deviceptr<int>, linear_tid:int, warp_id:int, warp_lane:int) =
        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
        match this.WARP_TIME_SLICING with
        | false ->
            fun (temp_storage:deviceptr<'T>) ->
                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ranks.[ITEM]
                    if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
                    temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
                    if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
                    items.[ITEM] <- temp_storage.[item_offset]

        | true ->
            fun (temp_storage:deviceptr<'T>) ->
                let temp_items = __local__.Array<'T>(this.ITEMS_PER_THREAD)
                for SLICE = 0 to (props.TIME_SLICES - 1) do
                    __syncthreads()

                    let SLICE_OFFSET = props.TIME_SLICED_ITEMS * SLICE

                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < props.WARP_TIME_SLICED_ITEMS) then
                            if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
                            temp_storage.[item_offset] <- items.[ITEM]

                    __syncthreads()

                    if warp_id = SLICE then
                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
                            if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
                            temp_items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    items.[ITEM] <- temp_items.[ITEM]

    [<ReflectedDefinition>]
    member this.ScatterToStriped(items:deviceptr<'T>, ranks:deviceptr<int>) = //, linear_tid:int) =
        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
        let vars = BlockExchangeVars<'T>.Init(props)
        let linear_tid = vars.linear_tid

        match this.WARP_TIME_SLICING with
        | false ->
            fun (temp_storage:deviceptr<'T>) ->
                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ranks.[ITEM]
                    if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
                    temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
                    if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
                    items.[ITEM] <- temp_storage.[item_offset]

        | true ->
            fun (temp_storage:deviceptr<'T>) ->
                let temp_items = __local__.Array<'T>(this.ITEMS_PER_THREAD)

                for SLICE = 0 to (props.TIME_SLICES - 1) do
                    let SLICE_OFFSET = SLICE * props.TIME_SLICED_ITEMS
                    let SLICE_OOB = SLICE_OFFSET + props.TIME_SLICED_ITEMS

                    __syncthreads()

                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < props.WARP_TIME_SLICED_ITEMS) then
                            if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
                            temp_storage.[item_offset] <- items.[ITEM]

                    __syncthreads()

                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS

                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                            if (item_offset >= 0) && (item_offset < props.TIME_SLICED_ITEMS) then
                                if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
                                temp_items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
                    items.[ITEM] <- temp_items.[ITEM]


//[<Record>]
//type BlockExchangeProps =
//    {
//        LOG_WARP_THREADS : int
//        WARP_THREADS : int
//        WARPS : int
//        LOG_SMEM_BANKS : int
//        SMEM_BANKS : int
//        TILE_ITEMS : int
//        TIME_SLICES : int
//        TIME_SLICED_THREADS : int
//        TIME_SLICED_ITEMS : int
//        WARP_TIME_SLICED_THREADS : int
//        WARP_TIME_SLICED_ITEMS : int
//        INSERT_PADDING : bool
//        PADDING_ITEMS : int
//    }
//
//    [<ReflectedDefinition>]
//    static member Init(_BLOCK_THREADS:int, _ITEMS_PER_THREAD:int, _WARP_TIME_SLICING:bool) =
//        let LOG_WARP_THREADS = CUB_PTX_LOG_WARP_THREADS
//        let WARP_THREADS = 1 <<< LOG_WARP_THREADS
//        let WARPS = (_BLOCK_THREADS + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
//        let LOG_SMEM_BANKS = CUB_PTX_LOG_SMEM_BANKS
//        let SMEM_BANKS = 1 <<< LOG_SMEM_BANKS
//        let TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
//        let TIME_SLICES = if _WARP_TIME_SLICING then WARPS else 1
//        let TIME_SLICED_THREADS = if _WARP_TIME_SLICING then CUB_MIN _BLOCK_THREADS WARP_THREADS else _BLOCK_THREADS
//        let TIME_SLICED_ITEMS = TIME_SLICED_THREADS * _ITEMS_PER_THREAD
//        let WARP_TIME_SLICED_THREADS = CUB_MIN _BLOCK_THREADS WARP_THREADS
//        let WARP_TIME_SLICED_ITEMS = WARP_TIME_SLICED_THREADS * _ITEMS_PER_THREAD
//        let INSERT_PADDING = ((_ITEMS_PER_THREAD &&& (_ITEMS_PER_THREAD - 1)) = 0)
//        let PADDING_ITEMS = if INSERT_PADDING then (TIME_SLICED_ITEMS >>> LOG_SMEM_BANKS) else 0
//        {   LOG_WARP_THREADS = LOG_WARP_THREADS
//            WARP_THREADS = WARP_THREADS
//            WARPS = WARPS
//            LOG_SMEM_BANKS = LOG_SMEM_BANKS
//            SMEM_BANKS = SMEM_BANKS
//            TILE_ITEMS = TILE_ITEMS
//            TIME_SLICES = TIME_SLICES
//            TIME_SLICED_THREADS = TIME_SLICED_THREADS
//            TIME_SLICED_ITEMS = TIME_SLICED_ITEMS
//            WARP_TIME_SLICED_THREADS = WARP_TIME_SLICED_THREADS
//            WARP_TIME_SLICED_ITEMS = WARP_TIME_SLICED_ITEMS
//            INSERT_PADDING = INSERT_PADDING
//            PADDING_ITEMS = PADDING_ITEMS }
//
//    static member Default(_BLOCK_THREADS:int, _ITEMS_PER_THREAD:int) =
//        BlockExchangeProps.Init(_BLOCK_THREADS, _ITEMS_PER_THREAD, false)
//
//
////let inline privateStorage() = cuda { return! <@ fun (n:int) -> __shared__.Array<'T>(n) |> __array_to_ptr @> |> Compiler.DefineFunction }
//
//[<Record>]
//type BlockExchangeVars<'T> =
//    {
//        temp_storage : deviceptr<'T>
//        linear_tid : int
//        warp_id : int
//        warp_lane : int
//        warp_offset : int
//    }
//
//    [<ReflectedDefinition>]
//    static member Init(props:BlockExchangeProps) =
//        let linear_tid = threadIdx.x
//        let warp_id = linear_tid >>> props.LOG_WARP_THREADS
//        {   temp_storage = __null()
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
//            warp_id = linear_tid >>> props.LOG_WARP_THREADS
//            warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }
//
//    [<ReflectedDefinition>]
//    static member Init(props:BlockExchangeProps, temp_storage:deviceptr<'T>) =
//        let linear_tid = threadIdx.x
//        let warp_id = linear_tid >>> props.LOG_WARP_THREADS
//        {   temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
//            warp_id = linear_tid >>> props.LOG_WARP_THREADS
//            warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }            
//        
//    [<ReflectedDefinition>]
//    static member Init(props:BlockExchangeProps, linear_tid:int) =
//        let warp_id = linear_tid >>> props.LOG_WARP_THREADS
//        {   temp_storage = __null()
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
//            warp_id = linear_tid >>> props.LOG_WARP_THREADS
//            warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }
//
//    [<ReflectedDefinition>]
//    static member Init(props:BlockExchangeProps, temp_storage:deviceptr<'T>, linear_tid:int) =
//        let warp_id = linear_tid >>> props.LOG_WARP_THREADS
//        {   temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
//            warp_id = linear_tid >>> props.LOG_WARP_THREADS
//            warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }    
//
//   
//
//[<Record>]
//type BlockExchange<'T> =
//    {
//        BLOCK_THREADS : int
//        ITEMS_PER_THREAD : int
//        WARP_TIME_SLICING : bool
//    }
//
//    [<ReflectedDefinition>]
//    member this.BlockToStriped(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangeProps.Init(this.BLOCK_THREADS,this.ITEMS_PER_THREAD,this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars<'T>.Init(props)
//        let linear_tid = vars.linear_tid
//        let warp_id = vars.warp_id
//        let warp_lane = vars.warp_lane
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<'T>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ITEM * this.BLOCK_THREADS + linear_tid
//                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//        | true ->
//            fun (temp_storage:deviceptr<'T>) ->
//                let temp_items = __local__.Array<'T>(this.ITEMS_PER_THREAD)
//                
//                for SLICE = 0 to (props.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * props.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + props.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < props.TIME_SLICED_ITEMS) then
//                                if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        
//    [<ReflectedDefinition>]
//    member this.BlockTOWarpStriped(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
//        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars<'T>.Init(props)
//        let warp_lane = vars.warp_lane
//        let warp_offset = vars.warp_offset
//        let warp_id = vars.warp_id
//                        
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<'T>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * props.WARP_TIME_SLICED_THREADS) + warp_lane
//                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<'T>) ->
//                for SLICE = 0 to (props.TIME_SLICES - 1) do
//                    __syncthreads()
//                        
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * props.WARP_TIME_SLICED_THREADS) + warp_lane
//                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
//        
//    [<ReflectedDefinition>]
//    member this.StripedToBlocked(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars<'T>.Init(props)
//        let linear_tid = vars.linear_tid
//        let warp_id = vars.warp_id
//        let warp_lane = vars.warp_lane
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<'T>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
//                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<'T>) ->
//                let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//                let temp_items = __local__.Array<'T>(this.ITEMS_PER_THREAD)
//
//                for SLICE = 0 to (props.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * props.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + props.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//                            
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < props.TIME_SLICED_ITEMS) then
//                                if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                                temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        
//    [<ReflectedDefinition>]
//    member this.WarpStripedToBlocked(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
//        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars<'T>.Init(props)
//        let warp_offset = vars.warp_offset
//        let warp_lane = vars.warp_lane
//        let warp_id = vars.warp_id
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<'T>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * props.WARP_TIME_SLICED_THREADS) + warp_lane
//                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                    if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//        | true ->
//            fun (temp_storage:deviceptr<'T>) ->
//                for SLICE = 0 to (props.TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * props.WARP_TIME_SLICED_THREADS) + warp_lane
//                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                            if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
//        
//    [<ReflectedDefinition>]
//    member this.ScatterToBlocked(items:deviceptr<'T>, ranks:deviceptr<int>, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<'T>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<'T>) ->
//                let temp_items = __local__.Array<'T>(this.ITEMS_PER_THREAD)
//                for SLICE = 0 to (props.TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    let SLICE_OFFSET = props.TIME_SLICED_ITEMS * SLICE
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < props.WARP_TIME_SLICED_ITEMS) then
//                            if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//
//    [<ReflectedDefinition>]
//    member this.ScatterToStriped(items:deviceptr<'T>, ranks:deviceptr<int>) = //, linear_tid:int) =
//        let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars<'T>.Init(props)
//        let linear_tid = vars.linear_tid
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<'T>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
//                    if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<'T>) ->
//                let temp_items = __local__.Array<'T>(this.ITEMS_PER_THREAD)
//
//                for SLICE = 0 to (props.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * props.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + props.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < props.WARP_TIME_SLICED_ITEMS) then
//                            if props.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) props.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < props.TIME_SLICED_ITEMS) then
//                                if props.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> props.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]