[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Exchange

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Ptx

//type TemplateParameters =
//    {
//        BLOCK_THREADS       :   int
//        ITEMS_PER_THREAD    :   int
//        WARP_TIME_SLICING   :   bool
//    }
//
//    member this.Get = (this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//
//    static member Default(block_threads, items_per_thread) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            WARP_TIME_SLICING   = false
//        }

module private Internal =
    module Constants =
        let LOG_WARP_THREADS = CUB_PTX_LOG_WARP_THREADS
        let WARP_THREADS = 1 <<< LOG_WARP_THREADS

        let WARPS = 
            fun block_threads _ _ -> 
                (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS

        let LOG_SMEM_BANKS = CUB_PTX_LOG_SMEM_BANKS
        let SMEM_BANKS = 1 <<< LOG_SMEM_BANKS

        let TILE_ITEMS = 
            fun block_threads items_per_thread _ ->
                block_threads * items_per_thread

        let TIME_SLICES = 
            fun block_threads _ warp_time_slicing -> 
                if warp_time_slicing then (block_threads, (), ()) |||> WARPS else 1

        let TIME_SLICED_THREADS = 
            fun block_threads _ warp_time_slicing ->
                if warp_time_slicing then (block_threads, WARP_THREADS) ||> CUB_MIN else block_threads

        let TIME_SLICED_ITEMS = 
            fun block_threads items_per_thread warp_time_slicing -> 
                ((block_threads, (), warp_time_slicing) |||> TIME_SLICED_THREADS) * items_per_thread

        let WARP_TIME_SLICED_THREADS = 
            fun block_threads _ _ -> 
                (block_threads, WARP_THREADS) ||> CUB_MIN

        let WARP_TIME_SLICED_ITEMS = 
            fun block_threads items_per_thread _ ->
                ((block_threads, (), ()) |||> WARP_TIME_SLICED_THREADS) * items_per_thread

        let INSERT_PADDING = 
            fun _ items_per_thread _ -> 
                ((items_per_thread &&& (items_per_thread - 1)) = 0)

        let PADDING_ITEMS = 
            fun block_threads items_per_thread warp_time_slicing -> 
                if ((), items_per_thread, ()) |||> INSERT_PADDING then 
                    (block_threads, items_per_thread, warp_time_slicing) 
                    |||> TIME_SLICED_ITEMS >>> LOG_SMEM_BANKS 
                else 
                    0

    module Sig =
        module BlockToStriped =
            type DefaultExpr            = Expr<deviceptr<int> -> unit>
            type WithTimeslicingExpr    = Expr<deviceptr<int> -> unit>

        module BlockToWarpStriped =
            type DefaultExpr            = Expr<deviceptr<int> -> unit>
            type WithTimeslicingExpr    = Expr<deviceptr<int> -> unit>

        module StripedToBlocked =
            type DefaultExpr            = Expr<deviceptr<int> -> unit>
            type WithTimeslicingExpr    = Expr<deviceptr<int> -> unit>

        module WarpStripedToBlocked =
            type DefaultExpr            = Expr<deviceptr<int> -> unit>
            type WithTimeslicingExpr    = Expr<deviceptr<int> -> unit>

        module ScatterToBlocked =
            type DefaultExpr            = Expr<deviceptr<int> -> deviceptr<int> -> unit>
            type WithTimeslicingExpr    = Expr<deviceptr<int> -> deviceptr<int> -> unit>

        module ScatterToStriped =
            type DefaultExpr            = Expr<deviceptr<int> -> deviceptr<int> -> unit>
            type WithTimeslicingExpr    = Expr<deviceptr<int> -> deviceptr<int> -> unit>


module BlockedToStriped =
    open Internal

    type API =
        {
            Default  : Sig.BlockToStriped.DefaultExpr
            WithTimeslicing     : Sig.BlockToStriped.WithTimeslicingExpr
        }

    let private Default block_threads items_per_thread warp_time_slicing =
        let INSERT_PADDING =    Constants.INSERT_PADDING
                                <|||    (block_threads, items_per_thread, warp_time_slicing)
        
        let LOG_SMEM_BANKS =    Constants.LOG_SMEM_BANKS                                

        fun (temp_storage:deviceptr<int>) (linear_tid:int) (warp_lane:int) (warp_id:int) _ ->
            <@ fun (items:deviceptr<int>) ->                           
                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
                    if INSERT_PADDING then 
                        item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = ITEM * block_threads + linear_tid
                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]
            @>

    let private WithTimeslicing block_threads items_per_thread warp_time_slicing =
        let INSERT_PADDING =    Constants.INSERT_PADDING
                                <|||    (block_threads, items_per_thread, warp_time_slicing)

        let TIME_SLICES =       Constants.TIME_SLICES
                                <|||    (block_threads, items_per_thread, warp_time_slicing)

        let TIME_SLICED_ITEMS = Constants.TIME_SLICED_ITEMS
                                <|||    (block_threads, items_per_thread, warp_time_slicing)

        let LOG_SMEM_BANKS =    Constants.LOG_SMEM_BANKS

        fun (temp_storage:deviceptr<int>) (linear_tid:int) (warp_lane:int) (warp_id:int) _ ->
            <@ fun (items:deviceptr<int>) ->
                let temp_items = __local__.Array(items_per_thread)
                
                for SLICE = 0 to (TIME_SLICES - 1) do
                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS

                    __syncthreads()

                    if warp_id = SLICE then
                        for ITEM = 0 to (items_per_thread - 1) do
                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                            temp_storage.[item_offset] <- items.[ITEM]

                    __syncthreads()

                    for ITEM = 0 to (items_per_thread - 1) do
                        let STRIP_OFFSET = ITEM * block_threads
                        let STRIP_OOB = STRIP_OFFSET + block_threads

                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                                temp_items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (items_per_thread - 1) do
                    items.[ITEM] <- temp_items.[ITEM]
        @>


    let api block_threads items_per_thread warp_time_slicing =
        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
            {
                Default =    Default
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)

                WithTimeslicing =       WithTimeslicing
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)
            }


module BlockedToWarpStriped =
    open Internal

    type API =
        {
            Default  : Sig.BlockToWarpStriped.DefaultExpr
            WithTimeslicing     : Sig.BlockToWarpStriped.WithTimeslicingExpr
        }

    let private Default block_threads items_per_thread warp_time_slicing =
        let LOG_SMEM_BANKS =            Constants.LOG_SMEM_BANKS

        let INSERT_PADDING =            Constants.INSERT_PADDING
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)

        let WARP_TIME_SLICED_THREADS =  Constants.WARP_TIME_SLICED_THREADS
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)

        fun (temp_storage:deviceptr<int>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) ->
            <@ fun (items:deviceptr<int>) ->

                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = warp_offset + ITEM + (warp_lane * items_per_thread)
                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]
            @>

    let private WithTimeslicing block_threads items_per_thread warp_time_slicing =
        let TIME_SLICES =               Constants.TIME_SLICES
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)

        let INSERT_PADDING =            Constants.INSERT_PADDING
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)

        let LOG_SMEM_BANKS =            Constants.LOG_SMEM_BANKS

        let WARP_TIME_SLICED_THREADS =  Constants.WARP_TIME_SLICED_THREADS
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)

        fun (temp_storage:deviceptr<int>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) ->
            <@ fun (items:deviceptr<int>) ->

                for SLICE = 0 to (TIME_SLICES - 1) do
                    __syncthreads()
                        
                    if warp_id = SLICE then
                        for ITEM = 0 to (items_per_thread - 1) do
                            let mutable item_offset = ITEM + (warp_lane * items_per_thread)
                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                            temp_storage.[item_offset] <- items.[ITEM]

                        for ITEM = 0 to (items_per_thread - 1) do
                            let mutable item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                            items.[ITEM] <- temp_storage.[item_offset]
            @>


    let api block_threads items_per_thread warp_time_slicing =
        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
            {
                Default =    Default
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)

                WithTimeslicing =       WithTimeslicing
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)
            }


module StripedToBlocked =
    open Internal

    type API =
        {
            Default             : Sig.StripedToBlocked.DefaultExpr
            WithTimeslicing     : Sig.StripedToBlocked.WithTimeslicingExpr
        }

    let private Default block_threads items_per_thread warp_time_slicing =
        let INSERT_PADDING =    Constants.INSERT_PADDING
                                <|||    (block_threads, items_per_thread, warp_time_slicing)        

        let LOG_SMEM_BANKS =    Constants.LOG_SMEM_BANKS

        fun (temp_storage:deviceptr<int>) (linear_tid:int) _ _ _ ->
            <@ fun (items:deviceptr<int>) ->
                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = (ITEM * block_threads) + linear_tid
                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]
            @>

    let private WithTimeslicing block_threads items_per_thread warp_time_slicing =
        let init = (block_threads, items_per_thread, warp_time_slicing)
        let INSERT_PADDING =    init |||>   Constants.INSERT_PADDING
        let LOG_SMEM_BANKS =                Constants.LOG_SMEM_BANKS
        let TIME_SLICES =       init |||>   Constants.TIME_SLICES
        let TIME_SLICED_ITEMS = init |||>   Constants.TIME_SLICED_ITEMS
        
        fun (temp_storage:deviceptr<int>) (linear_tid:int) (warp_lane:int) (warp_id:int) _ ->
            <@ fun (items:deviceptr<int>) ->
                let temp_items = __local__.Array(items_per_thread)

                for SLICE = 0 to (TIME_SLICES - 1) do
                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS

                    __syncthreads()

                    for ITEM = 0 to (items_per_thread - 1) do
                        let STRIP_OFFSET = ITEM * block_threads
                        let STRIP_OOB = STRIP_OFFSET + block_threads
                            
                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                                temp_storage.[item_offset] <- items.[ITEM]

                    __syncthreads()

                    if warp_id = SLICE then
                        for ITEM = 0 to (items_per_thread - 1) do
                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                            temp_items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (items_per_thread - 1) do
                    items.[ITEM] <- temp_items.[ITEM]
            @>


    let api block_threads items_per_thread warp_time_slicing =
        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
            {
                Default =    Default
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)

                WithTimeslicing =       WithTimeslicing
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)
            }


module WarpStripedToBlocked =
    open Internal

    type API =
        {
            Default  : Sig.WarpStripedToBlocked.DefaultExpr
            WithTimeslicing     : Sig.WarpStripedToBlocked.WithTimeslicingExpr
        }

    let private Default block_threads items_per_thread warp_time_slicing =
        let WARP_TIME_SLICED_THREADS =  Constants.WARP_TIME_SLICED_THREADS
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
        
        let INSERT_PADDING =            Constants.INSERT_PADDING
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)        
        
        let LOG_SMEM_BANKS =            Constants.LOG_SMEM_BANKS

        fun (temp_storage:deviceptr<int>) _ (warp_lane:int) (warp_id:int) (warp_offset:int) ->
            <@ fun (items:deviceptr<int>) ->
                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = warp_offset + ITEM + (warp_lane * items_per_thread)
                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]
            @>

    let private WithTimeslicing block_threads items_per_thread warp_time_slicing =
        let TIME_SLICES =               Constants.TIME_SLICES
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)        
        
        let WARP_TIME_SLICED_THREADS =  Constants.WARP_TIME_SLICED_THREADS
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
        
        let INSERT_PADDING =            Constants.INSERT_PADDING
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)        
        
        let LOG_SMEM_BANKS =            Constants.LOG_SMEM_BANKS
                
        fun (temp_storage:deviceptr<int>) _ (warp_lane:int) (warp_id:int) _ ->
            <@ fun (items:deviceptr<int>) ->
                for SLICE = 0 to (TIME_SLICES - 1) do
                    __syncthreads()

                    if warp_id = SLICE then
                        for ITEM = 0 to (items_per_thread - 1) do
                            let mutable item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                            temp_storage.[item_offset] <- items.[ITEM]

                        for ITEM = 0 to (items_per_thread - 1) do
                            let mutable item_offset = ITEM + (warp_lane * items_per_thread)
                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                            items.[ITEM] <- temp_storage.[item_offset]        
            @>


    let api block_threads items_per_thread warp_time_slicing =
        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
            {
                Default =    Default
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)

                WithTimeslicing =       WithTimeslicing
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)
            }


module ScatterToBlocked =
    open Internal

    type API =
        {
            Default  : Sig.ScatterToBlocked.DefaultExpr
            WithTimeslicing     : Sig.ScatterToBlocked.WithTimeslicingExpr
        }

    let private Default block_threads items_per_thread warp_time_slicing =
        let init = (block_threads, items_per_thread, warp_time_slicing)
        let INSERT_PADDING =    init |||>   Constants.INSERT_PADDING
        let LOG_SMEM_BANKS =                Constants.LOG_SMEM_BANKS   
        
        fun (temp_storage:deviceptr<int>) (linear_tid:int) _ _ _ ->
            <@ fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = ranks.[ITEM]
                    if INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
                    if INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    items.[ITEM] <- temp_storage.[item_offset]        
            @>

    let private WithTimeslicing block_threads items_per_thread warp_time_slicing =
        let init = (block_threads, items_per_thread, warp_time_slicing)
        let INSERT_PADDING =            init |||>   Constants.INSERT_PADDING
        let LOG_SMEM_BANKS =                        Constants.LOG_SMEM_BANKS
        let TIME_SLICES =               init |||>   Constants.TIME_SLICES
        let TIME_SLICED_ITEMS =         init |||>   Constants.TIME_SLICED_ITEMS
        let WARP_TIME_SLICED_ITEMS =    init |||>   Constants.WARP_TIME_SLICED_ITEMS

        fun (temp_storage:deviceptr<int>) (linear_tid:int) (warp_lane:int) (warp_id:int) _ ->
            <@ fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
                let temp_items = __local__.Array(items_per_thread)
                for SLICE = 0 to (TIME_SLICES - 1) do
                    __syncthreads()

                    let SLICE_OFFSET = TIME_SLICED_ITEMS * SLICE

                    for ITEM = 0 to (items_per_thread - 1) do
                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS) then
                            if INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                            temp_storage.[item_offset] <- items.[ITEM]

                    __syncthreads()


                    if warp_id = SLICE then
                        for ITEM = 0 to (items_per_thread - 1) do
                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
                            if INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                            temp_items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (items_per_thread - 1) do
                    items.[ITEM] <- temp_items.[ITEM]        
            @>


    let api block_threads items_per_thread warp_time_slicing =
        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
            {
                Default =    Default
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)

                WithTimeslicing =       WithTimeslicing
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)
            }


module ScatterToStriped =
    open Internal

    type API =
        {
            Default  : Sig.ScatterToStriped.DefaultExpr
            WithTimeslicing     : Sig.ScatterToStriped.WithTimeslicingExpr
        }

    let private Default block_threads items_per_thread warp_time_slicing =
        let init = (block_threads, items_per_thread, warp_time_slicing)
        let INSERT_PADDING =            init |||>   Constants.INSERT_PADDING
        let LOG_SMEM_BANKS =                        Constants.LOG_SMEM_BANKS        

        fun (temp_storage:deviceptr<int>) (linear_tid:int) _ _ _ ->
            <@ fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = ranks.[ITEM]
                    if INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (items_per_thread - 1) do
                    let mutable item_offset = (ITEM * block_threads) + linear_tid
                    if INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    items.[ITEM] <- temp_storage.[item_offset]        
            @>

    let private WithTimeslicing block_threads items_per_thread warp_time_slicing =
        let init = (block_threads, items_per_thread, warp_time_slicing)
        let INSERT_PADDING =            init |||>   Constants.INSERT_PADDING
        let LOG_SMEM_BANKS =                        Constants.LOG_SMEM_BANKS
        let TIME_SLICES =               init |||>   Constants.TIME_SLICES
        let TIME_SLICED_ITEMS =         init |||>   Constants.TIME_SLICED_ITEMS
        let WARP_TIME_SLICED_ITEMS =    init |||>   Constants.WARP_TIME_SLICED_ITEMS        
        
        fun (temp_storage:deviceptr<int>) (linear_tid:int) _ _ _ ->
            <@ fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
                let temp_items = __local__.Array(items_per_thread)
                for SLICE = 0 to (TIME_SLICES - 1) do
                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS

                    __syncthreads()

                    for ITEM = 0 to (items_per_thread - 1) do
                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS) then
                            if INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                            temp_storage.[item_offset] <- items.[ITEM]

                    __syncthreads()

                    for ITEM = 0 to (items_per_thread - 1) do
                        let STRIP_OFFSET = ITEM * block_threads
                        let STRIP_OOB = STRIP_OFFSET + block_threads

                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
                                temp_items.[ITEM] <- temp_storage.[item_offset]

                for ITEM = 0 to (items_per_thread - 1) do
                    items.[ITEM] <- temp_items.[ITEM]
            @>


    let api block_threads items_per_thread warp_time_slicing =
        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
            {
                Default =    Default
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)

                WithTimeslicing =       WithTimeslicing
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                                        <|||    (warp_lane, warp_id, warp_offset)
            }

module BlockExchange =
    type API =
        {
            BlockedToStriped        : BlockedToStriped.API
            BlockedToWarpStriped    : BlockedToWarpStriped.API
            StripedToBlocked        : StripedToBlocked.API
            WarpStripedToBlocked    : WarpStripedToBlocked.API
            ScatterToBlocked        : ScatterToBlocked.API
            ScatterToStriped        : ScatterToStriped.API
        }

    let api block_threads items_per_thread warp_time_slicing =
        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
            {
                BlockedToStriped        =   BlockedToStriped.api
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                                            <|||    (warp_lane, warp_id, warp_offset)

                BlockedToWarpStriped    =   BlockedToWarpStriped.api
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                                            <|||    (warp_lane, warp_id, warp_offset)

                StripedToBlocked        =   StripedToBlocked.api
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                                            <|||    (warp_lane, warp_id, warp_offset)
                                                            
                WarpStripedToBlocked    =   WarpStripedToBlocked.api
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                                            <|||    (warp_lane, warp_id, warp_offset)
                                                            
                ScatterToBlocked        =   ScatterToBlocked.api
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                                            <|||    (warp_lane, warp_id, warp_offset)
                                                            
                ScatterToStriped        =   ScatterToStriped.api
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                                            <|||    (warp_lane, warp_id, warp_offset)            
            }

//let linear_tid = 
//    fun (tidx:int option) ->
//        match tidx with
//        | Some tidx -> tidx
//        | None -> threadIdx.x
//
//let warp_id = 
//    fun (linear_tid:int option) ->
//        match linear_tid with
//        | Some linear_tid -> 
//            linear_tid >>> LOG_WARP_THREADS
//        | None ->
//            threadIdx.x >>> LOG_WARP_THREADS
//
//let warp_lane = 
//    fun (linear_tid:int option) ->
//        match linear_tid with
//        | Some linear_tid ->
//            linear_tid &&& (WARP_THREADS - 1)
//        | None ->
//            threadIdx.x &&& (WARP_THREADS - 1)
//
//let warp_offset = 
//    fun block_threads items_per_thread -> 
//        fun (linear_tid:int option) ->
//            match linear_tid with
//            | Some linear_tid ->
//                (linear_tid |> Some |> warp_id) * ((block_threads, items_per_thread, ()) |||> WARP_TIME_SLICED_ITEMS)
//            | None ->
//                (threadIdx.x |> Some |> warp_id) * ((block_threads, items_per_thread, ()) |||> WARP_TIME_SLICED_ITEMS)



//let blockedToStriped (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING      = ((),items_per_thread,()) |||> INSERT_PADDING
//    let TIME_SLICED_ITEMS   = (block_threads, items_per_thread, warp_time_slicing) |||> TIME_SLICED_ITEMS
//    let TIME_SLICES         = (block_threads, (), warp_time_slicing) |||> TIME_SLICES
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
//                    if INSERT_PADDING then 
//                        item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = ITEM * block_threads + linear_tid
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(items_per_thread)
//            fun (items:deviceptr<int>) ->
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    let warp_id = linear_tid |> Some |> warp_id
//                    let warp_lane = linear_tid |> Some |> warp_lane
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
        


//let blockedToWarpStriped (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING              = ((), items_per_thread, ())                |||> INSERT_PADDING
//    let WARP_TIME_SLICED_THREADS    = (block_threads, (), ())                   |||> WARP_TIME_SLICED_THREADS
//    let TIME_SLICES                 = (block_threads, (), warp_time_slicing)    |||> TIME_SLICES
//
//    let warp_offset = (block_threads, items_per_thread) ||> warp_offset
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_offset = linear_tid |> Some |> warp_offset
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * items_per_thread)
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_id = linear_tid |> Some |> warp_id
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    __syncthreads()
//                        
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * items_per_thread)
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
        
//
//let stripedToBlocked (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING      = ((), items_per_thread, ())                              |||> INSERT_PADDING
//    let TIME_SLICES         = (block_threads, (), warp_time_slicing)                  |||> TIME_SLICES
//    let TIME_SLICED_ITEMS   = (block_threads, items_per_thread, warp_time_slicing)    |||> TIME_SLICED_ITEMS
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (ITEM * block_threads) + linear_tid
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(items_per_thread)
//
//            fun (items:deviceptr<int>) ->
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//                            
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                                temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    let warp_id = linear_tid |> Some |> warp_id
//                    let warp_lane = linear_tid |> Some |> warp_lane
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        

//let warpStripedToBlocked (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let WARP_TIME_SLICED_THREADS    = (block_threads, (), ())                   |||> WARP_TIME_SLICED_THREADS
//    let INSERT_PADDING              = ((), items_per_thread, ())                |||> INSERT_PADDING
//    let TIME_SLICES                 = (block_threads, (), warp_time_slicing)    |||> TIME_SLICES
//    let warp_offset                 = (block_threads, items_per_thread)          ||> warp_offset
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_offset = linear_tid |> Some |> warp_offset
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * items_per_thread)
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_id = linear_tid |> Some |> warp_id
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * items_per_thread)
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
        
//
//let scatterToBlocked (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING          = ((), items_per_thread, ())                            |||> INSERT_PADDING
//    let TIME_SLICES             = (block_threads, (), warp_time_slicing)                |||> TIME_SLICES
//    let TIME_SLICED_ITEMS       = (block_threads, items_per_thread, warp_time_slicing)  |||> TIME_SLICED_ITEMS
//    let WARP_TIME_SLICED_ITEMS  = (block_threads, items_per_thread, ())                 |||> WARP_TIME_SLICED_ITEMS
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
//                let temp_items = __local__.Array(items_per_thread)
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    let SLICE_OFFSET = TIME_SLICED_ITEMS * SLICE
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS) then
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    let warp_id = linear_tid |> Some |> warp_id
//                    let warp_lane = linear_tid |> Some |> warp_lane
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]


//let scatterToStriped (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING          = ((), block_threads, ())                               |||> INSERT_PADDING
//    let TIME_SLICES             = (block_threads, (), warp_time_slicing)                |||> TIME_SLICES
//    let TIME_SLICED_ITEMS       = (block_threads, items_per_thread, warp_time_slicing)  |||> TIME_SLICED_ITEMS
//    let WARP_TIME_SLICED_ITEMS  = (block_threads, items_per_thread, ())                 |||> WARP_TIME_SLICED_ITEMS
//    
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) (is_valid:deviceptr<int>) (valid_items:int) ->    
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (ITEM * block_threads) + linear_tid
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(items_per_thread)
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) (is_valid:deviceptr<int>) (valid_items:int) ->    
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS) then
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]

//let vars (temp_storage:deviceptr<int> option) (linear_tid:int option) =
//    match temp_storage, linear_tid with
//    | Some temp_storage, Some linear_tid -> temp_storage,       linear_tid
//    | None,              Some linear_tid -> privateStorage(),   linear_tid
//    | Some temp_storage, None ->            temp_storage,       threadIdx.x
//    | None,              None ->            privateStorage(),   threadIdx.x

//let blockExchange (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    fun (temp_storage:deviceptr<int> option) (linear_tid:int option) ->
//        let temp_storage, linear_tid = (temp_storage, linear_tid) ||> vars

//            
//
//[<Record>]
//type Constants =
//    {
//        LOG_WARP_THREADS            : int
//        WARP_THREADS                : int
//        WARPS                       : int
//        LOG_SMEM_BANKS              : int
//        SMEM_BANKS                  : int
//        TILE_ITEMS                  : int
//        TIME_SLICES                 : int
//        TIME_SLICED_THREADS         : int
//        TIME_SLICED_ITEMS           : int
//        WARP_TIME_SLICED_THREADS    : int
//        WARP_TIME_SLICED_ITEMS      : int
//        INSERT_PADDING              : bool
//        PADDING_ITEMS               : int
//    }
//
//    [<ReflectedDefinition>]
//    static member Init(block_threads:int, items_per_thread:int, warp_time_slicing:bool) =
//        let log_warp_threads            = CUB_PTX_LOG_WARP_THREADS
//        let warp_threads                = 1 <<< LOG_WARP_THREADS
//        let warps                       = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
//        let log_smem_banks              = CUB_PTX_LOG_SMEM_BANKS
//        let smem_banks                  = 1 <<< LOG_SMEM_BANKS
//        let tile_items                  = block_threads * items_per_thread
//        let time_slices                 = if warp_time_slicing then warps else 1
//        let time_sliced_threads         = if warp_time_slicing then CUB_MIN block_threads WARP_THREADS else block_threads
//        let time_sliced_items           = time_sliced_threads * items_per_thread
//        let warp_time_sliced_threads    = CUB_MIN block_threads WARP_THREADS
//        let warp_time_sliced_items      = warp_time_sliced_threads * items_per_thread
//        let insert_padding              = ((items_per_thread &&& (items_per_thread - 1)) = 0)
//        let padding_items               = if insert_padding then (time_sliced_items >>> LOG_SMEM_BANKS) else 0
//        {   
//            LOG_WARP_THREADS            = log_warp_threads
//            WARP_THREADS                = warp_threads
//            WARPS                       = warps
//            LOG_SMEM_BANKS              = log_smem_banks
//            SMEM_BANKS                  = SMEM_BANKS
//            TILE_ITEMS                  = tile_items
//            TIME_SLICES                 = time_slices
//            TIME_SLICED_THREADS         = time_sliced_threads
//            TIME_SLICED_ITEMS           = time_sliced_items
//            WARP_TIME_SLICED_THREADS    = warp_time_sliced_threads
//            WARP_TIME_SLICED_ITEMS      = warp_time_sliced_items
//            INSERT_PADDING              = insert_padding
//            PADDING_ITEMS               = padding_items
//        }
//
////    static member Default(block_threads:int, items_per_thread:int) =
////        BlockExchange.Init(block_threads, items_per_thread, false)
////
////
//////let inline privateStorage() = cuda { return! <@ fun (n:int) -> __shared__.Array(n) |> __array_to_ptr @> |> Compiler.DefineFunction }
////
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage    : deviceptr<int>
//        mutable linear_tid      : int
//        mutable warp_id         : int
//        mutable warp_lane       : int
//        mutable warp_offset     : int
//    }
//
//    static member Default() =
//        {
//            temp_storage    = __null()
//            linear_tid      = 0
//            warp_id         = 0
//            warp_lane       = 0
//            warp_offset     = 0
//        }

//    [<ReflectedDefinition>]
//    static member Init(c:Constants) =
//        let linear_tid = threadIdx.x
//        let warp_id = linear_tid >>> c.LOG_WARP_THREADS
//        {   temp_storage = __null()
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (c.WARP_THREADS - 1)
//            warp_id = linear_tid >>> c.LOG_WARP_THREADS
//            warp_offset = warp_id * c.WARP_TIME_SLICED_ITEMS }
//
//    [<ReflectedDefinition>]
//    static member Init(c:Constants, temp_storage:deviceptr<int>) =
//        let linear_tid = threadIdx.x
//        let warp_id = linear_tid >>> c.LOG_WARP_THREADS
//        {   temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (c.WARP_THREADS - 1)
//            warp_id = linear_tid >>> c.LOG_WARP_THREADS
//            warp_offset = warp_id * c.WARP_TIME_SLICED_ITEMS }            
//        
//    [<ReflectedDefinition>]
//    static member Init(c:Constants, linear_tid:int) =
//        let warp_id = linear_tid >>> c.LOG_WARP_THREADS
//        {   temp_storage = __null()
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (c.WARP_THREADS - 1)
//            warp_id = linear_tid >>> c.LOG_WARP_THREADS
//            warp_offset = warp_id * c.WARP_TIME_SLICED_ITEMS }
//
//    [<ReflectedDefinition>]
//    static member Init(c:Constants, temp_storage:deviceptr<int>, linear_tid:int) =
//        let warp_id = linear_tid >>> c.LOG_WARP_THREADS
//        {   temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (c.WARP_THREADS - 1)
//            warp_id = linear_tid >>> c.LOG_WARP_THREADS
//            warp_offset = warp_id * c.WARP_TIME_SLICED_ITEMS }    
//
//let PrivateStorage() = __null()
//
//[<Record>]
//type BlockExchange =
//    {
//        BLOCK_THREADS       : int
//        ITEMS_PER_THREAD    : int
//        WARP_TIME_SLICING   : bool
//        Constants           : Constants
//        ThreadFields        : ThreadFields
//    }
//
//    member this.Initialize() =
//        let linear_tid              = threadIdx.x
//        let warp_threads            = this.Constants.WARP_THREADS
//        let log_warp_threads        = this.Constants.LOG_WARP_THREADS
//        let warp_time_sliced_items  = this.Constants.WARP_TIME_SLICED_ITEMS
//        let warp_id                 = linear_tid >>> log_warp_threads
//        this.ThreadFields.temp_storage  <- PrivateStorage()
//        this.ThreadFields.linear_tid    <- linear_tid
//        this.ThreadFields.warp_lane     <- linear_tid &&& (warp_threads - 1)
//        this.ThreadFields.warp_id       <- linear_tid >>> log_warp_threads
//        this.ThreadFields.warp_offset   <- warp_id * warp_time_sliced_items
//        this
//
//    member this.Initialize(temp_storage:deviceptr<int>) =
//        let linear_tid              = threadIdx.x
//        let warp_threads            = this.Constants.WARP_THREADS
//        let log_warp_threads        = this.Constants.LOG_WARP_THREADS
//        let warp_time_sliced_items  = this.Constants.WARP_TIME_SLICED_ITEMS
//        let warp_id                 = linear_tid >>> log_warp_threads
//        this.ThreadFields.temp_storage  <- temp_storage
//        this.ThreadFields.linear_tid    <- linear_tid
//        this.ThreadFields.warp_lane     <- linear_tid &&& (warp_threads - 1)
//        this.ThreadFields.warp_id       <- linear_tid >>> log_warp_threads
//        this.ThreadFields.warp_offset   <- warp_id * warp_time_sliced_items
//        this
//
//    member this.Initialize(linear_tid:int) =
//        let warp_threads            = this.Constants.WARP_THREADS
//        let log_warp_threads        = this.Constants.LOG_WARP_THREADS
//        let warp_time_sliced_items  = this.Constants.WARP_TIME_SLICED_ITEMS
//        let warp_id                 = linear_tid >>> log_warp_threads
//        this.ThreadFields.temp_storage  <- PrivateStorage()
//        this.ThreadFields.linear_tid    <- linear_tid
//        this.ThreadFields.warp_lane     <- linear_tid &&& (warp_threads - 1)
//        this.ThreadFields.warp_id       <- linear_tid >>> log_warp_threads
//        this.ThreadFields.warp_offset   <- warp_id * warp_time_sliced_items
//        this
//
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid:int) =
//        let warp_threads            = this.Constants.WARP_THREADS
//        let log_warp_threads        = this.Constants.LOG_WARP_THREADS
//        let warp_time_sliced_items  = this.Constants.WARP_TIME_SLICED_ITEMS
//        let warp_id                 = linear_tid >>> log_warp_threads
//        this.ThreadFields.temp_storage  <- temp_storage
//        this.ThreadFields.linear_tid    <- linear_tid
//        this.ThreadFields.warp_lane     <- linear_tid &&& (warp_threads - 1)
//        this.ThreadFields.warp_id       <- linear_tid >>> log_warp_threads
//        this.ThreadFields.warp_offset   <- warp_id * warp_time_sliced_items
//        this
//
//
//    member this.StripedToBlocked(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//        
//        stripedToBlocked
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//        
//
//    member this.BlockedToStriped(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        blockedToStriped
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.WarpStripedToBlocked(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        warpStripedToBlocked
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.BlockedToWarpStriped(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        blockedToWarpStriped
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToBlocked(items:deviceptr<int>, ranks:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        scatterToBlocked
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        scatterToStriped
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>, is_valid:deviceptr<int>, valid_items:int) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//        let insert_padding      = this.Constants.INSERT_PADDING
//        let log_smem_banks      = this.Constants.LOG_SMEM_BANKS
//
//        for ITEM = 0 to items_per_thread - 1 do
//            let mutable item_offset = ranks.[ITEM]
//            if insert_padding then item_offset <- (item_offset, log_smem_banks, item_offset) |||> SHR_ADD
//            if is_valid.[ITEM] <> 0 then temp_storage.[item_offset] <- items.[ITEM]
//
//        __syncthreads()
//
//        for ITEM = 0 to items_per_thread - 1 do
//            let mutable item_offset = (ITEM * block_threads) + linear_tid
//            if item_offset < valid_items then
//                if insert_padding then item_offset <- (item_offset, log_smem_banks, item_offset) |||> SHR_ADD
//                items.[ITEM] <- temp_storage.[item_offset]
//
//
//    static member Create(block_threads:int, items_per_thread:int, warp_time_slicing:bool) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            WARP_TIME_SLICING   = warp_time_slicing
//            Constants           = Constants.Init(block_threads, items_per_thread, warp_time_slicing)
//            ThreadFields        = ThreadFields.Default()
//        }
//
//
//    static member Create(block_threads:int, items_per_thread:int) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            WARP_TIME_SLICING   = false
//            Constants           = Constants.Init(block_threads, items_per_thread, false)
//            ThreadFields        = ThreadFields.Default()
//        }
////[<Record>]
//type BlockExchange =
//    {
//        BLOCK_THREADS : int
//        ITEMS_PER_THREAD : int
//        WARP_TIME_SLICING : bool
//    }
//
//    [<ReflectedDefinition>]
//    member this.BlockToStriped(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS,this.ITEMS_PER_THREAD,this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//        let warp_id = vars.warp_id
//        let warp_lane = vars.warp_lane
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ITEM * this.BLOCK_THREADS + linear_tid
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.ITEMS_PER_THREAD)
//                
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
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
//                            if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
//                                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        
//    [<ReflectedDefinition>]
//    member this.BlockTOWarpStriped(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let warp_lane = vars.warp_lane
//        let warp_offset = vars.warp_offset
//        let warp_id = vars.warp_id
//                        
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    __syncthreads()
//                        
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
//        
//    [<ReflectedDefinition>]
//    member this.StripedToBlocked(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//        let warp_id = vars.warp_id
//        let warp_lane = vars.warp_lane
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//                let temp_items = __local__.Array(this.ITEMS_PER_THREAD)
//
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//                            
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
//                                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        
//    [<ReflectedDefinition>]
//    member this.WarpStripedToBlocked(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let warp_offset = vars.warp_offset
//        let warp_lane = vars.warp_lane
//        let warp_id = vars.warp_id
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
//        
//    [<ReflectedDefinition>]
//    member this.ScatterToBlocked(items:deviceptr<int>, ranks:deviceptr<int>, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.ITEMS_PER_THREAD)
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    let SLICE_OFFSET = c.TIME_SLICED_ITEMS * SLICE
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//
//    [<ReflectedDefinition>]
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>) = //, linear_tid:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.ITEMS_PER_THREAD)
//
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
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
//                            if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
//                                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]