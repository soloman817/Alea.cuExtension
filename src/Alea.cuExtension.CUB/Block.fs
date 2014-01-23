module Alea.cuExtension.CUB.Block
    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

module Discontinuity =
    let f() = "discontinuity"

module Exchange =
    open Macro
    open Ptx
    
    [<Record>]
    type BlockExchangeProps =
        {
            LOG_WARP_THREADS : int
            WARP_THREADS : int
            WARPS : int
            LOG_SMEM_BANKS : int
            SMEM_BANKS : int
            TILE_ITEMS : int
            TIME_SLICES : int
            TIME_SLICED_THREADS : int
            TIME_SLICED_ITEMS : int
            WARP_TIME_SLICED_THREADS : int
            WARP_TIME_SLICED_ITEMS : int
            INSERT_PADDING : bool
            PADDING_ITEMS : int
        }

        [<ReflectedDefinition>]
        static member Init(_BLOCK_THREADS:int, _ITEMS_PER_THREAD:int, _WARP_TIME_SLICING:bool) =
            let LOG_WARP_THREADS = CUB_PTX_LOG_WARP_THREADS
            let WARP_THREADS = 1 <<< LOG_WARP_THREADS
            let WARPS = (_BLOCK_THREADS + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
            let LOG_SMEM_BANKS = CUB_PTX_LOG_SMEM_BANKS
            let SMEM_BANKS = 1 <<< LOG_SMEM_BANKS
            let TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
            let TIME_SLICES = if _WARP_TIME_SLICING then WARPS else 1
            let TIME_SLICED_THREADS = if _WARP_TIME_SLICING then CUB_MIN _BLOCK_THREADS WARP_THREADS else _BLOCK_THREADS
            let TIME_SLICED_ITEMS = TIME_SLICED_THREADS * _ITEMS_PER_THREAD
            let WARP_TIME_SLICED_THREADS = CUB_MIN _BLOCK_THREADS WARP_THREADS
            let WARP_TIME_SLICED_ITEMS = WARP_TIME_SLICED_THREADS * _ITEMS_PER_THREAD
            let INSERT_PADDING = ((_ITEMS_PER_THREAD &&& (_ITEMS_PER_THREAD - 1)) = 0)
            let PADDING_ITEMS = if INSERT_PADDING then (TIME_SLICED_ITEMS >>> LOG_SMEM_BANKS) else 0
            {   LOG_WARP_THREADS = LOG_WARP_THREADS
                WARP_THREADS = WARP_THREADS
                WARPS = WARPS
                LOG_SMEM_BANKS = LOG_SMEM_BANKS
                SMEM_BANKS = SMEM_BANKS
                TILE_ITEMS = TILE_ITEMS
                TIME_SLICES = TIME_SLICES
                TIME_SLICED_THREADS = TIME_SLICED_THREADS
                TIME_SLICED_ITEMS = TIME_SLICED_ITEMS
                WARP_TIME_SLICED_THREADS = WARP_TIME_SLICED_THREADS
                WARP_TIME_SLICED_ITEMS = WARP_TIME_SLICED_ITEMS
                INSERT_PADDING = INSERT_PADDING
                PADDING_ITEMS = PADDING_ITEMS }

        static member Default(_BLOCK_THREADS:int, _ITEMS_PER_THREAD:int) =
            BlockExchangeProps.Init(_BLOCK_THREADS, _ITEMS_PER_THREAD, false)

    type TimeSlicing = | YES | NO

    let inline privateStorage() = cuda { return! <@ fun (n:int) -> __shared__.Array<'T>(n) |> __array_to_ptr @> |> Compiler.DefineFunction }

    [<Record>]
    type BlockExchangePrivateVars<'T> =
        {
            temp_storage : Template<Function<(int -> deviceptr<'T>)>>
            linear_tid : int
            warp_id : int
            warp_lane : int
            warp_offset : int
        }

        [<ReflectedDefinition>]
        static member Init(props:BlockExchangeProps) =
            let linear_tid = threadIdx.x
            let warp_id = linear_tid >>> props.LOG_WARP_THREADS
            {   temp_storage = privateStorage()
                linear_tid = linear_tid
                warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
                warp_id = linear_tid >>> props.LOG_WARP_THREADS
                warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }

        [<ReflectedDefinition>]
        static member Init(props:BlockExchangeProps, temp_storage:Template<Function<(int -> deviceptr<'T>)>>) =
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
            {   temp_storage = privateStorage()
                linear_tid = linear_tid
                warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
                warp_id = linear_tid >>> props.LOG_WARP_THREADS
                warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }

        [<ReflectedDefinition>]
        static member Init(props:BlockExchangeProps, temp_storage:Template<Function<int->deviceptr<'T>>>, linear_tid:int) =
            let warp_id = linear_tid >>> props.LOG_WARP_THREADS
            {   temp_storage = temp_storage
                linear_tid = linear_tid
                warp_lane = linear_tid &&& (props.WARP_THREADS - 1)
                warp_id = linear_tid >>> props.LOG_WARP_THREADS
                warp_offset = warp_id * props.WARP_TIME_SLICED_ITEMS }    

   

    [<Record>]
    type BlockExchange<'T> =
        {
            BLOCK_THREADS : int
            ITEMS_PER_THREAD : int
            WARP_TIME_SLICING : bool
//            Private : BlockExchangePrivateVars<'T>
        }

        
        member this.BlockToStriped(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
            let props = BlockExchangeProps.Init(this.BLOCK_THREADS,this.ITEMS_PER_THREAD,this.WARP_TIME_SLICING)
            let pvars = BlockExchangePrivateVars<'T>.Init(props)
            let linear_tid = pvars.linear_tid
            let warp_id = pvars.warp_id
            let warp_lane = pvars.warp_lane

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

        member this.BlockTOWarpStriped(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
            let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
            let pvars = BlockExchangePrivateVars<'T>.Init(props)
            let warp_lane = pvars.warp_lane
            let warp_offset = pvars.warp_offset
            let warp_id = pvars.warp_id
                        
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

        member this.StripedToBlocked(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
            let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
            let pvars = BlockExchangePrivateVars<'T>.Init(props)
            let linear_tid = pvars.linear_tid
            let warp_id = pvars.warp_id
            let warp_lane = pvars.warp_lane

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

        member this.WarpStripedToBlocked(items:deviceptr<'T>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
            let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
            let pvars = BlockExchangePrivateVars<'T>.Init(props)
            let warp_offset = pvars.warp_offset
            let warp_lane = pvars.warp_lane
            let warp_id = pvars.warp_id

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

        member this.ScatterToStriped(items:deviceptr<'T>, ranks:deviceptr<int>) = //, linear_tid:int) =
            let props = BlockExchangeProps.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
            let pvars = BlockExchangePrivateVars<'T>.Init(props)
            let linear_tid = pvars.linear_tid

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

        
module Histogram =
    let f() = "histogram"

module Load =
    open Macro
    open Vector

    [<Record>]
    type LoadDirectBlocked<'T> =
        {
            mutable ITEMS_PER_THREAD    : int
            [<RecordExcludedField>] real : RealTraits<'T>
        }

        [<ReflectedDefinition>]
        member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]

        [<ReflectedDefinition>]
        member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
            let bounds = valid_items - (linear_tid * this.ITEMS_PER_THREAD)
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]

        [<ReflectedDefinition>]
        member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            this.LoadDirectBlocked(linear_tid, block_itr, items, valid_items)

        [<ReflectedDefinition>]
        static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
            {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
                real = real }

        [<ReflectedDefinition>]
        static member Default(real:RealTraits<'T>) =
            {   ITEMS_PER_THREAD = 128;
                real = real }


    [<Record>]
    type LoadDirectBlockedVectorized<'T> =
        {
            mutable ITEMS_PER_THREAD    : int
            [<RecordExcludedField>] real : RealTraits<'T>
        }

        
        [<ReflectedDefinition>]
        member inline this.LoadDirectBlockedVectorized(linear_tid:int, block_ptr:deviceptr<'T>, items:deviceptr<'T>) =
            let MAX_VEC_SIZE = CUB_MIN 4 this.ITEMS_PER_THREAD
            let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((this.ITEMS_PER_THREAD % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
            let VECTORS_PER_THREAD = this.ITEMS_PER_THREAD / VEC_SIZE
            let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret

            let vec_items = __local__.Array<CubVector<'T>>(VECTORS_PER_THREAD) |> __array_to_ptr

            for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM].Ptr |> __ptr_to_obj

        [<ReflectedDefinition>]
        static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
            {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
                real = real }

        [<ReflectedDefinition>]
        static member Default(real:RealTraits<'T>) =
            {   ITEMS_PER_THREAD = 128;
                real = real }

    [<Record>]
    type LoadDirectStriped<'T> =
        {
            mutable BLOCK_THREADS : int
            mutable ITEMS_PER_THREAD : int
            [<RecordExcludedField>] real : RealTraits<'T>
        }

        [<ReflectedDefinition>]
        member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]

        [<ReflectedDefinition>]
        member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
            let bounds = valid_items - linear_tid
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
                if (ITEM * this.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]

        [<ReflectedDefinition>]
        member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            this.LoadDirectStriped(linear_tid, block_itr, items, valid_items)

        [<ReflectedDefinition>]
        static member Create(real:RealTraits<'T>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int) =
            {   BLOCK_THREADS = _BLOCK_THREADS
                ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
                real = real }

        [<ReflectedDefinition>]
        static member Default(real:RealTraits<'T>) =
            {   BLOCK_THREADS = 128;
                ITEMS_PER_THREAD = 128;
                real = real }


    [<Record>]
    type LoadDirectWarpStriped<'T> =
        {
            mutable ITEMS_PER_THREAD : int
            [<RecordExcludedField>] real : RealTraits<'T>
        }

        [<ReflectedDefinition>]
        member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD

            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]

        [<ReflectedDefinition>]
        member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD
            let bounds = valid_items - warp_offset - tid

            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
                if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]

        [<ReflectedDefinition>]
        member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            this.LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items)

        [<ReflectedDefinition>]
        static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
            {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
                real = real }

        [<ReflectedDefinition>]
        static member Default(real:RealTraits<'T>) =
            {   ITEMS_PER_THREAD = 128;
                real = real }


    type BlockLoadAlgorithm =
        | BLOCK_LOAD_DIRECT
        | BLOCK_LOAD_VECTORIZE
        //| BLOCK_LOAD_TRANSPOSE
        //| BLOCK_LOAD_WARP_TRANSPOSE

    type LoadSignatures<'T> =
        abstract Load : int * deviceptr<'T> * deviceptr<'T> -> unit
        abstract Load : int * deviceptr<'T> * deviceptr<'T> * int -> unit
        abstract Load : int * deviceptr<'T> * deviceptr<'T> * int * 'T -> unit

    
    let loadInternal (_ALGORITHM:BlockLoadAlgorithm) =
        fun (real:RealTraits<'T>) (items_per_thread:int) (block_threads:int option) ->
            match _ALGORITHM with
            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
                {   new LoadSignatures<'T> with
                        member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
                            (LoadDirectBlocked.Create(real, items_per_thread)).LoadDirectBlocked(linear_tid, block_itr, items)

                        member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
                            (LoadDirectBlocked.Create(real, items_per_thread)).LoadDirectBlocked(linear_tid, block_itr, items, valid_items)

                        member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
                            (LoadDirectBlocked.Create(real, items_per_thread)).LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default)  }
            
            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->
                {   new LoadSignatures<'T> with
                        member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
                            (LoadDirectBlockedVectorized.Create(real, items_per_thread)).LoadDirectBlockedVectorized(linear_tid, block_itr, items)

                        member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
                            //(LoadDirectBlockedVectorized.Create(real, items_per_thread)).LoadDirectBlockedVectorized(linear_tid, block_itr, items)
                            this.Load(linear_tid, block_itr, items)

                        member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
                            //(LoadDirectBlockedVectorized.Create(real, items_per_thread)).LoadDirectBlockedVectorized(linear_tid, block_itr, items)
                            this.Load(linear_tid, block_itr, items) }

            //| BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE
            //| BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE


//    [<Record>]
//    type LoadInternal<'T> =
//        {
//            mutable real : RealTraits<'T>
//            mutable ITEMS_PER_THREAD : int option
//            mutable BLOCK_THREADS : int option
//            mutable ALGORITHM : BlockLoadAlgorithm
//            mutable temp_storage : deviceptr<'T> option
//            mutable linear_tid : int option
//            mutable LoadDirectBlocked : LoadDirectBlocked<'T> option
//            mutable LoadDirectBlockedVectorized : LoadDirectBlockedVectorized<'T> option
//            mutable LoadDirectStriped : LoadDirectStriped<'T> option
//        }
//
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
//                this.LoadDirectBlocked <- LoadDirectBlocked.Create(this.real, this.ITEMS_PER_THREAD.Value) |> Some
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> 
//                this.LoadDirectBlockedVectorized <- LoadDirectBlockedVectorized.Create(this.real, this.ITEMS_PER_THREAD.Value) |> Some
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:int) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        static member inline Create(real:RealTraits<'T>, _ALGORITHM:BlockLoadAlgorithm, linear_tid:int) =
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
//        static member inline Create(real:RealTraits<'T>, _ALGORITHM:BlockLoadAlgorithm) =
//            {   real = real;
//                ITEMS_PER_THREAD = None;
//                BLOCK_THREADS = None;
//                ALGORITHM = _ALGORITHM;
//                temp_storage = None;
//                linear_tid = None;
//                LoadDirectBlocked = None;
//                LoadDirectStriped = None}

//
//    [<Record>]
//    type BlockLoad<'T> =
//        {
//            real : RealTraits<'T>
//            mutable BLOCK_THREADS      : int
//            mutable ITEMS_PER_THREAD   : int
//            mutable ALGORITHM          : BlockLoadAlgorithm
//            mutable WARP_TIME_SLICING  : bool
//            TempStorage : Expr<unit -> deviceptr<'T>> option
//            LoadInternal : LoadInternal<'T> option
//        }
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:int) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        static member Create(real:RealTraits<'T>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int, _ALGORITHM:BlockLoadAlgorithm, _WARP_TIME_SLICING:bool) =
//            {   real = real;
//                BLOCK_THREADS = _BLOCK_THREADS;
//                ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//                ALGORITHM = _ALGORITHM;
//                WARP_TIME_SLICING = _WARP_TIME_SLICING;
//                TempStorage = None;
//                LoadInternal = LoadInternal<'T>.Create(real, _ALGORITHM) |> Some}


//    type BlockLoad<'T>(?_ITEMS_PER_THREAD_:int,?BLOCK_THREADS:int) =
//        abstract LoadDirectBlocked : (int * deviceptr<'T> * deviceptr<'T>) -> unit
//        abstract LoadDirectBlocked : (int * deviceptr<'T> * deviceptr<'T> * int) -> unit
//        abstract LoadDirectBlocked : (int * deviceptr<'T> * deviceptr<'T> * int * 'T) -> unit
//        abstract LoadDirectBlockedVectorized : (int * deviceptr<'T> * deviceptr<'T>) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<'T> * deviceptr<'T>) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<'T> * deviceptr<'T> * int) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<'T> * deviceptr<'T> * int * 'T) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<'T> * deviceptr<'T>) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<'T> * deviceptr<'T> * int) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<'T> * deviceptr<'T> * int * 'T) -> unit

module RadixRank =
    let f() = "radix rank"
module RadixSort =
    let f() = "radix sort"
module RakingLayout =
    let f() = "raking layout"
module Reduce =
    let f() = "reduce"
module Scan =
    let f() = "scan"
module Shift =
    let f() = "shift"
module Store =
    let f() = "store"


module Specializations =

    module HistogramAtomic =
        let f() = "histogram atomic"

    module HistogramSort =
        let f() = "histogram sort"

    module ReduceRanking =
        let f() = "reduce ranking"

    module ReduceWarpReduction =
        let f() = "reduce warp reduction"

    module ScanRanking =
        let f() = "scan ranking"

    module ScanWarpScans =
        let f() = "scan warp scans"