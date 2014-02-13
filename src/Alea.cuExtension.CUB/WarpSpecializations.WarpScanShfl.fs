[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl

open System    

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Utilities



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
            
//type TemplateParameters =
//    {
//        LOGICAL_WARPS           : int
//        LOGICAL_WARP_THREADS    : int
//    }

type Constants =
    {
        STEPS   : int
        SHFL_C  : int
    }

    static member Init(logical_warp_threads:int) =
        let steps = logical_warp_threads |> log2
        {
            STEPS = steps
            SHFL_C = ((-1 <<< steps) &&& 31) <<< 8
        }


[<Record>]
type ThreadFields =
    {
        mutable warp_id : int
        mutable lane_id : int
    }

    static member Init(warp_id, lane_id) =
        {
            warp_id = warp_id
            lane_id = lane_id
        }


[<Record>]
type WarpScanShfl<'T> =
    {
        LOGICAL_WARPS        : int
        LOGICAL_WARP_THREADS : int 
        Constants           : Constants
        ThreadFields        : ThreadFields
    }

    member this.Initialize(temp_storage, warp_id, lane_id) =
        this.ThreadFields.warp_id <- warp_id
        this.ThreadFields.lane_id <- lane_id
        this

    /// Broadcast
    member this.Broadcast(input:'T,src_lane:int) =
        ShuffleBroadcast(input, src_lane, this.LOGICAL_WARP_THREADS)


    //---------------------------------------------------------------------
    // Inclusive operations
    //---------------------------------------------------------------------

    /// Inclusive prefix sum with aggregate (single-SHFL)
    member this.InclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>, ?single_shfl:bool) =
        let LOGICAL_WARP_THREADS = this.LOGICAL_WARP_THREADS
        let STEPS = this.Constants.STEPS
        let temp : uint32 = input |> __obj_reinterpret

        // Iterate scan steps
        for STEP = 0 to STEPS - 1 do ()
            // Use predicate set from SHFL to guard against invalid peers
//            asm(
//                "{"
//                "  .reg .u32 r0;"
//                "  .reg .pred p;"
//                "  shfl.up.b32 r0|p, %1, %2, %3;"
//                "  @p add.u32 r0, r0, %4;"
//                "  mov.u32 %0, r0;"
//                "}"
//                : "=r"(temp) : "r"(temp), "r"(1 << STEP), "r"(SHFL_C), "r"(temp));
        
        let temp = temp |> __obj_reinterpret
        output := temp

        // Grab aggregate from last warp lane
        warp_aggregate := this.Broadcast(!output, LOGICAL_WARP_THREADS - 1)


    /// Inclusive prefix sum with aggregate (multi-SHFL)
    member this.InclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>) = //, single_shfl)        ///< [in] Marker type indicating whether only one SHFL instruction is required
        // Delegate to generic scan
        this.InclusiveScan(input, output, (+), warp_aggregate)


    /// Inclusive prefix sum with aggregate (specialized for float)
    member this.InclusiveSum(input:float32, output:Ref<float32>, warp_aggregate:Ref<float32>) =
        
        output = input

        // Iterate scan steps
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Use predicate set from SHFL to guard against invalid peers
            asm(
                "{"
                "  .reg .f32 r0;"
                "  .reg .pred p;"
                "  shfl.up.b32 r0|p, %1, %2, %3;"
                "  @p add.f32 r0, r0, %4;"
                "  mov.f32 %0, r0;"
                "}"
                : "=f"(output) : "f"(output), "r"(1 << STEP), "r"(SHFL_C), "f"(output));
        }

        // Grab aggregate from last warp lane
        warp_aggregate = Broadcast(output, LOGICAL_WARP_THREADS - 1);
    }


    /// Inclusive prefix sum with aggregate (specialized for unsigned long long)
    member this.InclusiveSum(
        unsigned long long  input,              ///< [in] Calling thread's input item.
        unsigned long long  &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        unsigned long long  &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        output = input;

        // Iterate scan steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Use predicate set from SHFL to guard against invalid peers
            asm(
                "{"
                "  .reg .u32 r0;"
                "  .reg .u32 r1;"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.up.b32 r0|p, lo, %2, %3;"
                "  shfl.up.b32 r1|p, hi, %2, %3;"
                "  @p add.cc.u32 r0, r0, lo;"
                "  @p addc.u32 r1, r1, hi;"
                "  mov.b64 %0, {r0, r1};"
                "}"
                : "=l"(output) : "l"(output), "r"(1 << STEP), "r"(SHFL_C));
        }

        // Grab aggregate from last warp lane
        warp_aggregate = Broadcast(output, LOGICAL_WARP_THREADS - 1);
    }


    /// Inclusive prefix sum with aggregate (generic)
    template <typename _T>
    member this.InclusiveSum(
        _T               input,             ///< [in] Calling thread's input item.
        _T               &output,           ///< [out] Calling thread's output item.  May be aliased with \p input.
        _T               &warp_aggregate)   ///< [out] Warp-wide aggregate reduction of input items.
    {
        // Whether sharing can be done with a single SHFL instruction (vs multiple SFHL instructions)
        Int2Type<(Traits<_T>::PRIMITIVE) && (sizeof(_T) <= sizeof(unsigned int))> single_shfl;

        InclusiveSum(input, output, warp_aggregate, single_shfl);
    }


    /// Inclusive prefix sum
    member this.InclusiveSum(
        T               input,              ///< [in] Calling thread's input item.
        T               &output)            ///< [out] Calling thread's output item.  May be aliased with \p input.
    {
        T warp_aggregate;
        InclusiveSum(input, output, warp_aggregate);
    }


    /// Inclusive scan with aggregate
    template <typename ScanOp>
    member this.InclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        output = input;

        // Iterate scan steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            // Grab addend from peer
            const int OFFSET = 1 << STEP;
            T temp = ShuffleUp(output, OFFSET);

            // Perform scan op if from a valid peer
            if (lane_id >= OFFSET)
                output = scan_op(temp, output);
        }

        // Grab aggregate from last warp lane
        warp_aggregate = Broadcast(output, LOGICAL_WARP_THREADS - 1);
    }


    /// Inclusive scan
    template <typename ScanOp>
    member this.InclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        T warp_aggregate;
        InclusiveScan(input, output, scan_op, warp_aggregate);
    }


    //---------------------------------------------------------------------
    // Exclusive operations
    //---------------------------------------------------------------------

    /// Exclusive scan with aggregate
    template <typename ScanOp>
    member this.ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        // Compute inclusive scan
        T inclusive;
        InclusiveScan(input, inclusive, scan_op, warp_aggregate);

        // Grab result from predecessor
        T exclusive = ShuffleUp(inclusive, 1);

        output = (lane_id == 0) ?
            identity :
            exclusive;
    }


    /// Exclusive scan
    template <typename ScanOp>
    member this.ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        T warp_aggregate;
        ExclusiveScan(input, output, identity, scan_op, warp_aggregate);
    }


    /// Exclusive scan with aggregate, without identity
    template <typename ScanOp>
    member this.ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        // Compute inclusive scan
        T inclusive;
        InclusiveScan(input, inclusive, scan_op, warp_aggregate);

        // Grab result from predecessor
        output = ShuffleUp(inclusive, 1);
    }


    /// Exclusive scan without identity
    template <typename ScanOp>
    member this.ExclusiveScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &output,            ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary scan operator
    {
        T warp_aggregate;
        ExclusiveScan(input, output, scan_op, warp_aggregate);
    }

    static member Create(tp:TemplateParameters) = //temp_storage, warp_id, lane_id) =
        let c = tp.LOGICAL_WARP_THREADS |> Constants.Init
        {
            TemplateParameters = tp
            Constants = c
            ThreadFields = (0,0) |> ThreadFields.Init
        }

