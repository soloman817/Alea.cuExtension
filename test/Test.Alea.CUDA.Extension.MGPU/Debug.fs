module Test.Alea.CUDA.Extension.MGPU.Debug
// NOT BEING USED FOR NOW
open System
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Intrinsics
open Alea.CUDA.Extension.MGPU.Static
//open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil

open NUnit.Framework


//let worker = getDefaultWorker()
//
//let rng = System.Random()
//
//type IScanOp<'TI, 'TV, 'TR> =
//    abstract Commutative : int
//    abstract Identity : 'TI // the init value
//    abstract HExtract : ('TI -> int -> 'TV)
//    abstract DExtract : Expr<'TI -> int -> 'TV>
//    abstract HPlus : ('TV -> 'TV -> 'TV)
//    abstract DPlus : Expr<'TV -> 'TV -> 'TV>
//    abstract HCombine : ('TI -> 'TV -> 'TR)
//    abstract DCombine : Expr<'TI -> 'TV -> 'TR>
//
//type ScanOpType =
//    | ScanOpTypeAdd
//    | ScanOpTypeMul
//    | ScanOpTypeMin
//    | ScanOpTypeMax
//
//let inline scanOp (opType:ScanOpType) (ident:'T) =
//    { new IScanOp<'T, 'T, 'T> with
//        member this.Commutative = 1
//        member this.Identity = ident
//        member this.HExtract = fun t index -> t
//        member this.DExtract = <@ fun t index -> t @>
//        member this.HPlus =
//            match opType with
//            | ScanOpTypeAdd -> ( + )
//            | ScanOpTypeMul -> ( * )
//            | ScanOpTypeMin -> min
//            | ScanOpTypeMax -> max
//        member this.DPlus =
//            match opType with
//            | ScanOpTypeAdd -> <@ ( + ) @>
//            | ScanOpTypeMul -> <@ ( * ) @>
//            | ScanOpTypeMin -> <@ min @>
//            | ScanOpTypeMax -> <@ max @>
//        member this.HCombine = fun t1 t2 -> t2 
//        member this.DCombine = <@ fun t1 t2 -> t2 @> }
//
//type IReduce<'TI, 'TV> =
//        {
//            NumBlocks : int // how may 'TV
//            Action : ActionHint -> DevicePtr<'TI> ->DevicePtr<'TV> -> unit
//            Result : 'TV[] -> 'TV
//        }
//
//
//type Plan =
//    {
//        NT : int
//        VT: int
//    }
//
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=16);Align(16)>]
//type Extent16 =
//    val dummy : byte
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=8);Align(8)>]
//type Extent8 =
//    val dummy : byte
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=4);Align(4)>]
//type Extent4 =
//    val dummy : byte
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=2);Align(2)>]
//type Extent2 =
//    val dummy : byte
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=1);Align(1)>]
//type Extent1 =
//    val dummy : byte
//
//let createSharedExpr (align:int) (size:int) =
//    let length = divup size align
//    match align with
//    | 16 -> <@ __shared__<Extent16>(length).Ptr(0).Reinterpret<byte>() @>
//    | 8  -> <@ __shared__<Extent8>(length).Ptr(0).Reinterpret<byte>() @>
//    | 4  -> <@ __shared__<Extent4>(length).Ptr(0).Reinterpret<byte>() @>
//    | 2  -> <@ __shared__<Extent2>(length).Ptr(0).Reinterpret<byte>() @>
//    | 1  -> <@ __shared__<Extent1>(length).Ptr(0).Reinterpret<byte>() @>
//    | _ -> failwithf "wrong align of %d" align
//
//[<Test>]
//let ``kernel test`` () =
//    let ctaReduce (NT:int) (op:IScanOp<'TI, 'TV, 'TR>) =
//        //let size = NT
//        //let capacity = NT + NT / WARP_SIZE
//        //let _, sLogPow2OfNT = sLogPow2 NT 1
//        let capacity = 0
//        //let plus = op.DPlus
//        let reduce =
//            <@ fun (tid:int) (x:'TV) (storage:RWPtr<'TV>) ->
//                //let plus = %plus
//                let mutable x = x
////                let dest = brev(uint32(tid)) >>> (32 - sLogPow2OfNT)
////                let dest = int(dest)
////                storage.[dest + dest / WARP_SIZE] <- x
////                __syncthreads()
////                let src = tid + tid / WARP_SIZE
////                let mutable destCount = NT / 2
////                while destCount >= 1 do
////                    if tid < destCount then
////                        if (NT / 2 = destCount) then x <- storage.[src]
////                        let src2 = destCount + tid
////                        x <- plus x storage.[src2 + src2 / WARP_SIZE]
////                        storage.[src] <- x
////                    __syncthreads()
////                    destCount <- destCount / 2        
//                let total = storage.[0]
//                //__syncthreads()
//                total @>
//        capacity, reduce
//
//    let kernelReduce (plan:Plan) (op:IScanOp<'TI, 'TV, 'TR>) =
//        let NT = plan.NT
//        let VT = plan.VT
//        let NV = NT * VT
//
//        let capacity, reduce = ctaReduce NT op
//        let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
//        let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
//        let sharedAlign = max alignOfTI alignOfTV
//        let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
//        let createSharedExpr = createSharedExpr sharedAlign sharedSize
//               
//        let commutative = op.Commutative
//        let identity = op.Identity
//        let extract = op.DExtract
//        let plus = op.DPlus
//        let deviceGlobalToReg = deviceGlobalToReg NT VT
//
//        <@ fun (data_global:DevicePtr<'TI>) (count:int) (task:int2) (reduction_global:DevicePtr<'TV>) ->
//            let extract = %extract
//            let plus = %plus
//            let deviceGlobalToReg = %deviceGlobalToReg
//            let reduce = %reduce
//
//            let shared = %(createSharedExpr)
//            let sharedReduce = shared.Reinterpret<'TV>()
//            let sharedInputs = shared.Reinterpret<'TI>()
//
//            let tid = threadIdx.x
//            let block = blockIdx.x
//            let first = VT * tid
//
//            let mutable range = computeTaskRangeEx block task NV count
//
//            let mutable total = extract identity -1
//            let mutable totalDefined = false
//
////            while range.x < range.y do
////                let count2 = min NV (count - range.x)
////                                
////                let inputs = __local__<'TI>(VT).Ptr(0)
////                deviceGlobalToReg count2 (data_global + range.x) tid inputs 0
////
////                if commutative <> 0 then
////                    for i = 0 to VT - 1 do
////                        let index = NT * i + tid
////                        if index < count2 then
////                            let x = extract inputs.[i] (range.x + index)
////                            total <- if i > 0 || totalDefined then plus total x else x
////                else                    
////                    ()
////
////                range.x <- range.x + NV
////                totalDefined <- true
////
//            if commutative <> 0 then                      // uncommenting this if statement causes abort
//                total <- reduce tid total sharedReduce    //
//
//            if tid = 0 then reduction_global.[block] <- total @>
//
//    
//    let reduce (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
//        //let cutOff = 20000
//        let plan1 = { NT = 512; VT = 5 }
//        let plan2 = { NT = 128; VT = 9 }
//        let! kernelReduce1 = kernelReduce plan1 op |> defineKernelFunc
//        //let! kernelReduce2 = kernelReduce plan2 op |> defineKernelFunc
//        let hplus = op.HPlus
//
//        return PFunc(fun (m:Module) ->
//            let worker = m.Worker
//            let kernelReduce1 = kernelReduce1.Apply m
//            //let kernelReduce2 = kernelReduce2.Apply m
//
//            fun (count:int) ->
//                let numBlocks, task, lp, kernelReduce =
//                    //if count < cutOff then
//                        printfn "Count < Cutoff"
//                        let plan = plan1
//                        let kernelReduce = kernelReduce1
//                        let NV = plan.NT * plan.VT
//                        let numTiles = divup count NV
//                        let numBlocks = 1
//                        let task = int2(numTiles, 1)
//                        let lp = LaunchParam(1, plan.NT)
//                        numBlocks, task, lp, kernelReduce
////                    else
////                        printfn "Count > Cutoff"
////                        let plan = plan2
////                        let kernelReduce = kernelReduce2
////                        let NV = plan.NT * plan.VT
////                        let numTiles = divup count NV
////                        let numBlocks = min (worker.Device.NumSm * 25) numTiles
////                        let task = divideTaskRange numTiles numBlocks
////                        let lp = LaunchParam(numBlocks, plan.NT)
////                        numBlocks, task, lp, kernelReduce
//
//                let action (hint:ActionHint) (data:DevicePtr<'TI>) (reduction:DevicePtr<'TV>) =
//                    let lp = lp |> hint.ModifyLaunchParam
//                    kernelReduce.Launch lp data count task reduction
//
//                let result (reduction:'TV[]) =
//                    reduction |> Array.reduce hplus
//
//                { NumBlocks = numBlocks; Action = action; Result = result } ) }
//        
//    let pReduce (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
//        let! api = reduce op
//        printfn "pReduce 1"
//        return PFunc(fun (m:Module) ->
//            //printfn "pReduce 2"
//            let worker = m.Worker
//            let api = api.Apply m
//            fun (data:DArray<'TI>) ->
//                pcalc {
//                    let count = data.Length
//                    ()
//                    //let api = api count
//                    //let! reduction = DArray.createInBlob worker api.NumBlocks
//                    //do! PCalc.action (fun hint -> api.Action hint data.Ptr reduction.Ptr)
//                    (*let result =
//                        fun () ->
//                            pcalc {
//                                let! reduction = reduction.Gather()
//                                return api.Result reduction }
//                        |> Lazy.Create*)
//                    (*return result*)} ) }
//
//    let testReduce (op:IScanOp<'TI, 'TV, 'TR>) =
//        let reduce = worker.LoadPModule(pReduce op).Invoke
//
//        fun (gold:'TI[] -> 'TV) (*(verify:'TV -> 'TV -> unit)*) (data:'TI[]) ->
//            let calc = pcalc {
//                let! data = DArray.scatterInBlob worker data
//                let! result = reduce data 
//                return 0 }
//
//            let hOutput = gold data
//            let dOutput = PCalc.run calc
//            printfn "count(%d) h(%A) (d:%A)" data.Length hOutput dOutput
//            //verify hOutput dOutput
//
//    let sizes = [12; 128; 512] //; 1024; 1200; 4096; 5000; 8191; 8192; 8193]//; 9000; 10000; 2097152; 8388608; 33554432; 33554431; 33554433]
//
//    
//    let op = scanOp ScanOpTypeAdd 0.0
//    let gold data = data |> Array.sum
//    //let eps = 1e-8
//    //let verify (h:float) (d:float) = Assert.That(d, Is.EqualTo(h).Within(eps))
//    let test = testReduce op gold //verify
//
//    sizes |> Seq.iter (fun count ->
//        test (Array.init count (fun i -> float(i)))
//        test (Array.init count (fun i -> -float(i)))
//        test (let rng = Random(2) in Array.init count (fun _ -> rng.NextDouble() - 0.5)) )