module Test.Alea.CUDA.Extension.MGPU.Reduce

open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Reduce
open Alea.CUDA.Extension.MGPU.CTAScan
open NUnit.Framework

let worker = Engine.workers.DefaultWorker

let testReduce (op:IScanOp<'TI, 'TV, 'TR>) =
    let reduce = worker.LoadPModule(PArray.reduce op).Invoke

    fun (gold:'TI[] -> 'TV) (verify:'TV -> 'TV -> unit) (data:'TI[]) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! result = reduce data
            return! result.Value }

        let hOutput = gold data
        let dOutput = PCalc.run calc
        printfn "count(%d) h(%A) (d:%A)" data.Length hOutput dOutput
        verify hOutput dOutput

let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193]//; 9000; 10000; 2097152; 8388608; 33554432; 33554431; 33554433]

[<Test>]
let ``sum float``() =
    let op = scanOp ScanOpTypeAdd 0.0
    let gold data = data |> Array.sum
    let eps = 1e-8
    let verify (h:float) (d:float) = Assert.That(d, Is.EqualTo(h).Within(eps))
    let test = testReduce op gold verify

    sizes |> Seq.iter (fun count ->
        test (Array.init count (fun i -> float(i)))
        test (Array.init count (fun i -> -float(i)))
        test (let rng = Random(2) in Array.init count (fun _ -> rng.NextDouble() - 0.5)) )

[<Struct;Align(16)>]
type IterativeMean =
    val mean : float
    val count : float

    [<ReflectedDefinition>]
    new (mean, count) = { mean = mean; count = count }

    [<ReflectedDefinition>]
    new (mean) = { mean = mean; count = 1.0 }

    [<ReflectedDefinition>]
    static member ( + ) (lhs:IterativeMean, rhs:IterativeMean) =
        if lhs.count = 0.0 then rhs
        elif rhs.count = 0.0 then lhs
        else
            let count = lhs.count + rhs.count
            let mean = (lhs.count * lhs.mean + rhs.count * rhs.mean) / count
            IterativeMean(mean, count)

    [<ReflectedDefinition>]
    static member get_Zero() = IterativeMean()

    override this.ToString() = sprintf "(%f,%d)" this.mean (int(this.count))

[<Test>]
let ``mean float (iteratively)``() =
    let op = { new IScanOp<float, IterativeMean, float> with
                 member this.Commutative = 1
                 member this.Identity = CUDART_NAN
                 member this.HExtract = fun t index -> if index >= 0 then IterativeMean(t, 1.0) else IterativeMean(CUDART_NAN, 0.0)
                 member this.DExtract = <@ fun t index -> if index >= 0 then IterativeMean(t, 1.0) else IterativeMean(CUDART_NAN, 0.0) @>
                 member this.HPlus = ( + )
                 member this.DPlus = <@ ( + ) @>
                 member this.HCombine = failwith "not needed"
                 member this.DCombine = failwith "not needed" }
    let gold data = IterativeMean(data |> Array.average, float(data.Length))
    let eps = 1e-16
    let verify (h:IterativeMean) (d:IterativeMean) =
        Assert.That(d.mean, Is.EqualTo(h.mean).Within(eps))
        Assert.AreEqual(d.count, h.count)
    let test = testReduce op gold verify

    sizes |> Seq.iter (fun count ->
        test (Array.init count (fun i -> float(i)))
        test (Array.init count (fun i -> -float(i)))
        test (let rng = Random(2) in Array.init count (fun _ -> rng.NextDouble() - 0.5)) )

[<Test>] // uhmm, this is a little wrong, need check where it is
let ``mean float (non-iteratively)``() =
    let op = { new IScanOp<float, float, float> with
                 member this.Commutative = 1
                 member this.Identity = CUDART_NAN
                 member this.HExtract = fun t index -> if index >= 0 then t else CUDART_NAN
                 member this.DExtract = <@ fun t index -> if index >= 0 then t else CUDART_NAN @>
                 member this.HPlus =
                    fun t1 t2 ->
                        let t1' = DeviceFunction.__double_as_longlong(t1)
                        let t2' = DeviceFunction.__double_as_longlong(t2)
                        let nan = DeviceFunction.__double_as_longlong(CUDART_NAN)
                        if t1' = nan then t2
                        elif t2' = nan then t1
                        else t1 * 0.5 + t2 * 0.5
                 member this.DPlus =
                    <@ fun t1 t2 ->
                        let t1' = DeviceFunction.__double_as_longlong(t1)
                        let t2' = DeviceFunction.__double_as_longlong(t2)
                        let nan = DeviceFunction.__double_as_longlong(CUDART_NAN)
                        if t1' = nan then t2
                        elif t2' = nan then t1
                        else t1 * 0.5 + t2 * 0.5 @>
                 member this.HCombine = failwith "not needed"
                 member this.DCombine = failwith "not needed" }
    let gold data = data |> Array.average
    let eps = 1e-16
    let verify (h:float) (d:float) = Assert.That(d, Is.EqualTo(h).Within(eps))
    let test = testReduce op gold verify

    sizes |> Seq.iter (fun count ->
        test (Array.init count (fun i -> float(i)))
        test (Array.init count (fun i -> -float(i)))
        test (let rng = Random(2) in Array.init count (fun _ -> rng.NextDouble() - 0.5)) )


    