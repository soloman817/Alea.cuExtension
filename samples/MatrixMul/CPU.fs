module Sample.MatrixMul.CPU

let kernel (C:float32[]) (A:float32[]) (B:float32[]) (wA:int) (wB:int) =
    let hA = A.Length / wA
    for i = 0 to hA - 1 do
        for j = 0 to wB - 1 do
            let mutable sum = 0.0f
            for k = 0 to wA - 1 do
                let a = A.[i * wA + k]
                let b = B.[k * wB + j]
                sum <- sum + a * b
            C.[i * wB + j] <- sum

let calc () =
    fun (A:float32[]) (B:float32[]) (wA:int) (wB:int) ->
        let hA = A.Length / wA
        let C = Array.zeroCreate<float32> (hA * wB)
        kernel C A B wA wB
        C
