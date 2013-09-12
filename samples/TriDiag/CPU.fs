module Sample.TriDiag.CPU

/// Solves tridiagonal system according to   
/// G.E. Forsythe and C.B. Moler: Computer solutions to linear algebraic systems, Prentice-Hall, 1967
let solve (e:float[], d:float[], f:float[]) (b:float[]) =
    let n = e.Length
    let m = Array.zeroCreate n
    let u = Array.zeroCreate n
    let x = Array.zeroCreate n

    u.[0] <- d.[0]
    for i = 1 to n-1 do
        m.[i] <- e.[i]/u.[i-1]
        u.[i] <- d.[i] - m.[i]*f.[i-1]
        
    x.[0] <- b.[0]
    for i = 1 to n-1 do
        x.[i] <- b.[i] - m.[i]*x.[i-1]

    x.[n-1] <- x.[n-1]/u.[n-1]
    for i = n-2 downto 0 do
        x.[i] <- (x.[i] - f.[i]*x.[i+1])/u.[i] 
            
    x   

