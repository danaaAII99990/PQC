namespace QuantumROXHash {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Measurement;
    
    operation XORBlocks(block1 : Qubit[], block2 : Qubit[]) : Unit is Adj+Ctl {
        for i in 0..Length(block1)-1 {
            CNOT(block1[i], block2[i]);
        }
    }
    
    operation RotateLeft(register : Qubit[], shift : Int) : Unit is Adj+Ctl {
        let size = Length(register);
        // For each position, calculate new position after rotation
        for i in 0..size-1 {
            let newPos = (i + shift) % size;
            if i != newPos {  // Only SWAP if positions are different
                SWAP(register[i], register[newPos]);
            }
        }
    }
    
    operation CustomCompressionFunction(
        block1 : Qubit[], 
        block2 : Qubit[]
    ) : Unit is Adj+Ctl {
        let blockSize = Length(block1);
        
        // Quantum modular addition using CNOT gates
        for i in 0..blockSize-1 {
            CNOT(block1[i], block2[i]);
        }
        
        // Rotate left by 5 bits
        RotateLeft(block2, 5);
        
        // Final XOR
        XORBlocks(block1, block2);
    }
    
    operation ShuffleBytes(data : Qubit[]) : Unit is Adj+Ctl {
        let size = Length(data);
        // Modified Fisher-Yates shuffle that avoids self-swaps
        for i in size-1..-1..1 {
            let j = i - 1;  // Always swap with previous position to avoid self-swaps
            if i != j {     // Extra safety check
                SWAP(data[i], data[j]);
            }
        }
    }
    
    // Rest of the code remains the same...
    operation ApplyROXHashOracle(
        message : Qubit[],
        hashOutput : Qubit[],
        blockSize : Int
    ) : Unit is Adj+Ctl {
        let messageLength = Length(message);
        let numBlocks = messageLength / blockSize;
        
        use currentHash = Qubit[blockSize] {
            // Process each block
            for blockIndex in 0..numBlocks-1 {
                let startIdx = blockIndex * blockSize;
                let currentBlock = message[startIdx..startIdx + blockSize - 1];
                
                use (ro1Output, ro2Output) = (Qubit[blockSize], Qubit[blockSize]) {
                    // Create superposition for oracle simulation
                    for i in 0..blockSize-1 {
                        H(ro1Output[i]);
                        H(ro2Output[i]);
                    }
                    
                    // XOR oracle outputs
                    XORBlocks(ro1Output, ro2Output);
                    
                    // Apply transformations
                    XORBlocks(currentHash, ro2Output);
                    CustomCompressionFunction(currentHash, currentBlock);
                    
                    // Cleanup auxiliary qubits
                    for i in 0..blockSize-1 {
                        H(ro1Output[i]);
                        H(ro2Output[i]);
                    }
                }
            }
            
            // Final operations
            ShuffleBytes(currentHash);
            XORBlocks(currentHash, hashOutput);
        }
    }
    
    operation ROXHashPreimageSearch(target : Int[]) : Int[] {
        let blockSize = 5;
        let messageSize = blockSize * 2;
        
        mutable result = [0, size = messageSize];
        
        use (message, hashOutput) = (Qubit[messageSize], Qubit[blockSize]) {
            // Create initial superposition
            for i in 0..messageSize-1 {
                H(message[i]);
            }
            
            // Apply ROX hash oracle
            ApplyROXHashOracle(message, hashOutput, blockSize);
            
            // Grover iterations
            let iterations = Round(PI() * Sqrt(IntAsDouble(2^messageSize)) / 4.0);
            
            for i in 1..iterations {
                // Oracle phase kickback
                within {
                    for idx in 0..Length(hashOutput)-1 {
                        X(hashOutput[idx]);
                    }
                } apply {
                    Controlled Z(Most(hashOutput), Tail(hashOutput));
                }
                
                // Diffusion operator
                within {
                    for idx in 0..Length(message)-1 {
                        H(message[idx]);
                        X(message[idx]);
                    }
                } apply {
                    Controlled Z(Most(message), Tail(message));
                }
                
                for idx in 0..Length(message)-1 {
                    H(message[idx]);
                }
            }
            
            // Measure results
            for i in 0..messageSize-1 {
                set result w/= i <- ResultArrayAsInt([M(message[i])]);
            }
            
            ResetAll(message);
            ResetAll(hashOutput);
        }
        
        return result;
    }
    
    @EntryPoint()
    operation RunROXHashTest() : Unit {
        let target = [0, size = 64];  // Replace with actual hash target
        
        Message("Starting ROX Hash preimage search...");
        let preimage = ROXHashPreimageSearch(target);
        Message($"Found potential preimage: {preimage}");
    }
}