namespace GroverSearch {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Diagnostics;

    // Function to compute Boltzmann entropy with a dynamic microstates calculation and probability factor
    operation BoltzmannEntropy(input: Int, nQubits: Int) : Double {
        let k = 1.380649e-23; // Boltzmann constant
        let W = 2^nQubits; // Dynamic microstates based on the number of qubits
        let probability = 1.0 / IntAsDouble(W); // Uniform probability distribution
        return k * Log(IntAsDouble(W)) * probability; // Adjusted Boltzmann entropy calculation
    }

    // Define the oracle for Grover's algorithm
    operation Oracle(target: Qubit[], targetEntropy: Double, nQubits: Int) : Unit {
        let n = Length(target);
        mutable foundMatch = false;

        for i in 0..(2^n - 1) {
            // Check the entropy and set flag if a match is found
            if (BoltzmannEntropy(i, nQubits) == targetEntropy) {
                set foundMatch = true;
            }
        }

        // Only flip qubit if a match was found
        if (foundMatch) {
            X(target[0]); // Flip one designated qubit to indicate a match
            Message("Match found for target entropy, oracle applied.");
        } else {
            Message("No match found for target entropy, oracle not applied.");
        }
    }

    // Operation to display qubit states as a binary string
    operation DisplayQubitStatesAsBinary(target: Qubit[]) : String {
        mutable stateString = "";
        for qubit in target {
            let bit = Measure([PauliZ], [qubit]);  // Measure in the Z-basis
            set stateString += (bit == One) ? "1" | "0";
            // Reset qubit to |0⟩ if measured as |1⟩ to avoid state disturbance
            if bit == One {
                X(qubit);
            }
        }
        return stateString;
    }

    // Grover's search operation with iterative target entropy adjustment
    operation GroverSearch(targetEntropy: Double, nQubits: Int) : Result {
        use target = Qubit[nQubits];

        // Prepare qubits in superposition
        ApplyToEach(H, target);

        // Grover's iterations (updated based on n)
        let iterations = Round(PI() * Sqrt(IntAsDouble(2^nQubits)) / 4.0);
        mutable dynamicTargetEntropy = targetEntropy;

        // Grover's search loop
        for iter in 0..iterations - 1 {
            // Apply the oracle with the current target entropy
            Oracle(target, dynamicTargetEntropy, nQubits);

            // Display qubit states after oracle
            let stateAfterOracle = DisplayQubitStatesAsBinary(target);
            Message($"Iteration {iter + 1} - State after Oracle: {stateAfterOracle}");

            // Apply the diffusion operator
            ApplyToEach(H, target);
            ApplyToEach(X, target);
            Z(target[0]);
            ApplyToEach(X, target);
            ApplyToEach(H, target);

            // Display qubit states after diffusion operator
            let stateAfterDiffusion = DisplayQubitStatesAsBinary(target);
            Message($"Iteration {iter + 1} - State after Diffusion: {stateAfterDiffusion}");

            // Adjust the dynamic target entropy slightly with each iteration
            set dynamicTargetEntropy += 1e-23; // Small increment to adjust entropy target
        }

        // Final measurement to get result
        DumpMachine();
        let result = MeasureAllZ(target);
        ResetAll(target);
        return result;
    }

    // Entry point to the program
    @EntryPoint()
    operation Main() : Unit {
        let nQubits = 5;
        let targetEntropy = 1.9139859233858158e-23; // Initial target entropy
        let preimage = GroverSearch(targetEntropy, nQubits);
        Message($"Preimage found: {preimage}");
    }
}
