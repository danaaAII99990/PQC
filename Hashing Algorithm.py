import os
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
import random

class SharedState:
    def __init__(self):
        self.unique_inputs = set()  # Set of all unique inputs
        self.entropy = 0.0  # Track dynamic entropy
        self.state_history = []  # Track the history of states for analysis
    
    def register_input(self, input_data, oracle_output):
        """Add a unique input to the shared state and track entropy."""
        hashed_input = hashlib.sha256(input_data).digest()
        if hashed_input not in self.unique_inputs:
            self.unique_inputs.add(hashed_input)
            # Update entropy dynamically based on input and output
            self.entropy = self.calculate_dynamic_entropy(oracle_output)

    def calculate_dynamic_entropy(self, oracle_output):
        """Simulate entropy change with each new oracle output."""
        W = len(self.unique_inputs)  # Number of unique inputs
        randomness = sum(oracle_output) / len(oracle_output)  # A simple measure of randomness
        dynamic_entropy = math.log(W + 1) * randomness  # Update entropy based on randomness
        self.state_history.append(dynamic_entropy)
        return dynamic_entropy

    def get_microstate(self):
        """Return a dynamic representation of the microstate (entropy)."""
        return self.entropy


class RandomOracle:
    def __init__(self, shared_state, output_size=32):
        self.oracle_data = {}
        self.output_size = output_size
        self.shared_state = shared_state

    def query(self, input_data):
        """Generate random output for each input."""
        random_output = os.getrandom(self.output_size)
        hashed_input = hashlib.sha256(input_data).digest()

        if hashed_input not in self.oracle_data:
            self.oracle_data[hashed_input] = random_output

        # Register input with the shared state
        self.shared_state.register_input(input_data, random_output)
        return random_output


def pad_message(message, block_size):
    padding_length = block_size - (len(message) % block_size)
    return message + b'\x80' + b'\x00' * (padding_length - 1)

def xor_blocks(block1, block2):
    return bytes(a ^ b for a, b in zip(block1, block2))

def rotate_left(value, shift, size=32):
    """Performs a left rotation on a given value."""
    return ((value << shift) & (2**size - 1)) | (value >> (size - shift))

def custom_compression_function(block1, block2):
    if len(block1) != len(block2):
        raise ValueError("Blocks must be of the same length")

    block1_int = int.from_bytes(block1, byteorder='big')
    block2_int = int.from_bytes(block2, byteorder='big')

    # Step 1: Modular addition of the two blocks
    combined = (block1_int + block2_int) % (2 ** (8 * len(block1)))

    # Step 2: Rotate left by 5 bits
    rotated_value = rotate_left(combined, 5, size=8 * len(block1))

    # Step 3: XOR the rotated value with the second block integer
    xor_result = rotated_value ^ block2_int

    # Step 4: Apply bitwise AND with a mask (to limit value size)
    mask = (2 ** (8 * len(block1))) - 1
    and_result = xor_result & mask

    return and_result.to_bytes(len(block1), byteorder='big')

# XOR multiple outputs from oracles
def xor_multiple_outputs(*outputs):
    """XOR multiple oracle outputs together to increase entropy."""
    result = outputs[0]
    for output in outputs[1:]:
        result = bytes(a ^ b for a, b in zip(result, output))
    return result

# Shuffle the bytes to further reduce serial correlation
def shuffle_bytes(data):
    """Shuffle the bytes to further randomize the data."""
    byte_list = list(data)
    random.shuffle(byte_list)
    return bytes(byte_list)


def rox_hash_with_dynamic_disorder_monitoring(message, block_size=64):
    shared_state = SharedState()  # Create shared state for both oracles
    ro1 = RandomOracle(shared_state, output_size=64)
    ro2 = RandomOracle(shared_state, output_size=64)
    
    padded_message = pad_message(message, block_size)
    current_hash = b'\x00' * block_size
    blocks = [padded_message[i:i + block_size] for i in range(0, len(padded_message), block_size)]

    block_disorder = []  # To store disorder observations
    
    with ThreadPoolExecutor() as executor:
        for block_index, block in enumerate(blocks):
            print(f"Original Block {block_index}: {block.hex()}")

            # Parallel oracle queries
            ro1_future = executor.submit(ro1.query, block + current_hash)
            ro2_future = executor.submit(ro2.query, block + current_hash)
            
            ro1_output = ro1_future.result()
            ro2_output = ro2_future.result()
            
            print(f"RO1 Output for Block {block_index}: {ro1_output.hex()}")
            print(f"RO2 Output for Block {block_index}: {ro2_output.hex()}")
            
            # XOR multiple oracle outputs
            mixed_output = xor_multiple_outputs(ro1_output, ro2_output)
            
            # Apply transformations
            disordered_hash = xor_blocks(current_hash, mixed_output)
            disordered_hash = custom_compression_function(disordered_hash, block)

            print(f"Transformed Block {block_index}: {disordered_hash.hex()}")
            block_disorder.append((block, disordered_hash))

            # Update the current hash for the next iteration
            current_hash = disordered_hash

    # Shuffle the final hash output to further randomize it
    final_hash = shuffle_bytes(current_hash)
    print(f"Shuffled Final Hash: {final_hash.hex()}")

    # Calculate dynamic entropy
    print("\nDynamic Entropy Analysis:")
    entropy_ro1 = ro1.shared_state.get_microstate()
    entropy_ro2 = ro2.shared_state.get_microstate()
    print(f"RO1 Dynamic Entropy: {entropy_ro1} (based on {shared_state.get_microstate()} unique inputs)")
    print(f"RO2 Dynamic Entropy: {entropy_ro2} (based on {shared_state.get_microstate()} unique inputs)")

    # Log summary of disorder analysis
    print("\nDisorder Analysis:")
    for i, (original, transformed) in enumerate(block_disorder):
        disorder = sum(1 for a, b in zip(original, transformed) if a != b)
        print(f"Block {i} Disorder: {disorder} bytes changed")
    
    return final_hash


# Test the function
message = b"Hello, Quantum World!" * 128
rox_hash_output = rox_hash_with_dynamic_disorder_monitoring(message)
print(f"\nROX Hash Output: {rox_hash_output.hex()}")

altered_message = b"A" * 128
rox_hash_output = rox_hash_with_dynamic_disorder_monitoring(altered_message)
print(f"\nROX Hash Output: {rox_hash_output.hex()}")

# Differential Analysis
def differential_analysis():
    message = os.urandom(64)
    altered_message = bytearray(message)
    altered_message[0] ^= 0x01  # Flip a single bit

    original_hash = rox_hash_with_dynamic_disorder_monitoring(message)
    altered_hash = rox_hash_with_dynamic_disorder_monitoring(altered_message)

    # Compare the hashes
    differences = sum(1 for a, b in zip(original_hash, altered_hash) if a != b)
    print(f"Input Difference: 1 bit flipped")
    print(f"Hash Differences: {differences} bytes different")

differential_analysis()

