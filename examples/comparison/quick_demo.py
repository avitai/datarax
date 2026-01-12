# File: examples/comparison/quick_demo.py
"""Quick demonstration of the core difference between Datarax and Grain.

Datarax (stateful) and Grain (stateless) approaches.

Run this to immediately see why stateful is better!
"""

import flax.nnx as nnx
import numpy as np


print("=" * 60)
print("DATARAX vs GRAIN: CORE DIFFERENCE IN 100 LINES")
print("=" * 60)

# ============================================================
# GRAIN APPROACH: External State Management (Complex)
# ============================================================


def grain_iterate_data(data, batch_size, state):
    """Grain-style: Must pass and return state."""
    position = state["position"]

    # Get batch
    end = min(position + batch_size, len(data))
    batch = data[position:end]

    # Update state (must remember to do this!)
    new_state = {"position": end, "samples_seen": state["samples_seen"] + (end - position)}

    return batch, new_state  # Must return both!


# Using Grain approach - complex!
print("\n1. GRAIN APPROACH (Stateless):")
print("-" * 40)

data = np.arange(10)
state = {"position": 0, "samples_seen": 0}  # Manual state

# Process batches - must handle state manually
for i in range(3):
    batch, state = grain_iterate_data(data, 3, state)  # Pass state
    print(f"  Batch {i}: {batch}, state={state}")

print("\nProblems:")
print("  ✗ Must pass state to every function")
print("  ✗ Must remember to update state")
print("  ✗ Easy to forget state = ... assignment")
print("  ✗ State scattered across code")

# ============================================================
# DATARAX APPROACH: Internal State Management (Simple)
# ============================================================


class StatefulLoader(nnx.Module):
    """Datarax style: State managed internally."""

    def __init__(self, data, batch_size=3):
        self.data = data
        self.batch_size = batch_size

        # State as NNX Variables - automatic tracking!
        self.position = nnx.Variable(0)
        self.samples_seen = nnx.Variable(0)

    def get_batch(self):
        """Get batch - no state passing needed!"""
        end = min(self.position.value + self.batch_size, len(self.data))
        batch = self.data[self.position.value : end]

        # Update internal state automatically
        batch_size = end - self.position.value
        self.position.value = end
        self.samples_seen.value += batch_size

        return batch

    def reset(self):
        """Reset state - clean and simple."""
        self.position.value = 0


# Using Datarax approach - simple!
print("\n2. DATARAX APPROACH (Stateful):")
print("-" * 40)

loader = StatefulLoader(np.arange(10), batch_size=3)

# Process batches - no state management needed!
for i in range(3):
    batch = loader.get_batch()  # No state passing!
    print(f"  Batch {i}: {batch}, position={loader.position.value}")

print("\nAdvantages:")
print("  ✓ No state passing needed")
print("  ✓ State updates are automatic")
print("  ✓ Can't forget to update state")
print("  ✓ State encapsulated in module")
print("  ✓ Works with JAX transformations")

# Bonus: Automatic checkpointing!
print("\n3. BONUS - AUTOMATIC CHECKPOINTING:")
print("-" * 40)

# Get state (one line!)
checkpoint = {"position": loader.position.value, "samples": loader.samples_seen.value}
print(f"  Checkpoint: {checkpoint}")

# Reset and restore (clean!)
loader.reset()
print(f"  After reset: position={loader.position.value}")

loader.position.value = checkpoint["position"]
print(f"  After restore: position={loader.position.value}")

print("\n" + "=" * 60)
print("SUMMARY: Stateful is simpler, cleaner, and less error-prone!")
print("=" * 60)
