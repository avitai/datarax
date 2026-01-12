"""Check GPU availability and configuration for JAX."""

import jax


print("JAX version:", jax.__version__)
print("Available devices:")
print(jax.devices())

print("\nGPU devices:")
try:
    gpu_devices = jax.devices("gpu")
    print(gpu_devices)
    print("CUDA available:", len(gpu_devices) > 0)
except Exception as e:
    print("Error:", e)
