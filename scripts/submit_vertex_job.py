#!/usr/bin/env python
"""
Submit Datarax tests as a Vertex AI Custom Job.

Usage:
    python scripts/submit_vertex_job.py --image_uri gcr.io/my-project/datarax:latest

Configuration is loaded from 'vertex_config.yaml' if present.
"""

import argparse
import yaml
import os
import sys
from google.cloud import aiplatform


def load_config(config_path="vertex_config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def submit_job(args, config):
    # Merge args and config
    project_id = args.project_id or config.get("project_id")
    location = args.location or config.get("location", "us-central1")
    staging_bucket = args.staging_bucket or config.get("staging_bucket")
    image_uri = args.image_uri or config.get("image_uri")

    if not image_uri:
        print("Error: image_uri must be provided via --image_uri or vertex_config.yaml")
        sys.exit(1)

    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)

    # Configure Worker Pool
    # For distributed training using JAX, we typically use replica_count > 1
    # and all nodes run the same image.

    machine_type = args.machine_type or config.get("machine_type", "g2-standard-4")
    # g2-standard-4 has 1 L4 GPU.
    # a2-highgpu-1g has 1 A100 GPU.

    accelerator_type = args.accelerator_type or config.get("accelerator_type", "NVIDIA_L4")
    accelerator_count = int(args.accelerator_count or config.get("accelerator_count", 1))
    replica_count = int(args.replica_count or config.get("replica_count", 1))

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": replica_count,
            "container_spec": {
                "image_uri": image_uri,
                "command": [],  # Uses Docker ENTRYPOINT
                "args": args.test_args,  # Pass test args like ["tests/benchmarks"]
                "env": [
                    {"name": "JAX_PLATFORM_NAME", "value": "cuda"},
                    # Add any other required env vars here
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=args.display_name or "datarax-test-distributed",
        worker_pool_specs=worker_pool_specs,
    )

    print(f"Submitting job '{job.display_name}' to Vertex AI...")
    print(
        f"Worker Pool: {replica_count} x {machine_type} ({accelerator_count} x {accelerator_type})"
    )

    job.run(sync=True, service_account=config.get("service_account"))

    print("Job finished.")


def main():
    parser = argparse.ArgumentParser(description="Submit Vertex AI Job")
    parser.add_argument("--image_uri", help="Docker image URI (e.g. gcr.io/prj/img:tag)")
    parser.add_argument("--project_id", help="GCP Project ID")
    parser.add_argument("--location", help="GCP Region")
    parser.add_argument("--staging_bucket", help="GCS Bucket for staging")
    parser.add_argument("--display_name", help="Job display name")
    parser.add_argument("--machine_type", help="Machine type (default: g2-standard-4)")
    parser.add_argument("--accelerator_type", help="Accelerator type (default: NVIDIA_L4)")
    parser.add_argument("--accelerator_count", type=int, help="Number of GPUs per VM")
    parser.add_argument("--replica_count", type=int, help="Number of worker replicas (default: 1)")
    parser.add_argument("test_args", nargs=argparse.REMAINDER, help="Arguments to pass to pytest")

    args = parser.parse_args()
    config = load_config()

    submit_job(args, config)


if __name__ == "__main__":
    main()
