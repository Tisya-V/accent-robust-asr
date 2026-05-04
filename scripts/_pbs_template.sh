#!/bin/bash
# PBS Template for RDS HPC
# PBS directives format:
#PBS -N job_name                                    # Job name
#PBS -l select=1:ngpus=3:ncpus=12:mem=96gb          # 1 chunk, 3 GPUs, 12 CPUs, 96GB memory
#PBS -l walltime=12:00:00                           # 12 hour job
#PBS -o logs/job_name.out                           # stdout
#PBS -e logs/job_name.err                           # stderr
#PBS -j oe                                          # Join stdout/stderr (optional)

# This is a template. Copy and modify for actual job scripts.
