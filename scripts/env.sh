#!/bin/bash
# Backward compatibility wrapper — source pbs_env.sh
# This file is kept for backward compatibility with existing scripts.
# New scripts should source scripts/pbs_env.sh directly.

source "$(dirname "${BASH_SOURCE[0]}")/pbs_env.sh"
