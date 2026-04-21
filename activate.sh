#!/usr/bin/env bash

# shellcheck disable=SC2317
_datarax_activate_die() {
    echo "error: $*" >&2
    return 1 2>/dev/null || exit 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    _datarax_activate_die "use 'source ./activate.sh' so the environment stays active"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE_SCRIPT="$PROJECT_ROOT/.venv/bin/activate"
MANAGED_ENV_FILE="${DATARAX_MANAGED_ENV_FILE:-$PROJECT_ROOT/.datarax.env}"

if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    _datarax_activate_die "virtual environment not found; run ./setup.sh first"
fi

_datarax_reset_previous_managed_env() {
    local variable
    for variable in ${DATARAX_MANAGED_ENV_VARS:-}; do
        unset "$variable"
    done
    unset DATARAX_MANAGED_ENV_VARS
}

# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

_datarax_reset_previous_managed_env

if [[ -f "$MANAGED_ENV_FILE" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "$MANAGED_ENV_FILE"
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "$PROJECT_ROOT/.env"
fi

if [[ -f "$PROJECT_ROOT/.env.local" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "$PROJECT_ROOT/.env.local"
fi

echo "Datarax environment active (${DATARAX_BACKEND:-auto})."
echo "Use 'uv run python scripts/verify_datarax_gpu.py' to inspect the active JAX backend."
