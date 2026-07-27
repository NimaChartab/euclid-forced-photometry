#!/usr/bin/env bash
#
# install.sh — set up the euclid_phot environment in one command.
#
# `pip install -e .` already pulls in every PyPI dependency (they are declared
# in pyproject.toml). This script additionally installs the three packages the
# README installs by hand, because none of them is a plain `pip install`:
#
#   1. Tractor          — needs `pip install --no-build-isolation` with numpy +
#                         cython already present (its setup.py imports numpy at
#                         build time and builds the _mp_fourier C extension).
#   2. astrometry.net   — not pip-installable (its setup.py defers to `make`);
#                         the pure-Python modules Tractor needs are exposed via
#                         a source clone placed on a .pth path.
#   3. unwise_psf        — not packaged; same .pth trick. Optional (--wise),
#                         only needed for the unWISE W1/W2 leg, with fitsio and
#                         setuptools<81 alongside it.
#
# Safe to re-run: every step checks whether it is already done.
#
# Usage:
#   ./scripts/install.sh            # VIS + NISP
#   ./scripts/install.sh --wise     # also the optional unWISE W1/W2 leg
#   ./scripts/install.sh --python /path/to/python   # install into a given env
#   ./scripts/install.sh --help
#
set -euo pipefail

WANT_WISE=0
PY_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wise) WANT_WISE=1; shift ;;
        --python) PY_OVERRIDE="${2:-}"; shift 2 ;;
        -h|--help)
            awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"
            exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- Locate the repo root (this script lives in scripts/) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "could not find pyproject.toml near $SCRIPT_DIR" >&2
    exit 1
fi
cd "$REPO_ROOT"

# --- Pick the Python interpreter to install into -----------------------------
# Priority: explicit --python > active venv > active conda env > a fresh .venv.
if [[ -n "$PY_OVERRIDE" ]]; then
    PY="$PY_OVERRIDE"
    log "Using interpreter from --python: $PY"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
    log "Using the active virtualenv: $VIRTUAL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    PY="$CONDA_PREFIX/bin/python"
    log "Using the active conda env: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
    if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
        echo "  note: installing into conda 'base'. Consider 'conda create -n euclid python=3.11' first."
    fi
else
    if [[ ! -d "$REPO_ROOT/.venv" ]]; then
        command -v python3 >/dev/null || { echo "python3 not found on PATH" >&2; exit 1; }
        log "No environment active; creating .venv"
        python3 -m venv "$REPO_ROOT/.venv"
    else
        log "No environment active; reusing existing .venv"
    fi
    PY="$REPO_ROOT/.venv/bin/python"
    CREATED_VENV=1
fi

"$PY" -c 'import sys; print("    python", sys.version.split()[0], "->", sys.executable)'
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
    echo "euclid_phot needs Python >= 3.10, but the selected interpreter is $("$PY" -V 2>&1)." >&2
    echo "Create a newer environment first, e.g.:  conda create -n euclid python=3.11" >&2
    echo "then re-run, or pass --python /path/to/a/3.10+/python." >&2
    exit 1
fi

# Write a .pth file into the env's site-packages so a source clone is importable.
# $1 = .pth filename, $2 = directory to add to the import path.
write_pth() {
    "$PY" - "$1" "$2" <<'PYEOF'
import sys, sysconfig, pathlib
name, target = sys.argv[1], sys.argv[2]
sp = pathlib.Path(sysconfig.get_path("purelib"))
sp.mkdir(parents=True, exist_ok=True)
(sp / name).write_text(str(pathlib.Path(target).resolve()) + "\n")
print(f"    {sp / name} -> {pathlib.Path(target).resolve()}")
PYEOF
}

has_module() { "$PY" -c "import $1" >/dev/null 2>&1; }

# Keep a vendored clone pristine: importing it generates __pycache__, which
# would show as changes in IDEs that auto-detect nested git repos. The
# clone's .git/info/exclude is a local-only ignore, so nothing upstream is
# touched. No-op if the pattern is already there.
exclude_pycache() {
    local exclude="$1/.git/info/exclude"
    [[ -d "$1/.git" ]] || return 0
    grep -qxF "__pycache__/" "$exclude" 2>/dev/null \
        || echo "__pycache__/" >> "$exclude"
}

# Tractor's build runs SWIG to generate its C-extension wrappers (the
# _mp_fourier FFT module). SWIG is a build tool, not a pip dependency of
# Tractor, so ensure it is present before the build. The official `swig`
# wheel on PyPI works in any environment (venv/conda/system) with no system
# package manager, so try that first; fall back to conda/brew/apt.
ensure_swig() {
    if command -v swig >/dev/null 2>&1; then
        return
    fi
    log "SWIG (needed to build Tractor) was not found; installing it"
    # 1. Universal path: the `swig` wheel drops a binary in the env's bin dir.
    if "$PY" -m pip install swig >/dev/null 2>&1; then
        local scripts_dir
        scripts_dir="$("$PY" -c "import sysconfig; print(sysconfig.get_path('scripts'))")"
        export PATH="$scripts_dir:$PATH"   # so the Tractor build sees it
    fi
    if command -v swig >/dev/null 2>&1; then
        echo "    installed swig via pip -> $(command -v swig)"
        return
    fi
    # 2. Fallbacks if no wheel is available for this platform.
    local env_prefix
    env_prefix="$(cd "$(dirname "$PY")/.." && pwd)"
    if [[ -d "$env_prefix/conda-meta" ]] && command -v conda >/dev/null 2>&1; then
        conda install -y -p "$env_prefix" -c conda-forge swig
        export PATH="$env_prefix/bin:$PATH"
    elif command -v brew >/dev/null 2>&1; then
        brew install swig
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y swig
    else
        cat >&2 <<'EOF'
    SWIG was not found and could not be installed automatically. Install it,
    then re-run this script:
        pip install swig                     (usually works anywhere)
        conda install -c conda-forge swig
        brew install swig
        sudo apt-get install swig
EOF
        exit 1
    fi
    command -v swig >/dev/null 2>&1 \
        || { echo "swig still not on PATH after install" >&2; exit 1; }
}

# --- 1. The package + its PyPI dependencies (incl. numpy, jupyter) -----------
log "Installing euclid_phot and its PyPI dependencies"
"$PY" -m pip install -e ".[dev]"

# --- 2. Tractor (needs cython, then --no-build-isolation) --------------------
if has_module tractor; then
    log "Tractor already installed; skipping"
else
    log "Installing Tractor (with cython, --no-build-isolation)"
    ensure_swig
    # --no-build-isolation uses the env's own build backend, so setuptools and
    # wheel must be present (Python 3.12+ venvs no longer ship setuptools).
    # cython + numpy (a euclid_phot dep, already installed) build _mp_fourier.
    "$PY" -m pip install setuptools wheel cython
    "$PY" -m pip install --no-build-isolation \
        "git+https://github.com/dstndstn/tractor.git"
fi

# --- 3. astrometry.net pure-Python utilities (source clone on a .pth) --------
if has_module astrometry.util.fits; then
    log "astrometry.net utilities already importable; skipping"
else
    log "Setting up astrometry.net utilities"
    if [[ ! -d "$REPO_ROOT/.anet/astrometry" ]]; then
        mkdir -p "$REPO_ROOT/.anet"
        git clone --depth 1 https://github.com/dstndstn/astrometry.net.git \
            "$REPO_ROOT/.anet/astrometry"
    fi
    exclude_pycache "$REPO_ROOT/.anet/astrometry"
    write_pth astrometry_net.pth "$REPO_ROOT/.anet"
    has_module astrometry.util.fits \
        || { echo "astrometry.util.fits still not importable after setup" >&2; exit 1; }
fi

# --- 4. unWISE PSF leg (optional) --------------------------------------------
if [[ "$WANT_WISE" -eq 1 ]]; then
    if has_module unwise_psf; then
        log "unwise_psf already importable; skipping the WISE leg"
    else
        log "Setting up the unWISE W1/W2 leg (unwise_psf, fitsio, setuptools<81)"
        if [[ ! -d "$REPO_ROOT/.unwise_psf/py" ]]; then
            git clone --depth 1 https://github.com/legacysurvey/unwise_psf \
                "$REPO_ROOT/.unwise_psf"
        fi
        exclude_pycache "$REPO_ROOT/.unwise_psf"
        write_pth unwise_psf.pth "$REPO_ROOT/.unwise_psf/py"
        "$PY" -m pip install fitsio "setuptools<81"
        has_module unwise_psf \
            || { echo "unwise_psf still not importable after setup" >&2; exit 1; }
    fi
else
    log "Skipping the unWISE leg (pass --wise to include it)"
fi

# --- Done --------------------------------------------------------------------
log "Done."
if [[ "${CREATED_VENV:-0}" -eq 1 ]]; then
    echo "Created $REPO_ROOT/.venv — activate it before launching Jupyter:"
    echo "    source .venv/bin/activate"
fi
echo "Launch the notebooks with:"
echo "    jupyter lab notebooks/"
