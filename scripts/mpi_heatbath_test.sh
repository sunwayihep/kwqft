#!/usr/bin/env bash
# MPI regression tests for heatbath main program (MPI+OpenMP build: use 1 thread)
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"

BUILD_DIR="${BUILD_DIR:-build_omp_nc3_nd4}"
HEARTHBATH="${HEARTHBATH:-$(dirname "$0")/../${BUILD_DIR}/heatbath}"
BETA=6.0
NTRAJ=3
LATT="8 8 8 8"
FAIL=0
PASS=0

run_case() {
  local name="$1"
  shift
  local np="$1"
  shift
  local outfile
  outfile=$(mktemp /tmp/hb_mpi_XXXXXX.log)

  echo "============================================================"
  echo "CASE: $name  (np=$np, OMP_NUM_THREADS=$OMP_NUM_THREADS)"
  echo "CMD: mpirun -np $np $HEARTHBATH $*"
  echo "============================================================"

  if ! mpirun -np "$np" "$HEARTHBATH" "$@" >"$outfile" 2>&1; then
    echo "  FAILED: non-zero exit"
    tail -30 "$outfile"
    FAIL=$((FAIL + 1))
    rm -f "$outfile"
    return
  fi

  # Rank 0 prints plaquette lines
  local cold_plaq hot_plaq cold_p hot_p
  cold_plaq=$(grep -m1 '^Plaquette: [0-9]' "$outfile" | awk '{print $2}')
  hot_plaq=$(grep '^Plaquette: [0-9]' "$outfile" | tail -1 | awk '{print $2}')
  cold_p=$(grep -m1 '^Polyakov Loop: [0-9-]' "$outfile" | sed -n 's/.*|P| = \([0-9.eE+-]*\).*/\1/p' || true)
  hot_p=$(grep '^Polyakov Loop: [0-9-]' "$outfile" | tail -1 | sed -n 's/.*|P| = \([0-9.eE+-]*\).*/\1/p' || true)

  local ok=1
  python3 - "$cold_plaq" "$hot_plaq" "$cold_p" "$hot_p" <<'PY' || ok=0
import sys, math
cold_plaq, hot_plaq, cold_p, hot_p = sys.argv[1:5]

def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

cp, hp, cP, hP = map(f, [cold_plaq, hot_plaq, cold_p, hot_p])
errs = []
if not math.isfinite(cp) or abs(cp - 1.0) > 1e-5:
    errs.append(f"cold plaquette={cold_plaq} (expect 1.0)")
if not math.isfinite(hp) or hp < 0.35 or hp > 0.75:
    errs.append(f"hot plaquette={hot_plaq} (expect ~0.55-0.70 after {3} traj, beta=6)")
if math.isfinite(cP) and abs(cP - 1.0) > 1e-5:
    errs.append(f"cold |P|={cold_p} (expect 1.0)")
if math.isfinite(hP) and (hP <= 0 or hP > 1.01):
    errs.append(f"hot |P|={hot_p} out of range")
if errs:
    for e in errs:
        print("  CHECK:", e)
    sys.exit(1)
print(f"  cold plaq={cp:.6f}  hot plaq={hp:.6f}  cold |P|={cP:.6f}  hot |P|={hP:.6f}")
PY

  if [[ $ok -eq 1 ]]; then
    echo "  PASSED"
    PASS=$((PASS + 1))
  else
    echo "  FAILED: physics checks"
    grep -E 'Plaquette:|Polyakov|\|P\||Error|FAILED' "$outfile" || true
    FAIL=$((FAIL + 1))
  fi
  rm -f "$outfile"
  echo
}

cd "$(dirname "$0")/.."

if [[ ! -x "$HEARTHBATH" ]]; then
  echo "heatbath not found at $HEARTHBATH"
  exit 1
fi

echo "Binary: $HEARTHBATH"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  OMP_PROC_BIND=$OMP_PROC_BIND"
echo "Global lattice: $LATT  beta=$BETA  ntraj=$NTRAJ"
echo

# Serial baseline
run_case "serial np=1" 1 -latt $LATT -beta $BETA -ntraj $NTRAJ

# 2-rank splits
run_case "np=2 x-split" 2 -geom 2 1 1 1 -latt $LATT -beta $BETA -ntraj $NTRAJ
run_case "np=2 t-split" 2 -geom 1 1 1 2 -latt $LATT -beta $BETA -ntraj $NTRAJ

# 4-rank
run_case "np=4 2x2x1x1" 4 -geom 2 2 1 1 -latt $LATT -beta $BETA -ntraj $NTRAJ
run_case "np=4 2x1x1x2" 4 -geom 2 1 1 2 -latt $LATT -beta $BETA -ntraj $NTRAJ

# 8-rank (full 2^3)
if mpirun -np 8 true 2>/dev/null; then
  run_case "np=8 2x2x2x1" 8 -geom 2 2 2 1 -latt $LATT -beta $BETA -ntraj $NTRAJ
fi

# Smaller lattice quick check
run_case "np=2 small 4^4" 2 -geom 2 1 1 1 -latt 4 4 4 4 -beta $BETA -ntraj 2

# Anisotropic xi0
run_case "np=2 aniso xi0=2" 2 -geom 2 1 1 1 -latt $LATT -beta $BETA -ntraj 2 -xi0 2.0

echo "============================================================"
echo "SUMMARY: $PASS passed, $FAIL failed"
echo "============================================================"
exit $FAIL
