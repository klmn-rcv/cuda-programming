#!/usr/bin/env bash
# Run a local project through an existing or newly created Colab runtime.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  colab_run.sh --session NAME --source DIR --build COMMAND --run COMMAND [options]

Required:
  --session NAME       Existing Colab session name.
  --source DIR         Local project directory to package and upload.
  --build COMMAND      Build command executed inside the uploaded source directory.
  --run COMMAND        Run command executed inside the uploaded source directory.

Options:
  --log FILE           Persist the combined build and run log at FILE.
                       Output is always printed to the console.
  --timeout SECONDS    Colab execution timeout. Default: 300.
  --gpu GPU           GPU requested only when creating a new session. Default: $COLAB_GPU or T4.
  --replace-session   Stop and recreate an existing same-named session before running. Remote session files are lost.
  -h, --help           Show this help.

Examples:
  colab_run.sh --session gemm-t4 \
    --source /home/jihonglu/CUDA/CUDA-From-Correctness-To-Performance-Code \
    --build 'make clean all' \
    --run './gemm_test 512 512 512 gpu_tiling'
EOF
}

session=""
source_dir=""
build_cmd=""
run_cmd=""
log_file=""
timeout=300
gpu="${COLAB_GPU:-T4}"
replace_session=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      session="${2:?missing value for --session}"
      shift 2
      ;;
    --source)
      source_dir="${2:?missing value for --source}"
      shift 2
      ;;
    --build)
      build_cmd="${2:?missing value for --build}"
      shift 2
      ;;
    --run)
      run_cmd="${2:?missing value for --run}"
      shift 2
      ;;
    --log)
      log_file="${2:?missing value for --log}"
      shift 2
      ;;
    --timeout)
      timeout="${2:?missing value for --timeout}"
      shift 2
      ;;
    --gpu)
      gpu="${2:?missing value for --gpu}"
      shift 2
      ;;
    --replace-session)
      replace_session=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$session" || -z "$source_dir" || -z "$build_cmd" || -z "$run_cmd" ]]; then
  printf '%s\n' 'Missing required arguments.' >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$source_dir" ]]; then
  printf 'Source directory does not exist: %s\n' "$source_dir" >&2
  exit 2
fi

if [[ -z "${ALL_PROXY:-}" && -z "${all_proxy:-}" ]]; then
  printf '%s\n' 'No SOCKS proxy is configured in this shell. Run this from a login shell with ALL_PROXY configured.' >&2
  exit 2
fi

if [[ -n "${COLAB_BIN:-}" ]]; then
  colab_bin="$COLAB_BIN"
elif command -v colab >/dev/null 2>&1; then
  colab_bin="$(command -v colab)"
elif [[ -x "${HOME}/.venvs/colab-cli/bin/colab" ]]; then
  colab_bin="${HOME}/.venvs/colab-cli/bin/colab"
else
  printf '%s\n' 'Colab CLI was not found. Set COLAB_BIN or install it in ~/.venvs/colab-cli/bin/colab.' >&2
  exit 127
fi

auth_status="$("$colab_bin" --auth=adc status 2>&1 || true)"
if [[ "$auth_status" == *"No valid default credentials"* ]]; then
  printf "%s\n" "Colab ADC authorization is required. Run the following command in this Ubuntu terminal:" >&2
  printf "%s\n" "gcloud auth application-default login --no-launch-browser --scopes="openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"" >&2
  exit 2
fi

source_dir="$(realpath "$source_dir")"
project_name="$(basename "$source_dir")"
run_id="${project_name}-$(date +%Y%m%d-%H%M%S)"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

archive="$work_dir/${run_id}.tar.gz"
remote_archive="/content/${run_id}.tar.gz"
remote_project="/content/${run_id}"
remote_log="/content/${run_id}.log"
if [[ -n "$log_file" ]]; then
  local_log="$(realpath -m "$log_file")"
  mkdir -p "$(dirname "$local_log")"
else
  local_log="$work_dir/${run_id}.log"
fi

build_b64="$(printf '%s' "$build_cmd" | base64 -w 0)"
run_b64="$(printf '%s' "$run_cmd" | base64 -w 0)"
project_b64="$(printf '%s' "$remote_project" | base64 -w 0)"
archive_b64="$(printf '%s' "$remote_archive" | base64 -w 0)"
log_b64="$(printf '%s' "$remote_log" | base64 -w 0)"

printf 'Checking Colab session %q...\n' "$session"
session_status="$("$colab_bin" --auth=adc status -s "$session" 2>&1 || true)"
if [[ -z "$session_status" || "$session_status" == *"not found"* || "$session_status" == *"No active sessions"* ]]; then
  printf 'Session %q is unavailable; creating it with GPU %s...\n' "$session" "$gpu"
  "$colab_bin" --auth=adc new -s "$session" --gpu "$gpu"
elif [[ "$replace_session" == 1 ]]; then
  printf 'Replacing Colab session %q with GPU %s...\n' "$session" "$gpu"
  "$colab_bin" --auth=adc stop -s "$session"
  for attempt in 1 2 3 4 5; do
    if "$colab_bin" --auth=adc new -s "$session" --gpu "$gpu"; then
      break
    fi
    if [[ "$attempt" == 5 ]]; then
      printf 'Could not recreate session %q after stopping it.\n' "$session" >&2
      exit 1
    fi
    sleep 2
  done
else
  if [[ "$session_status" == *"Hardware:"* && "$session_status" != *"Hardware: $gpu"* ]]; then
    printf "Session %q uses different hardware. Requested: %s. Use --replace-session to recreate it, or choose a different session name.\n" "$session" "$gpu" >&2
    exit 2
  fi
  printf '%s\n' "$session_status"
fi

# A newly allocated runtime may not create /content until its Jupyter kernel
# receives the first execution request. Initialize it before using the file API.
printf '%s\n' 'Initializing Colab workspace...'
init_ok=0
init_log="$work_dir/colab-init.log"
for attempt in 1 2 3; do
  if "$colab_bin" --auth=adc exec -s "$session" >"$init_log" 2>&1 <<'PY'
import os

os.makedirs("/content", exist_ok=True)
PY
  then
    init_ok=1
    break
  fi
  printf 'Colab kernel is not ready yet; retrying (%s/3)...\n' "$attempt" >&2
  sleep 2
done
if [[ "$init_ok" != 1 ]]; then
  cat "$init_log" >&2
  exit 1
fi

printf 'Packaging %s...\n' "$source_dir"
tar -czf "$archive" -C "$source_dir" .

printf 'Uploading sources to %s...\n' "$remote_archive"
"$colab_bin" --auth=adc upload -s "$session" "$archive" "$remote_archive"

printf 'Building and running on Colab...\n'
execution_output="$($colab_bin --auth=adc exec -s "$session" --timeout "$timeout" <<PY
import base64
import pathlib
import subprocess

project = base64.b64decode("$project_b64").decode()
archive = base64.b64decode("$archive_b64").decode()
log_path = base64.b64decode("$log_b64").decode()
build = base64.b64decode("$build_b64").decode()
run = base64.b64decode("$run_b64").decode()

pathlib.Path(project).mkdir(parents=True, exist_ok=True)
steps = [
    f"tar -xzf {archive} -C {project}",
    build,
    run,
]

with open(log_path, "w", encoding="utf-8") as log:
    exit_code = 0
    for command in steps:
        log.write(f"$ {command}\\n")
        result = subprocess.run(
            command,
            shell=True,
            cwd=project,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(result.stdout)
        if result.returncode:
            exit_code = result.returncode
            break

print(f"COLAB_RUN_EXIT_CODE={exit_code}")
PY
)"
printf '%s\n' "$execution_output"

printf 'Downloading log to %s...\n' "$local_log"
"$colab_bin" --auth=adc download -s "$session" "$remote_log" "$local_log"
printf '\n--- Colab output ---\n'
cat "$local_log"
printf '\n--- End Colab output ---\n'

# The remote log is only a transport artifact. Retain a local copy only when
# the caller explicitly requested one with --log.
"$colab_bin" --auth=adc rm -s "$session" "$remote_log" >/dev/null || true

exit_code="$(printf '%s\n' "$execution_output" | sed -n 's/^COLAB_RUN_EXIT_CODE=//p' | tail -n 1)"
if [[ -z "$exit_code" ]]; then
  printf 'Could not determine the remote exit code. See %s\n' "$local_log" >&2
  exit 1
fi

if [[ "$exit_code" != 0 ]]; then
  if [[ -n "$log_file" ]]; then
    printf 'Remote command failed with exit code %s. Log: %s\n' "$exit_code" "$local_log" >&2
  else
    printf 'Remote command failed with exit code %s. Output is shown above.\n' "$exit_code" >&2
  fi
  exit "$exit_code"
fi

if [[ -n "$log_file" ]]; then
  printf 'Completed successfully. Log: %s\n' "$local_log"
else
  printf '%s\n' 'Completed successfully.'
fi
