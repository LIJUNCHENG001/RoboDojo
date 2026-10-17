#!/usr/bin/env bash
set -uo pipefail

ROBO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OLD_ROBO_ROOT="${OLD_ROBO_ROOT:-/mnt/workspace/RoboDojo}"
OLD_POLICY_DIR="${OLD_POLICY_DIR:-${OLD_ROBO_ROOT}/XPolicyLab/policy/OpenDM}"
LOCAL_CKPT="${LOCAL_CKPT:-/root/ljc/opendm/user_checkpoints/robodojo_press_by_number_base_dm05_bs64_lr2p5e-5/checkpoint-30000}"
DEPLOY_CFG="${OPENDM_DEPLOY_CONFIG:-${ROBO_ROOT}/XPolicyLab/policy/OpenDM/deploy_chunk25.yml}"
RUN_TAG="${RUN_TAG:-oldpolicy_$(date +%Y%m%d_%H%M%S)}"
BASE_LABEL="pressnum_dm05_bs64_lr2p5e5_ckpt30000_oldpolicy"
BASE_RUN="${BASE_LABEL}_50ep_4gpu"
RESULT_ROOT="${ROBO_ROOT}/eval_result/RoboDojo/press_by_number/OpenDM/arx_x5"
LOG_DIR="${ROBO_ROOT}/eval_result/_logs/${BASE_RUN}_${RUN_TAG}"
SUMMARY_PATH="${LOG_DIR}/summary.json"

mkdir -p "${LOG_DIR}"

test -s "${LOCAL_CKPT}/model.safetensors"
test -s "${LOCAL_CKPT}/norm_stats.json"
test -s "${DEPLOY_CFG}"
test -f "${OLD_POLICY_DIR}/setup_eval_policy_server.sh"
test -d "${ROBO_ROOT}/Assets/Eval_Layout/RoboDojo/arx_x5/0"

if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -q .; then
    echo "GPUs are busy; refusing to mix this evaluation with another job" >&2
    exit 3
fi

(
    cd "${OLD_POLICY_DIR}"
    env PYTHONPATH="${OLD_ROBO_ROOT}" conda run -n opendm python -c \
        'import XPolicyLab; from XPolicyLab.policy.OpenDM import model; print(f"server_package={XPolicyLab.__file__}"); print(f"server_model={model.__file__}")'
) | tee "${LOG_DIR}/server_import.txt"

if ! grep -Fq "server_model=${OLD_POLICY_DIR}/model.py" "${LOG_DIR}/server_import.txt"; then
    echo "Old policy import verification failed" >&2
    exit 4
fi

declare -a starts=(0 13 26 38)
declare -a ends=(12 25 37 49)
declare -a pids=()

for shard in 0 1 2 3; do
    gpu=${shard}
    start=${starts[$shard]}
    end=${ends[$shard]}
    count=$((end - start + 1))
    label=${BASE_LABEL}_shard${shard}_gpu${gpu}
    run_id=${BASE_RUN}_shard${shard}_gpu${gpu}_${RUN_TAG}
    parent=${RESULT_ROOT}/0_ckpt_name=${label},action_type=joint
    manifest=${parent}/_resume_${run_id}.json
    log=${LOG_DIR}/shard${shard}_gpu${gpu}.log
    mkdir -p "${parent}"

    python3 - "${manifest}" "${run_id}" "${start}" "${end}" "${ROBO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

path, run_id, start, end, root = sys.argv[1:]
start, end = int(start), int(end)
layout_dir = Path(root, "Assets", "Eval_Layout", "RoboDojo", "arx_x5", "0")
total_layouts = len(list(layout_dir.glob("press_by_number_*.json")))
excluded = [layout_id for layout_id in range(total_layouts) if not start <= layout_id <= end]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "run_id": run_id,
            "success_nums": 0,
            "fail_nums": 0,
            "total_score": 0,
            "abandoned_layout_ids": excluded,
            "details": {},
        },
        handle,
        indent=2,
    )
PY

    port=$(bash "${ROBO_ROOT}/XPolicyLab/utils/get_free_port.sh")
    (
        server_pid=""
        cleanup() {
            if [[ -n "${server_pid}" ]]; then
                kill "${server_pid}" 2>/dev/null || true
                wait "${server_pid}" 2>/dev/null || true
            fi
        }
        trap cleanup EXIT

        env \
            PYTHONPATH="${OLD_ROBO_ROOT}" \
            OPENDM_DEPLOY_CONFIG="${DEPLOY_CFG}" \
            MODEL_PATH="${LOCAL_CKPT}" \
            NORM_STATS_PATH="${LOCAL_CKPT}/norm_stats.json" \
            bash "${ROBO_ROOT}/scripts/robodojo.sh" server \
                --policy-dir "${OLD_POLICY_DIR}" \
                --task press_by_number \
                --ckpt "${label}" \
                --policy-env opendm \
                --env-cfg arx_x5 \
                --action-type joint \
                --seed 0 \
                --policy-gpu "${gpu}" \
                --policy-port "${port}" \
                --bind-host localhost &
        server_pid=$!

        bash "${ROBO_ROOT}/XPolicyLab/utils/wait_for_policy_server.sh" \
            localhost "${port}" "${server_pid}" "old OpenDM policy server" 1200

        cd "${ROBO_ROOT}"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate RoboDojo
        env \
            LD_LIBRARY_PATH="/root/miniforge3/envs/RoboDojo/lib:${LD_LIBRARY_PATH:-}" \
            VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
            VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
            EVAL_NUM="${count}" \
            ROBODOJO_RUN_ID="${run_id}" \
            bash scripts/robodojo.sh client \
                --policy-name OpenDM \
                --policy-host localhost \
                --policy-port "${port}" \
                --task press_by_number \
                --ckpt "${label}" \
                --env-cfg arx_x5 \
                --action-type joint \
                --seed 0 \
                --env-gpu "${gpu}" \
                --eval-num "${count}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
    echo "started shard=${shard} gpu=${gpu} layouts=${start}..${end} pid=${pids[-1]} port=${port} log=${log}"
done

rc=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        rc=1
    fi
done

python3 - "${RESULT_ROOT}" "${BASE_LABEL}" "${BASE_RUN}" "${RUN_TAG}" "${SUMMARY_PATH}" <<'PY'
import json
import os
import sys

root, base_label, base_run, run_tag, summary_path = sys.argv[1:]
rows = []
paths = []
for shard in range(4):
    label = f"{base_label}_shard{shard}_gpu{shard}"
    run_id = f"{base_run}_shard{shard}_gpu{shard}_{run_tag}"
    path = os.path.join(
        root,
        f"0_ckpt_name={label},action_type=joint",
        run_id,
        "_result.json",
    )
    paths.append(path)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.load(handle)["details"].values())

rows.sort(key=lambda row: int(row["layout_id"]))
successes = sum(bool(row.get("success")) for row in rows)
summary = {
    "cases": len(rows),
    "successes": successes,
    "success_rate": successes / len(rows) if rows else 0.0,
    "mean_score": sum(float(row.get("score", 0)) for row in rows) / len(rows) if rows else 0.0,
    "layouts": [int(row["layout_id"]) for row in rows],
    "failed_layouts": [int(row["layout_id"]) for row in rows if not row.get("success")],
    "result_files": paths,
}
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
print(f"summary={summary_path}")
PY

exit "${rc}"
