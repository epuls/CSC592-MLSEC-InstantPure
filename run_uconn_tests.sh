#!/usr/bin/env bash

LORA_DIR="logs/OSCP/unet_lora/"
N=1000
DEVICE="cuda"

run_test () {
  local name="$1"
  local attack="$2"
  local eps="$3"
  local iters="$4"
  local alpha="$5"
  local strength="$6"

  echo "Running ${name}..."

  python test.py \
    --model="LCM" \
    --output_dir="output/${name}/" \
    --num_validation_set="${N}" \
    --lora_input_dir="${LORA_DIR}" \
    --strength="${strength}" \
    --num_inference_step=5 \
    --device="${DEVICE}" \
    --attack_method="${attack}" \
    --eps="${eps}" \
    --iter="${iters}" \
    --alpha="${alpha}"
}

run_test "aa_eps4_s02"       "AutoAttack" 4  10 1 0.2
run_test "aa_eps8_s02"       "AutoAttack" 8  10 1 0.2
run_test "pgd_eps8_i10_s02"  "Linf_pgd"   8  10 1 0.2
run_test "pgd_eps8_i40_s02"  "Linf_pgd"   8  40 1 0.2
run_test "aa_eps8_s01"       "AutoAttack" 8  10 1 0.1
run_test "aa_eps8_s03"       "AutoAttack" 8  10 1 0.3

echo "All tests complete."