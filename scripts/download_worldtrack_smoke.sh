#!/usr/bin/env bash
set -euo pipefail

GDOWN_BIN="${GDOWN_BIN:-gdown}"
OUT_DIR="${OUT_DIR:-data/worldtrack_release}"

mkdir -p \
  "${OUT_DIR}/adt_mini" \
  "${OUT_DIR}/ds_mini" \
  "${OUT_DIR}/po_mini" \
  "${OUT_DIR}/pstudio_mini"

"${GDOWN_BIN}" --continue \
  "https://drive.google.com/uc?id=1dqQGUTajJ8tvfrC_FDV3xKulLX0GN748" \
  -O "${OUT_DIR}/adt_mini/Apartment_release_clean_seq131_0.npz"

"${GDOWN_BIN}" --continue \
  "https://drive.google.com/uc?id=1uzgdptcIo8gXD8rJCoV_LZNbLcKCTvD3" \
  -O "${OUT_DIR}/ds_mini/01f258-3_obj_source_left_6.npz"

"${GDOWN_BIN}" --continue \
  "https://drive.google.com/uc?id=1XVsRlbv9XSYkAd9Oz-ET7dX2Ej9By4jJ" \
  -O "${OUT_DIR}/po_mini/cab_e_3rd_1.npz"

"${GDOWN_BIN}" --continue \
  "https://drive.google.com/uc?id=1uTO82jc0gtOqHideylR-bCk3GvAyf00p" \
  -O "${OUT_DIR}/pstudio_mini/basketball_13.npz"

find "${OUT_DIR}" -type f -name "*.npz" -printf "%p %k KB\n" | sort
