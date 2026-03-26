#!/usr/bin/env bash
set -euo pipefail

GAME_ROOT="/mnt/c/Program Files (x86)/Steam/steamapps/common/Cyberpunk 2077"
MOD_DIR="$GAME_ROOT/bin/x64/plugins/cyber_engine_tweaks/mods/py_punk_probe"

mkdir -p "$MOD_DIR"
cp "game_mods/py_punk_probe/init.lua" "$MOD_DIR/init.lua"

printf 'deployed to %s\n' "$MOD_DIR"