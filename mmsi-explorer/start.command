#!/bin/zsh
cd "$(dirname "$0")/dist" || exit 1
open "http://localhost:4173"
python3 -m http.server 4173
