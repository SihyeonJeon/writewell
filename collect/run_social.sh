#!/bin/sh
# Fetch every thread source one at a time. fetch exits 3 when its memory guard trips; rerun until it is done.
cd "$(dirname "$0")/.." || exit 1
for s in ${*:-dcard plurk bahamut mobile01 backpackers babyhome komica ck101}; do
  while :; do
    .venv/bin/python collect/cc_social.py fetch "$s"; rc=$?
    [ $rc -eq 3 ] || break
  done
done
