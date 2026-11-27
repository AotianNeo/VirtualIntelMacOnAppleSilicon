#!/bin/sh -e
#
# Helper script for the build process to apply entitlements

in_place=:
if [ "$1" = --install ]; then
  shift
  in_place=false
fi

DST="$1"
SRC="$2"
ICON="$3"
ENTITLEMENT="$4"

if $in_place; then
  trap 'rm "$DST.tmp"' exit
  cp -pPf "$SRC" "$DST.tmp"
  SRC="$DST.tmp"
else
  cd "$MESON_INSTALL_DESTDIR_PREFIX"
fi

# Rez's resource fork is detritus to codesign, and this binary needs the
# hypervisor entitlement (HVF or hvdbt). Sign after dropping the fork.
if test -n "$ICON" && test -f "$ICON"; then
  Rez -append "$ICON" -o "$SRC" || true
  SetFile -a C "$SRC" || true
fi
xattr -cr "$SRC" || true
if test -z "$ENTITLEMENT"; then
  ENTITLEMENT=$(CDPATH= cd -- "$(dirname "$0")/../accel/hvf" && pwd)/entitlements.plist
fi
if test -n "$ENTITLEMENT" && test -f "$ENTITLEMENT"; then
  codesign --entitlements "$ENTITLEMENT" --force -s - "$SRC"
fi

mv -f "$SRC" "$DST"
trap '' exit
