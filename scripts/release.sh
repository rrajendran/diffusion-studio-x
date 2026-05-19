#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/release.sh 1.2.3        # explicit version
#   ./scripts/release.sh patch        # bump patch:  0.1.0 → 0.1.1
#   ./scripts/release.sh minor        # bump minor:  0.1.0 → 0.2.0
#   ./scripts/release.sh major        # bump major:  0.1.0 → 1.0.0

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo "error: $*" >&2; exit 1; }

current_version() {
  node -p "require('./package.json').version"
}

bump() {
  local part="$1" ver="$2"
  IFS='.' read -r major minor patch <<< "$ver"
  case "$part" in
    major) echo "$((major + 1)).0.0" ;;
    minor) echo "${major}.$((minor + 1)).0" ;;
    patch) echo "${major}.${minor}.$((patch + 1))" ;;
    *)     die "unknown bump type: $part" ;;
  esac
}

# ── resolve target version ────────────────────────────────────────────────────

ARG="${1:-}"
[ -z "$ARG" ] && die "usage: $0 <major|minor|patch|x.y.z>"

CURRENT="$(current_version)"

case "$ARG" in
  major|minor|patch) VERSION="$(bump "$ARG" "$CURRENT")" ;;
  [0-9]*) VERSION="$ARG" ;;
  *) die "expected major|minor|patch or a version like 1.2.3, got: $ARG" ;;
esac

[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "invalid version: $VERSION"

TAG="v${VERSION}"

echo "Current version : $CURRENT"
echo "New version     : $VERSION  ($TAG)"
echo ""

# ── pre-flight checks ─────────────────────────────────────────────────────────

# Must be on main
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || die "releases must be cut from main (currently on $BRANCH)"

# Working tree must be clean
[ -z "$(git status --porcelain)" ] || die "working tree is dirty — commit or stash changes first"

# Tag must not already exist
git rev-parse "$TAG" &>/dev/null && die "tag $TAG already exists"

# Remote must be reachable
git ls-remote --exit-code origin &>/dev/null || die "cannot reach origin"

echo "Pre-flight checks passed."
echo ""

# ── bump version in all three manifest files ──────────────────────────────────

update_json() {
  local file="$1" old_ver="$2" new_ver="$3"
  # In-place sed that works on both macOS (BSD sed) and Linux (GNU sed)
  sed -i.bak "s/\"version\": \"${old_ver}\"/\"version\": \"${new_ver}\"/" "$file"
  rm -f "${file}.bak"
}

update_toml() {
  local file="$1" old_ver="$2" new_ver="$3"
  sed -i.bak "s/^version = \"${old_ver}\"/version = \"${new_ver}\"/" "$file"
  rm -f "${file}.bak"
}

update_json  "package.json"                 "$CURRENT" "$VERSION"
update_json  "src-tauri/tauri.conf.json"    "$CURRENT" "$VERSION"
update_toml  "src-tauri/Cargo.toml"         "$CURRENT" "$VERSION"

echo "Updated versions in package.json, tauri.conf.json, Cargo.toml"

# ── commit, tag, push ─────────────────────────────────────────────────────────

git add package.json src-tauri/tauri.conf.json src-tauri/Cargo.toml
git commit -m "chore: release ${TAG}"

git tag "$TAG"

echo ""
echo "Pushing commit and tag to origin..."
git push origin main
git push origin "$TAG"

echo ""
echo "Done. Release workflow triggered for ${TAG}."
echo "Track progress: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]//' | sed 's/\.git$//')/actions"
