#!/bin/sh

set -eu

repository_url="https://github.com/xiph/rnnoise.git"
source_commit="70f1d256acd4b34a572f999a05c87bf00b67730d"
model_sha256="0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"
model_c_sha256="d6021b7697677c4d2274c912975e143765552b0e6f25500aa660fdb4a9849be5"
model_h_sha256="09ff880bddd0fc74a2ae0e5ec6c8d65714031b08d0c3f672493acd9e189c5855"

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
output_root="$project_root/build/rnnoise"
source_cache=""

usage() {
    printf '%s\n' \
        "Usage: $0 [--source-cache PATH] [--output-dir PATH]" \
        "" \
        "Without --source-cache, the pinned source and model are downloaded." \
        "A cache must be an RNNoise git checkout containing the generated" \
        "src/rnnoise_data.c and src/rnnoise_data.h model files."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source-cache)
            [ "$#" -ge 2 ] || {
                usage
                exit 2
            }
            source_cache=$2
            shift 2
            ;;
        --output-dir)
            [ "$#" -ge 2 ] || {
                usage
                exit 2
            }
            output_root=$2
            shift 2
            ;;
        --help|-h)
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

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        printf 'Neither shasum nor sha256sum is available.\n' >&2
        exit 1
    fi
}

verify_sha256() {
    expected=$1
    path=$2
    actual=$(sha256_file "$path")
    if [ "$actual" != "$expected" ]; then
        printf 'SHA-256 mismatch for %s\nexpected: %s\nactual:   %s\n' \
            "$path" "$expected" "$actual" >&2
        exit 1
    fi
}

stage_dir=$(mktemp -d "${TMPDIR:-/tmp}/rnnoise-build.XXXXXX")
trap 'rm -rf "$stage_dir"' EXIT HUP INT TERM
source_dir="$stage_dir/source"
mkdir -p "$source_dir"

if [ -n "$source_cache" ]; then
    [ -d "$source_cache/.git" ] || {
        printf 'Not an RNNoise git checkout: %s\n' "$source_cache" >&2
        exit 1
    }
    git -C "$source_cache" cat-file -e "$source_commit^{commit}"
    git -C "$source_cache" archive "$source_commit" | tar -x -C "$source_dir"
    for model_file in \
        rnnoise_data.c \
        rnnoise_data.h \
        rnnoise_data_little.c \
        rnnoise_data_little.h
    do
        cache_model="$source_cache/src/$model_file"
        [ -f "$cache_model" ] || {
            printf 'Missing cached model file: %s\n' "$cache_model" >&2
            exit 1
        }
        cp "$cache_model" "$source_dir/src/$model_file"
    done
else
    git clone --filter=blob:none --no-checkout "$repository_url" "$source_dir"
    git -C "$source_dir" checkout --detach "$source_commit"
    model_archive="$stage_dir/rnnoise_data-$model_sha256.tar.gz"
    curl -L --fail --retry 3 \
        "https://media.xiph.org/rnnoise/models/rnnoise_data-$model_sha256.tar.gz" \
        -o "$model_archive"
    verify_sha256 "$model_sha256" "$model_archive"
    tar -xzf "$model_archive" -C "$source_dir"
fi

resolved_commit=$(git -C "$source_cache" rev-parse "$source_commit" 2>/dev/null || true)
if [ -z "$source_cache" ]; then
    resolved_commit=$(git -C "$source_dir" rev-parse HEAD)
fi
[ "$resolved_commit" = "$source_commit" ] || {
    printf 'Source commit mismatch: %s\n' "$resolved_commit" >&2
    exit 1
}

verify_sha256 "$model_c_sha256" "$source_dir/src/rnnoise_data.c"
verify_sha256 "$model_h_sha256" "$source_dir/src/rnnoise_data.h"

platform=$(uname -s)
architecture=$(uname -m)
build_bin="$stage_dir/artifacts/bin"
build_lib="$stage_dir/artifacts/lib"
build_include="$stage_dir/artifacts/include"
mkdir -p "$build_bin" "$build_lib" "$build_include"

common_flags="-O3 -fPIC -DRNNOISE_BUILD -DDISABLE_DEBUG_FLOAT"
include_flags="-I$source_dir/include -I$source_dir/src"
sources="
$source_dir/src/denoise.c
$source_dir/src/rnn.c
$source_dir/src/pitch.c
$source_dir/src/kiss_fft.c
$source_dir/src/celt_lpc.c
$source_dir/src/nnet.c
$source_dir/src/nnet_default.c
$source_dir/src/parse_lpcnet_weights.c
$source_dir/src/rnnoise_data.c
$source_dir/src/rnnoise_tables.c
"

case "$platform" in
    Darwin)
        library_name="librnnoise.dylib"
        # shellcheck disable=SC2086
        cc $common_flags $include_flags -dynamiclib \
            -Wl,-install_name,@rpath/librnnoise.dylib \
            -o "$build_lib/$library_name" $sources -lm
        cc -O3 -I"$source_dir/include" \
            -o "$build_bin/rnnoise_demo" \
            "$source_dir/examples/rnnoise_demo.c" \
            -L"$build_lib" -lrnnoise \
            -Wl,-rpath,@loader_path/../lib
        ;;
    Linux)
        library_name="librnnoise.so"
        # shellcheck disable=SC2086
        cc $common_flags $include_flags -shared \
            -Wl,-soname,librnnoise.so \
            -o "$build_lib/$library_name" $sources -lm
        cc -O3 -I"$source_dir/include" \
            -o "$build_bin/rnnoise_demo" \
            "$source_dir/examples/rnnoise_demo.c" \
            -L"$build_lib" -lrnnoise \
            -Wl,-rpath,'$ORIGIN/../lib'
        ;;
    *)
        printf 'Unsupported platform for direct build: %s\n' "$platform" >&2
        exit 1
        ;;
esac

cp "$source_dir/include/rnnoise.h" "$build_include/rnnoise.h"
mkdir -p "$output_root/bin" "$output_root/lib" "$output_root/include"
cp "$build_bin/rnnoise_demo" "$output_root/bin/rnnoise_demo"
cp "$build_lib/$library_name" "$output_root/lib/$library_name"
cp "$build_include/rnnoise.h" "$output_root/include/rnnoise.h"

printf '%s\n' \
    "RNNoise build complete" \
    "source:       $repository_url" \
    "commit:       $source_commit" \
    "model sha256: $model_sha256" \
    "platform:     $platform" \
    "architecture: $architecture" \
    "CLI:          $output_root/bin/rnnoise_demo" \
    "library:      $output_root/lib/$library_name"
