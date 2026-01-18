default:
    @just --list

check:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    if command -v cargo-nextest >/dev/null 2>&1; then cargo nextest run; else cargo test; fi

test:
    if command -v cargo-nextest >/dev/null 2>&1; then cargo nextest run; else cargo test; fi

fmt:
    cargo fmt --all
