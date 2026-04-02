# Crate Publishing TODO

- [x] Add missing Cargo.toml metadata: `description`, `authors`, `repository`, `keywords`, `categories`, `readme`
- [ ] Add crate-level doc comment (`//!`) to `src/lib.rs`
- [ ] Add doc comments to `OilError` enum in `src/scale.rs`
- [ ] Decide whether `OilScale` struct fields should be public or private with accessors
- [ ] Add `#![warn(missing_docs)]` to `src/lib.rs` and fix any warnings
- [ ] Remove `Cargo.lock` from version control (convention for libraries)
