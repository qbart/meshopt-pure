# Meshoptimizer for Rust language

Meshoptimizer is a Rust library as pure rust port of library in C++ https://github.com/zeux/meshoptimizer.

The only C bindings that are allowed:
- Vulkan

Others must be human approved.

So it means that external dependencies should be ported as well or rust crates be used instead assuming they:
- are covering all required features
- are pure rust otherwise it breaks the main goal of this library port

## Dependecies to use

- KTX: https://github.com/qbart/ktx-rs

## Rules

1. Keep the copy of original mesh optimizer in `./ref/meshopt` and keep the `./ref` in gitignore
2. Follow the original API as close as possible, that includes freeing memory manually.
3. Do not use .unwrap / .expect. Library should create it's own error handling apprach (`err, ok := Operation` like in golang is preferred to simplify the API unless ktx lib assumes otherwise, however runtime exceptions are not allowed).
4. Make sure to port original tests 1:1.
5. Make sure to copy the test data like images to the library and version them in repository.
6. Maintain implementation audit logs at `./docs/spec/*.md` First write the spec there but be concise.
7. Maintain ROADMAP.md that keeps the progress about whole project.
8. Only use main branch.
9. `unsafe` is denied except in rare, special cases.

## Loop iteration

1. Analyze what is missing from required features
2. Implement API partially, provide atomic changes.
3. Commit after each change, use semenatic git commits, keep the messages short, skip the annoying Co-Author commits.
4. Verify tests
5. Fix what is needed until tests are passing.
7. Push changes
8. Notify about changes done in max 5 sentence summary.
