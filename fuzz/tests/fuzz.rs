//! Short deterministic fuzzing runs of every target (longer runs: `cargo run -p meshoptimizer-fuzz --release -- <target> N`).

use meshoptimizer_fuzz::{crand::CRand, driver, target};

fn check(name: &str, iterations: u64, max_len: usize) {
    let f = target(name);
    assert!(f.is_some(), "unknown target {name}");

    for seed in 0..4 {
        let crash = f.and_then(|f| driver::run(name, f, iterations, seed, max_len));
        assert!(crash.is_none(), "{name}: crash with seed {seed} on input {crash:?}");
    }
}

#[test]
fn codec() {
    check("codec", 500, 8192);
}

#[test]
fn simplify() {
    check("simplify", 100, 1024);
}

#[test]
fn cluster() {
    check("cluster", 200, 1024);
}

#[test]
fn crand_matches_glibc() {
    // glibc: srand(1); rand() x5 and srand(42); rand() x3
    let mut r = CRand::new(1);
    let a: Vec<i32> = (0..5).map(|_| r.rand()).collect();
    assert_eq!(a, [1804289383, 846930886, 1681692777, 1714636915, 1957747793]);

    let mut r = CRand::new(42);
    let b: Vec<i32> = (0..3).map(|_| r.rand()).collect();
    assert_eq!(b, [71876166, 708592740, 1483128881]);

    // srand(0) behaves like srand(1)
    assert_eq!(CRand::new(0).rand(), 1804289383);
}
