# Pre-BOLT artifact prototype

This prototype addresses pytorch/pytorch#202470: the final Python build can
overwrite an AOT-enabled pre-BOLT CUDA library with a kernel-free variant.

The upload is assembled in `build/lib/prebolt-artifact/`, separately from the
working `build/lib/prebolt/` directory:

```text
libc10.so
libc10_cuda.so
libtorch_cpu.so
<SOABI>/libtorch_python.so
cuda/<sha256>/libtorch_cuda.so
cuda-manifest.json
```

After native AOT stage 2, each CUDA iteration hashes its pre-BOLT CUDA library.
Identical bytes share a single archived file. Copies are independent of the
working library, so later relinks cannot overwrite earlier snapshots. The
Python-specific library is retained per SOABI as before. Shared libraries are
copied at the end of the build loop. The working CUDA copy is excluded from
the upload to avoid storing it twice. Artifact names and seven-day retention
are unchanged.

`cuda-manifest.json` maps each Python SOABI to the observed SHA-256, byte size,
wheel AOT status, retention status, and relative archive path. Read the entry
for your wheel's SOABI and copy the referenced CUDA library alongside the
corresponding Python library when assembling a profiling installation. A null
path means that exact variant was not retained; do not substitute another one
based only on its AOT status.

The default limit is **two distinct CUDA binaries per artifact**, set by
`PREBOLT_CUDA_MAX_VARIANTS`. When the limit is reached, identical binaries can
still reuse an existing copy. New variants are recorded with
`status: variant_limit_exceeded` and `path: null`, and emit a CI warning.
This preserves the first two distinct variants, not necessarily one of each
AOT state. Deduplication alone would not bound storage if all interpreters
produce different bytes. The cap deliberately trades coverage for bounded
storage, and needs validation against a real nightly before adoption.

With 615 MiB libraries, two variants across three CUDA versions and seven
daily builds occupy about 25.2 GiB uncompressed for CUDA libraries alone,
versus 12.6 GiB for the existing single-copy layout. Compression changes billed
artifact storage; measure the uploaded archive size as well as extracted sizes.

`wheel_native_aot` is probed from this iteration's installed wheel after stage
2. It does not certify that the archived pre-BOLT library executes AOT kernels.
CI validation should replace the matching wheel's libraries with its archived
copies, check `torch._native._native_aot_embedded()`, and run a representative
native AOT operation. Also inspect distinct hashes and skipped entries for all
Python versions and all three CUDA configurations. No full CUDA build or such
runtime validation has been performed for this prototype.

AI assistance: this prototype was prepared with Codex for human review.
