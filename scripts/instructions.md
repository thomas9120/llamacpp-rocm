# llama.cpp ROCm build notes

Run this from the repo root:

```powershell
.\scripts\Build-LlamaCppRocm.ps1
```

This updates the local `llama.cpp` checkout, auto-detects the latest ROCm nightly, builds for `gfx1151`, and stages the portable output here:

```text
scripts\build-output\gfx1151
```

gfx1151 builds apply the vendored `strix-halo-fa-v1` ROCm patch set by default. The patch teaches the tile flash-attention kernel to dequantize symmetric q4_0/q8_0 K/V on load and reuse it across GQA heads. Vulkan changes from the source branch are not included.

Staged artifacts include `llamacpp-rocm-patchset.txt`, containing the applied patch-set ID or `none` for an unpatched build.

Patch selection is explicit when needed:

```powershell
# Default behavior: patch gfx1151, leave other targets unchanged.
.\scripts\Build-LlamaCppRocm.ps1 -StrixHaloFaFix auto

# Build an unpatched baseline.
.\scripts\Build-LlamaCppRocm.ps1 -StrixHaloFaFix off

# Apply the patch to another target for controlled testing.
.\scripts\Build-LlamaCppRocm.ps1 -GpuTarget gfx1150 -StrixHaloFaFix on
```

When `-SkipClone` reuses a checkout from a previous patched build, `off` removes the recorded patch set before configuring the baseline.

For a faster rebuild when ROCm and `llama.cpp` are already present:

```powershell
.\scripts\Build-LlamaCppRocm.ps1 -SkipRocmDownload -SkipClone
```

HIP template compilation can consume substantial memory. Automatic builds cap parallel compilation at eight jobs; use `-BuildJobs` to choose a lower or higher limit. Ccache is disabled by default because the frontend bundled with Strawberry Perl can stall clean HIP builds; opt in with `-EnableCcache` for incremental builds after confirming it works on your setup. Build the expanded backend test suite with:

```powershell
.\scripts\Build-LlamaCppRocm.ps1 -BuildTests -BuildJobs 4
.\scripts\llama.cpp\build\bin\test-backend-ops.exe -o FLASH_ATTN_EXT
```

To test the opt-in HIP VMM allocator build suggested in upstream PR #94:

```powershell
.\scripts\Build-LlamaCppRocm-HipVmm.ps1
```

That variant passes `-DGGML_HIP_NO_VMM=OFF` and writes to separate `build-hip-vmm` / `build-output\<target>-hip-vmm` folders.

After building, run from the output folder:

```powershell
cd .\scripts\build-output\gfx1151
.\llama-server.exe -m <your-model.gguf> -ngl 99
```

The script prefers Visual Studio 2022 Build Tools for ROCm HIP compilation. Visual Studio 2026 / MSVC 14.5x is known to fail in ROCm HIP math headers.

## Long-context comparison

Use the same llama.cpp commit, ROCm version, model, and warmed page cache for both `-StrixHaloFaFix off` and `on` builds:

```powershell
.\llama-bench.exe -m <model.gguf> -fa 1 -ctk q8_0 -ctv q8_0 `
  -b 512 -ub 512 -p 512 -n 32 -d 16384,32768,65536 -r 2
```

The expected gain is in token generation at deep KV depth. Prefill at head sizes 64/128 and f16 KV should remain effectively unchanged. See [`STRIX-HALO.md`](../STRIX-HALO.md) for the source measurements and caveats.
