# llama.cpp ROCm build notes

Run this from the repo root:

```powershell
.\scripts\Build-LlamaCppRocm.ps1
```
.\scripts\Build-LlamaCppRocm.ps1 -GpuTarget gfx1151



This updates the local `llama.cpp` checkout, auto-detects the latest ROCm nightly, builds for `gfx1151`, and stages the portable output here:

```text
scripts\build-output\gfx1151
```

For a faster rebuild when ROCm and `llama.cpp` are already present:

```powershell
.\scripts\Build-LlamaCppRocm.ps1 -SkipRocmDownload -SkipClone
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
