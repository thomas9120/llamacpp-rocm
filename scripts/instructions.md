# llama.cpp ROCm build notes

Run this from the repo root:

```powershell
.\scripts\Build-LlamaCppRocm.ps1
```

This updates the local `llama.cpp` checkout, auto-detects the latest ROCm nightly, builds for `gfx1151`, and stages the portable output here:

```text
scripts\build-output\gfx1151
```

For a faster rebuild when ROCm and `llama.cpp` are already present:

```powershell
.\scripts\Build-LlamaCppRocm.ps1 -SkipRocmDownload -SkipClone
```

After building, run from the output folder:

```powershell
cd .\scripts\build-output\gfx1151
.\llama-server.exe -m <your-model.gguf> -ngl 99
```

The script prefers Visual Studio 2022 Build Tools for ROCm HIP compilation. Visual Studio 2026 / MSVC 14.5x is known to fail in ROCm HIP math headers.
