# Strix Halo quantized-KV flash-attention patch set

This directory contains the ROCm/HIP portion of Nathanw1014's Strix Halo flash-attention work. The build integration applies it only to `gfx1151` by default.

Source commits:

- `6b03608e63f48c9371bf5f00423da413ac0288de` - expand `FLASH_ATTN_EXT` quantized-KV coverage.
- `2a24abc639763332c0ff32cbc03a78f669ae03a0` - dequantize q4_0/q8_0 K/V while loading the tile kernel and prefer that kernel when the GQA optimization applies.

The test patch is an exact `git format-patch` export from [Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp/tree/strix-halo-fa-fixes). The HIP patch is derived from the source commit with only a duplicated prototype comment removed; its code is unchanged.

The Vulkan/RADV commits from that branch are intentionally excluded. This repository produces ROCm/HIP artifacts, and the Vulkan changes have separate runtime and memory tradeoffs.

`PATCHSET` is the stable identifier recorded by local and CI builds. Increment it whenever any patch changes.
