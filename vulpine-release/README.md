# vulpine-release

The nine gpui-fast crates in their published form (package names `gpui-fast`, `gpui-fast-macos`, `gpui-fast-apple`,
...), generated from this branch's `crates/` by the release script, unchanged:

```
bun script/prepare-release.ts --tag v<version>
rm -rf vulpine-release && cp -R target/release-workspace vulpine-release
```

then this README restored. It exists so a Cargo `[patch.crates-io]` entry can take the crates from git by rev
(`gpui-fast = { git = "https://github.com/voidfab/gpui-fast", rev = "..." }`): the source workspace's packages are
named `gpui`, `gpui_macos`, ... at versions that do not match the published ones, so Cargo would not find them.
Regenerate it in the branch's last commit whenever a commit below it changes `crates/`; vulpine's
`tools/apps/upstream_patches.py verify` checks that it matches a fresh `prepare-release` of the tagged commit.

Not part of longbridge/gpui-fast. The vulpine patches are the commits on this branch above the upstream tag; each
names its vulpine item and the patch file it was ported from.
