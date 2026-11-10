# Region and placement evidence

All eight packets have independent NAS membership, size and SHA256 verification.
Local paths begin `/mnt/v/output/zensally/`; NAS copies begin
`/mnt/tower/output/zensally/`. Append the directory below. The NAS was verified
through `dev`; a similarly named local directory on another host is not proof
of a mounted NAS. No API spending, TEST inference or model promotion.

| Directory | Members | Payload bytes | SHA256SUMS.json SHA256 |
|---|---:|---:|---|
| region-context-scout/2026-11-09 |469|158709583|9b68ca73fb8e4fb27764290d5b5d7243d8323b52ab2df1fd4787de6c3726d610|
| subject-native-faces/2026-11-09 |47|3945448|20adfd7777f535cce3ad328992b1f56ae76f7b0b87048f9a299bcdc98789441f|
| subject-origin-projection/2026-11-09 |40|18567884|928449d1041b28068b0dc4f0c64faf6469babc6694a590502cf6bf1fefe4c3e1|
| subject-origin-projection-native/2026-11-09 |40|18706262|939c4fde9fa859becd64b874f9fe704c8fff1d2841a888f344c193c16ba0349d|
| subject-origin-projection-padded/2026-11-09 |44|37020720|df980a2c9848f330b8e98deb3e5e51a6d80d30663a879f21459c86e272765319|
| subject-origin-review/2026-11-09 |59|31208903|8d4fba951435be613cef20ea99b7eaef82c87d41a84b7b3753c4b0d989523d1b|
| subject-origin-review-padded/2026-11-09 |59|30934516|10b556adaf409caf1bd4eef762fd59c6a204ee1a2a5e957cae498c90b19398cc|
| subject-origin-review/2026-11-09-initial-status |43|5207615|616f8c94fd926ee271d5db099daedfab4216ca5c0772261e4a37fb3df9e1ffcb|

The initial viewer is an untested generated archive with unreliable status text,
not a reviewed screenshot packet. Original producers are retained under
`source/as-run`; later source snapshots do not overwrite those identities.
The other two viewers each pass704 browser checks and have four of ten saved
screenshots visually inspected. Their visible failures remain unqualified.

Recheck a packet with `just composition-verify-remote-packet dev <NAS path>`.
The two reports describe [region fitting](region_context_2026-11-09.md) and
[native face/placement controls](subject_origin_projection_2026-11-09.md).
