# Dashboard override — NPK Spectral Bench

## Direction

Scientific instrument console for soil spectroscopy researchers. The signature is a live-looking MIR trace running through the hero with a calibrated 600–4000 cm⁻¹ ruler. It encodes the actual input rather than decorating the page.

## Tokens

- Instrument ink `#0B132B`, panel `#13213C`, paper `#F4F7F6`, line `#CBD7D5`.
- Nitrogen `#168657`, phosphorus `#C06C00`, potassium `#6D48C7`, spectral cyan `#008C95`.
- Display: Barlow Condensed. Body: Fira Sans. Readouts: Fira Code.
- Squared 3–6 px corners, dense panels, strong focus rings, no decorative animation.

## Layout

```text
+-- dark instrument hero: thesis ----------+-- MIR trace + ruler --+
+-- context warning ------------------------------------------------+
+-- sidebar navigation --+-- dense analytical workspace -----------+
```

Cards are quiet and charts carry the color. N/P/K color is never the sole carrier of meaning; labels and line styles remain visible.

