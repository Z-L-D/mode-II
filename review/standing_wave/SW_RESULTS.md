# Standing-wave re-test: results (2026-10-08)

**Preregistration:** `SW_PREREGISTRATION.md`, SHA-256 e9c74d7f…0b1dab, frozen 2026-10-08T10:45:59Z.

**Scripts** (all in this folder):
- `sw_load.py`: analyzer `DataManager.load_paths`, unchanged.
- `sw_tests.py`: tests P1–P6.
- `sw_posthoc.py` and `sw_posthoc2.py`: labelled post hoc.
- `sw_figure.py`.

**Outputs:** `out/`.

## Verdict

**The standing wave does not stand.**

What the Sept 15 analysis found is real and reproduces. It is a fixed depth shape in Z-Image MLP whose amplitude barely changes through denoising. It does not oscillate, so it is not a wave, standing or travelling.

Its dips sit near L 9, 12, 15, 18 and 21, and the deepest is at L14–15. Base and Turbo share this shape, but the regular 3-layer node spacing beats the nulls only in Base.

## Results

**P1. Reproduction (passes).** The rebuilt field matches the Sept 15 numbers. Geometry G-z, MLP:

| | ZIB | ZIT | Sept 15 |
|---|---|---|---|
| \|Ψ\| minima | 4 9 12 15 18 21 24 | 2 4 9 12 14 18 21 23 27 | ZIB 1 4 9 12 15 18 21 24 28; ZIT 1 4 9 12 15 18 21 23 |
| Global minimum | L15 | L14 | L15 both |
| Phase-gradient reversals | 13 | 13 | about 13 |
| Mode-1 energy, full | 91% | 87% | 88% / 83% |
| Mode-1 energy, σ ≤ 0.75 | 97% | 98% | about 96% |

**P2. Oscillation (fails in all 7 channels).**
- The mode's complex amplitude A(σ) never changes sign. Its phase moves only 0.42 rad over the whole trajectory, and |A| stays within 0.89–1.00 of its maximum, with a CV of 3–4%.
- High mode-1 energy is therefore mostly the time-mean shape. That shape carries 86% (ZIT MLP) and 89% (ZIB MLP) of the field energy.
- **Post hoc:** with the time-mean removed, the remaining part is a monotone drift, not an oscillation.
  - Its leading coefficient has Spearman ρ vs step of −0.98 (ZIT) and −0.96 (ZIB).
  - It has one sign change, which removing the mean forces.
  - All 7 channels behave the same way. SD3.5 L shows 2 sign changes, but with ρ = 0.71, not an oscillation.
- The nodes do not move: median displacement after the pinch is 0 layers. This matches the zero-shift depth registration already in Mode-I.

**P3. Node regularity against nulls (fails as a two-model result).** The statistic is the longest run of ΔL = 3 spacings between |Ψ| minima.

| | Run | p, phase-randomised Ψ | p, profile-randomised |
|---|---|---|---|
| ZIB MLP | 5 (9-12-15-18-21-24) | 0.005 | 0.008 |
| ZIT MLP | 1 | 0.86 | 0.95 |

- ZIT fails because its L14 and L15 dips are nearly tied (0.39 vs 0.40 of the median). Otherwise the ZIT and ZIB |Ψ| profiles are almost identical (figure, panel a).
- No attention channel beats the null: p = 0.15–0.71 for ZIT/ZIB attention, Mochi and SD3.5 M/L.

**P4. Coordinate robustness (passes, but tells us little).**
- At least 4 of the 5 central nodes are recovered within ±1 layer under raw units and under the entropy geometry, for both MLPs.
- **Post hoc:** the same criterion recovers 4.1 of 5 on average in the nulls (p = 0.32–0.34). Minima about 3 layers apart with a ±1 tolerance hit most targets by chance.

**P5. Location of the global minimum.**
- ZIB MLP: L15 in 500/500 prompt bootstraps under both geometries.
- ZIT MLP: L14 in 489/500 (G-z) and L15 in 500/500 (G-ent).
- Other channels put it at L20–L38, so it is not a common depth.

**P6. Generality.**
- Every channel has a highly separable, fixed depth shape (mode-1 energy 0.71–0.93 full, 0.92–0.99 after the pinch).
- None has a node lattice beyond the nulls.

**Earlier negatives that still stand** (Sept 15 §47–49):
- No 1D wave operator fits the modes (residual 0.93–0.95).
- The L15 "inverse potential" moves to L11/L13 when the gap is swapped for entropy.

## What survives, and where it goes

**What survives.** Z-Image Base and Turbo share a fixed MLP depth profile with sharp dips near L 9, 12, 15, 18 and 21, deepest at L14–15. This is one more instance of the depth-shape inheritance that Mode-I already reports (registration, tangent 0.94–0.999).

**Not to be claimed:**
- a wave, standing or travelling;
- nodes or a node lattice;
- a "throat";
- a universal depth.

**Papers:** no change to Mode-I or Mode-II. The claim register (T6) already excludes it.

**Possible future test:** the regular Base dips could be tested prospectively on a third Z-Image checkpoint, or by capturing at L14/L15 with finer depth sampling.

## Addendum: the May LLM version (ARC-AGI, Qwen3-8B / Qwen2.5-7B)

Details are in `may_arc/MAY_WAVE_RESULTS.md`. This addendum is post hoc.
- **Wave quantization:** 12/50, against 16/50 for uniform 8-bit, despite more bits (McNemar p = 0.29). It also had no random-layer control.
- **Node skipping:** 0/50 is a crash, not a measurement. `IdentityLayer` returns a tuple, so the model raised an error and an exception handler hid it.
- **The Qwen2.5-7B profile:** a fixed, reproducible depth pattern (archive r = 0.58–0.62; sign consistency 0.99–1.00 across generations). It has no period (lag-1 autocorrelation 0.04; p = 0.88) and no node/antinode difference.
- **Next step:** `may_arc/wave_layer_importance.py` is the decisive per-layer test.
