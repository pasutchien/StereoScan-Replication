# MxM window reduction: naive vs. bucketed matching

Frame 0 -> 1, `blob_max` class, window_radius=50, bin_size=50, fine NMS radius=3, coarse NMS radius=9.

| approach | matches | time (s) | occupied bins | overlap w/ naive |
|---|---|---|---|---|
| naive (full MxM window) | 1235 | 0.991 | - | - |
| bucketed, min_bin_samples=1 | 265 | 0.657 | 105 | 263/265 |
| bucketed, min_bin_samples=2 | 660 | 0.871 | 51 | 658/660 |
| bucketed, min_bin_samples=3 | 913 | 0.963 | 26 | 913/913 |
| bucketed, min_bin_samples=5 | 1224 | 1.046 | 1 | 1224/1224 |
| bucketed, min_bin_samples=8 | 1235 | 1.060 | 0 | 1235/1235 |

## Conclusion

Even though the bucketed approach can speed up matching, the number of matches drops a lot without finding many new matches in return. We're sticking with the full MxM search window rather than the coarse-pass bucketed narrowing.
