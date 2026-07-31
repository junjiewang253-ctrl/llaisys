#!/usr/bin/env python3
import json

import numpy as np
import torch

from qwen2.diagnose_rms_reduction import cascade_sum_squares


def main():
    dimension = 896
    epsilon = 1e-6
    for seed in range(10000):
        generator = torch.Generator().manual_seed(seed)
        source = torch.randn((dimension,), generator=generator).to(
            torch.bfloat16
        )
        avx512_sum = cascade_sum_squares(source, 16)
        avx2_sum = cascade_sum_squares(source, 8)
        if avx512_sum == avx2_sum:
            continue
        values = source.float().numpy()
        avx512_mean = np.float32(avx512_sum / np.float32(dimension))
        avx512_inverse = np.float32(
            np.float32(1.0)
            / np.sqrt(np.float32(avx512_mean + np.float32(epsilon)))
        )
        avx512 = torch.from_numpy(
            np.asarray(values * avx512_inverse, dtype=np.float32)
        ).to(torch.bfloat16)
        expected = (
            source.float()
            * torch.rsqrt(
                source.float().pow(2).mean() + epsilon
            )
        ).to(torch.bfloat16)
        mismatches = (avx512 != expected).nonzero().flatten()
        if mismatches.numel():
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "dimension": dimension,
                        "epsilon": epsilon,
                        "mismatch_count": int(mismatches.numel()),
                        "first_index": int(mismatches[0]),
                        "avx512_sum": float(avx512_sum),
                        "avx2_sum": float(avx2_sum),
                        "avx512_value": float(avx512[mismatches[0]]),
                        "torch_value": float(expected[mismatches[0]]),
                    },
                    sort_keys=True,
                )
            )
            return
    raise RuntimeError("no deterministic fixture found")


if __name__ == "__main__":
    main()
