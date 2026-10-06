"""Scalar-only optimizer diagnostics; no tensor, RNG or optimization changes."""
import numpy as np


def optimizer_summary(records):
    def quantiles(values):
        return dict(zip(('min', 'median', 'p95', 'max'), map(float, np.quantile(values, [0, .5, .95, 1]))))
    return dict(steps=len(records),
        preclip_norm=quantiles([r['preclip_norm'] for r in records]),
        clip_scale=quantiles([r['clip_scale'] for r in records]),
        clipped_step_fraction=float(np.mean([r['clip_scale'] < 1 for r in records])),
        group_preclip_norm={name: quantiles([r['group_preclip_norm'][name] for r in records])
                           for name in records[0]['group_preclip_norm']},
        scope='before_global_gradient_clip; scale is PyTorch clip coefficient; not parameter update magnitude')
