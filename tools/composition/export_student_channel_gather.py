#!/usr/bin/env python3
"""Qualify the same border sampler using contiguous16-channel Gather rows."""
import json
from pathlib import Path
import sys

import torch
from torch import nn

import export_unsplash_ranker
import gather_grid


class ChannelBorderGather(nn.Module):
    def forward(self, table, grid):
        batch, channels, height, width = table.shape
        x = ((grid[..., 0]+1)*((width-1)*.5)).clamp(0, width-1)
        y = ((grid[..., 1]+1)*((height-1)*.5)).clamp(0, height-1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0+1).clamp_max(width-1), (y0+1).clamp_max(height-1)
        dx, dy = (x-x0)[:, None], (y-y0)[:, None]
        # This exporter only admits the hash-bound batch1 source graphs.
        # One index selects an entire contiguous channel row, without expanding
        # the integer index tensor16 times for GatherElements.
        flat = table.permute(0, 2, 3, 1).reshape(height*width, channels)

        def take(xx, yy):
            values = flat.index_select(0, (yy*width+xx).flatten())
            return values.reshape(batch, grid.shape[1], grid.shape[2], channels).permute(0, 3, 1, 2)

        top = take(x0, y0)*(1-dx)+take(x1, y0)*dx
        bottom = take(x0, y1)*(1-dx)+take(x1, y1)*dx
        return top*(1-dy)+bottom*dy


def main():
    if '--gather-pool' not in sys.argv:
        raise ValueError('explicit gather-pool qualification flag required')
    original = gather_grid.BorderGridGather
    gather_grid.BorderGridGather = ChannelBorderGather
    try:
        export_unsplash_ranker.main()
    finally:
        gather_grid.BorderGridGather = original
    # The underlying gate remains identical; distinguish lowering/layout identity.
    output = Path(sys.argv[4])
    (output/'LOWERING_VARIANT.json').write_text(json.dumps(dict(
        variant='batch1 contiguous16-channel Gather rows', index_channel_broadcast=False,
        original_student_weights_unchanged=True, training_math_changed=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
