#!/usr/bin/env python3
"""Check native stock/reduced outputs; shared diagnostic tolerance stays fixed."""
import json
from pathlib import Path
import struct
import sys

a, b = (Path(v)/'outputs.f32' for v in sys.argv[1:])
aa, bb = a.read_bytes(), b.read_bytes()
if len(aa) != len(bb) or len(aa) != 134400*4:
    raise ValueError('complete matching YuNet output payload required')
maximum = 0.; bad = 0
for (x,), (y,) in zip(struct.iter_unpack('<f', aa), struct.iter_unpack('<f', bb)):
    import math
    if not math.isfinite(x) or not math.isfinite(y): raise ValueError('nonfinite output')
    delta = abs(x-y); maximum = max(maximum, delta); bad += delta > 1e-5+1e-5*abs(x)
result = dict(scalars=len(aa)//4,bits_exact=aa==bb,max_absolute=maximum,tolerance_violations=bad)
(b.parent/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if bad: raise ValueError('native diagnostic numeric tolerance failed')
