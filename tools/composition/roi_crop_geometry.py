"""Ranked subject boxes plus exact geometry, without a learned crop scorer.

All boxes use integer source-pixel [left, top, right, bottom] coordinates.
Protection covers supplied faces only; it cannot certify detector recall.
"""
from math import gcd

from subject_origin_projection import difference, union
from subject_policy import face_relation


def validate_box(width, height, box):
    if (len(box) != 4 or any(type(v) is not int for v in box)
            or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
        raise ValueError('positive in-source integer box required')


def envelope(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def place(width, height, cw, ch, required, faces, anchor):
    """Exact feasible origin closest to anchor; no supplied face may be cut.

    Sweep only face-relation boundaries, reusing the existing interval set
    primitives. Between boundaries face constraints are constant. This also
    supports selecting one subject while excluding other people completely.
    """
    lx, hx = max(0, required[2]-cw), min(width-cw, required[0])
    ly, hy = max(0, required[3]-ch), min(height-ch, required[1])
    if lx > hx or ly > hy:
        return None
    boundaries = {ly, hy+1}
    for left, top, right, bottom in faces:
        boundaries.update(max(ly, min(hy+1, v)) for v in
                          (top-ch+1, bottom-ch, top+1, bottom))
    best = None
    result = None
    ordered = sorted(boundaries)
    for first, stop in zip(ordered, ordered[1:]):
        y = max(first, min(stop-1, anchor[1]))
        forbidden = []
        for left, top, right, bottom in faces:
            if y+ch <= top or y >= bottom:
                continue
            if y <= top and y+ch >= bottom and right-left <= cw:
                forbidden.extend([(left-cw+1, right-cw-1), (left+1, right-1)])
            else:
                forbidden.append((left-cw+1, right-1))
        for low, high in difference([(lx, hx)], union(forbidden)):
            x = max(low, min(high, anchor[0]))
            key = (((x-anchor[0])*height)**2+((y-anchor[1])*width)**2, y, x)
            if best is None or key < best:
                best = key
                result = [x, y, x+cw, y+ch]
    return result


def solve(width, height, ratio, roi, faces, *, padding_percent=10):
    """Largest feasible crop, then a tight nested crop with context padding.

    Faces retained by the outer crop must also fit the inner crop, making
    every intervening nested rectangle safe for the supplied face boxes.
    A missing strict inner is explicit, never an invented Minimal/Maximal pair.
    """
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError('positive integer dimensions required')
    if type(padding_percent) is not int or not 0 <= padding_percent <= 100:
        raise ValueError('integer padding percent in 0..100 required')
    p, q = map(int, ratio.split(':'))
    if min(p, q) < 1:
        raise ValueError('positive aspect ratio required')
    divisor = gcd(p, q)
    p, q = p//divisor, q//divisor
    for box in [roi, *faces]:
        validate_box(width, height, box)
    maximum = min(width//p, height//q)
    minimum = max((roi[2]-roi[0]+p-1)//p, (roi[3]-roi[1]+q-1)//q)
    # Preserve the selected subject's relative source placement, not the
    # source center. This is an explicit placement default, not learned art.
    def anchor(cw, ch):
        return [((roi[0]+roi[2])*(width-cw))//(2*width),
                ((roi[1]+roi[3])*(height-ch))//(2*height)]
    outer = None
    for k in range(maximum, minimum-1, -1):
        outer = place(width, height, k*p, k*q, roi, faces, anchor(k*p, k*q))
        if outer is not None:
            break
    if outer is None:
        return dict(status='subject-does-not-fit-with-face-constraints', outer=None, inner=None)
    retained = [f for f in faces if face_relation(outer, f) == 'contained']
    required = envelope([roi, *retained])
    dx = ((required[2]-required[0])*padding_percent+99)//100
    dy = ((required[3]-required[1])*padding_percent+99)//100
    padded = [max(outer[0], required[0]-dx), max(outer[1], required[1]-dy),
              min(outer[2], required[2]+dx), min(outer[3], required[3]+dy)]
    k = max((padded[2]-padded[0]+p-1)//p, (padded[3]-padded[1]+q-1)//q)
    outer_k = (outer[2]-outer[0])//p
    if k >= outer_k:
        inner = None
    else:
        # Inside outer, containing all its retained faces also protects all
        # faces excluded by outer. No second face-relation search is needed.
        lx, hx = max(outer[0], padded[2]-k*p), min(padded[0], outer[2]-k*p)
        ly, hy = max(outer[1], padded[3]-k*q), min(padded[1], outer[3]-k*q)
        ax, ay = anchor(k*p, k*q)
        x, y = max(lx, min(hx, ax)), max(ly, min(hy, ay))
        inner = [x, y, x+k*p, y+k*q]
    return dict(status='nested-pair' if inner else 'single-frame-no-distinct-tight-crop',
                outer=outer, inner=inner,
                retained_faces=[i for i, f in enumerate(faces) if face_relation(outer, f) == 'contained'],
                excluded_faces=[i for i, f in enumerate(faces) if face_relation(outer, f) == 'excluded'],
                face_coverage_certified=False, artistic_quality_verified=False)


def rank_regions(regions, width, height):
    """Transparent default: confidence times square root of source area.

    Ranking estimates visible subject prominence, not semantic importance.
    The caller can supply an explicit subject instead of using this ordering.
    """
    for row in regions:
        validate_box(width, height, row['box'])
        if not 0 <= row['confidence'] <= 1:
            raise ValueError('finite confidence in 0..1 required')
    return sorted(regions, key=lambda r: (-r['confidence'] * (
        (r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1])/(width*height))**.5,
        r['kind'], r['box']))


if __name__ == '__main__':
    import json
    import sys
    request = json.load(sys.stdin)
    regions = rank_regions(request['regions'], request['width'], request['height'])
    result = [dict(subject=r, **solve(request['width'], request['height'], request['ratio'],
                                     r['box'], request.get('faces', []))) for r in regions]
    json.dump(dict(options=result, face_coverage_certified=False, artistic_quality_verified=False), sys.stdout, indent=2)
    sys.stdout.write('\n')
