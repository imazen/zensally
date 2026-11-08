"""Research-only padded head estimate; positive/missing geometry is uncertified."""
import math


def head_bounds(landmarks, width, height):
    if len(landmarks) != 33:
        raise ValueError('Pose Lite requires 33 points')
    required = landmarks[:13]
    if any(not all(math.isfinite(p[k]) for k in ['x', 'y', 'presence', 'visibility']) for p in required):
        return None, 'nonfinite facial/shoulder point'
    if any(min(p['presence'], p['visibility']) < .5 for p in required):
        return None, 'facial/shoulder confidence below0.5'
    face = required[:11]
    left, right = min(p['x'] for p in face), max(p['x'] for p in face)
    span = max(2.4*(right-left), .65*math.hypot(required[11]['x']-required[12]['x'],
                                            required[11]['y']-required[12]['y']))
    if span <= 0:
        return None, 'degenerate facial/shoulder span'
    bottom, center = max(p['y'] for p in face)+.25*span, (left+right)/2
    raw = [center-span/2, bottom-1.3*span, center+span/2, bottom]
    box = [max(0, math.floor(raw[0])), max(0, math.floor(raw[1])),
           min(width, math.ceil(raw[2])), min(height, math.ceil(raw[3]))]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None, 'no visible head region'
    return box, 'uncertified pose-derived padded head'
