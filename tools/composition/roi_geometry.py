"""Exact crop coordinates and explicit visible-box projection for ROI probes."""
from fractions import Fraction
from math import ceil, floor


def person_views(xywh, preview_shape, native_shape):
    pw, ph = preview_shape
    nw, nh = native_shape
    x, y, w, h = xywh
    if min(pw, ph, nw, nh, w, h) <= 0:
        raise ValueError('nonpositive source or person rectangle')
    left, top = floor(Fraction(x*nw, pw)), floor(Fraction(y*nh, ph))
    right, bottom = ceil(Fraction((x+w)*nw, pw)), ceil(Fraction((y+h)*nh, ph))
    dx, dy = ceil(Fraction(right-left, 10)), ceil(Fraction(bottom-top, 10))
    views = []
    for name, end in [('person', bottom), ('upper', top+ceil(Fraction(9*(bottom-top), 20)))]:
        box = [max(0, left-dx), max(0, top-dy), min(nw, right+dx), min(nh, end+dy)]
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError('empty visible person crop')
        views.append((name, box))
    return views


def project_face(xywh, bounds, shape, native_shape):
    x, y, w, h = xywh
    left, top, right, bottom = bounds
    ow, oh = shape
    nw, nh = native_shape
    if min(ow, oh, w, h, right-left, bottom-top) <= 0:
        raise ValueError('invalid face/ROI dimensions')
    raw = [left+x*(right-left)/ow, top+y*(bottom-top)/oh,
           left+(x+w)*(right-left)/ow, top+(y+h)*(bottom-top)/oh]
    visible = [max(0, raw[0]), max(0, raw[1]), min(nw, raw[2]), min(nh, raw[3])]
    if visible[0] >= visible[2] or visible[1] >= visible[3]:
        visible = None
    return dict(raw_native_box=raw, visible_native_box=visible)
