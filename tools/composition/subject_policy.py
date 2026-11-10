"""Experimental subject-choice geometry; complete face review remains explicit."""
import math


def valid_box(box):
    return (len(box)==4 and all(type(v) in [int,float] and math.isfinite(v) for v in box)
        and box[0]<box[2] and box[1]<box[3])


def face_relation(crop,face):
    """A face is wholly contained, wholly excluded, or intersected by a crop edge."""
    if not valid_box(crop) or not valid_box(face):raise ValueError('finite positive crop and face boxes required')
    if crop[0]<=face[0] and crop[1]<=face[1] and crop[2]>=face[2] and crop[3]>=face[3]:return 'contained'
    if crop[2]<=face[0] or face[2]<=crop[0] or crop[3]<=face[1] or face[3]<=crop[1]:return 'excluded'
    return 'cut'


def assess(crop,faces,selected):
    if not selected or len(set(selected))!=len(selected) or any(type(i) is not int or not 0<=i<len(faces) for i in selected):
        raise ValueError('one or more distinct selected face indices required')
    relations=[face_relation(crop,face) for face in faces]
    return dict(relations=relations,selected_faces_contained=all(relations[i]=='contained' for i in selected),
        cut_faces=[i for i,state in enumerate(relations) if state=='cut'],
        excluded_faces=[i for i,state in enumerate(relations) if state=='excluded'],
        geometry_compatible=all(relations[i]=='contained' for i in selected) and 'cut' not in relations,
        automatic_face_coverage_certified=False)


def admissible(crop,faces,selected,*,complete_faces_reviewed=False):
    """Choose a whole face/group while preserving every other intersected face."""
    if type(complete_faces_reviewed) is not bool:raise ValueError('explicit reviewed-coverage boolean required')
    result=assess(crop,faces,selected)
    return complete_faces_reviewed and result['geometry_compatible']
