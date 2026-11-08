"""Photographic composition scope; category tags are screening hints, not truth."""

NAME = 'people-commerce-v1'
PHOTO_CATEGORIES = {'people', 'general', 'nature', 'interiors', 'food'}
GENERATED_CATEGORIES = {'products', 'clipart', 'illustrations', 'renders'}
DOCUMENT_CATEGORIES = {'epa', 'noaa', 'nps', 'plots', 'mobile-screenshots',
                       'web-screenshots', 'scans-text', 'scans-patents', 'scans-illustrations'}

RUBRIC = '''This is the people-commerce-v1 PHOTOGRAPHIC composition study: news/editorial photographs, sales/lifestyle imagery, and ecommerce/product-in-use images. Documents, charts, scans, screenshots, and artwork reproductions are out of scope.
People are the priority: preserve every visible face/head, including small/background faces, profiles, and partially occluded faces. Preserve useful gestures, gaze direction, interactions, and the context that explains the scene; an attractive crop that changes an editorial event's meaning is unacceptable. Avoid awkward new cuts through joints, hands holding objects, or the person/product relationship. A distant person must not disappear merely because a closer person dominates. Existing source-edge truncation is not permission to remove more of a face.
For pets and other animals, preserve faces and the identifying subject/action; consider ears, paws, wings, and tails where composition needs them. For plants and scenery, retain intentional focal structure and environmental context. For ecommerce, keep the featured product, relevant packaging/branding, and its interaction with people readable; incidental photographed signs are context, not documents.
Natural-looking subject scale, breathing/looking room, and intentional negative space matter. Sales copy space may be valuable but is not mandatory and must not override face preservation or scene meaning. Do not invent an advertising layout, a news event, or demographic identities. Judge the actual photograph. Preserve uncertainty and allow contain rather than forcing a bad crop.'''


def prompt(task, legacy, document_sentence):
    profile = task.get('composition_profile')
    if profile is None:
        return legacy
    if profile != NAME:
        raise ValueError('unknown composition profile')
    if legacy.count(document_sentence) != 1:
        raise ValueError('legacy document instruction changed; review photographic profile')
    return legacy.replace(document_sentence, '') + '\n' + RUBRIC


def inherit(source, task):
    if 'composition_profile' in source:
        if source['composition_profile'] != NAME:
            raise ValueError('unknown composition profile')
        task['composition_profile'] = NAME
    return task


def category_scope(category):
    if category in PHOTO_CATEGORIES:
        return 'photographic_screening_candidate'
    if category in GENERATED_CATEGORIES:
        return 'generated_auxiliary_only'
    if category in DOCUMENT_CATEGORIES:
        return 'excluded_document_screen_chart_scan'
    return 'excluded_outside_photographic_scope'
