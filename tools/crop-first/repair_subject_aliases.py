#!/usr/bin/env python3
"""Apply an explicit reviewed ID alias map without changing cached provider bytes."""
import argparse
import copy
import json
from pathlib import Path

from direct_zones import canonical,sha
from subject_zones import validate


def repair(response,task,mapping):
    texts=[part for item in response['output'] for part in item.get('content',[]) if part['type']=='output_text']
    if len(texts)!=1:raise ValueError('one structured provider text required')
    original=json.loads(texts[0]['text']);result=copy.deepcopy(original)
    ids={r['id'] for r in result['rois']};subjects={r['id'] for r in result['rois'] if r['kind'] in ['person','animal','product']}
    if not mapping or set(mapping)&ids or not set(mapping.values())<=subjects or len(set(mapping.values()))!=len(mapping):
        raise ValueError('aliases must join distinct undefined IDs to distinct existing subjects')
    used=set()
    def rename(value):
        if value in mapping:used.add(value);return mapping[value]
        return value
    for roi in result['rois']:roi['owner_subject_id']=rename(roi['owner_subject_id'])
    for crop in result['crops']:
        for option in crop['options']:
            for key in ['selected_subject_ids','excluded_subject_ids','retained_roi_ids']:option[key]=[rename(value) for value in option[key]]
    if used!=set(mapping):raise ValueError('unused alias is not a justified repair')
    checked=copy.deepcopy(response)
    checked_texts=[part for item in checked['output'] for part in item.get('content',[]) if part['type']=='output_text']
    checked_texts[0]['text']=json.dumps(result);validate(checked,task)
    if [r['box'] for r in original['rois']]!=[r['box'] for r in result['rois']] or [[o['box'] for o in c['options']] for c in original['crops']]!=[[o['box'] for o in c['options']] for c in result['crops']]:
        raise ValueError('alias repair changed spatial coordinates')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['entry','task','aliases','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    for name,digest in json.loads((a.entry/'hashes.json').read_text()).items():
        if sha((a.entry/name).read_bytes())!=digest:raise ValueError('immutable provider cache differs')
    response=json.loads((a.entry/'response.json').read_text());task=json.loads(a.task.read_text());mapping=json.loads(a.aliases.read_text())
    try:validate(response,task)
    except ValueError as error:prior_error=str(error)
    else:raise ValueError('valid provider output does not need alias repair')
    result=repair(response,task,mapping);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'zones.json').write_text(json.dumps(result,indent=2)+'\n')
    record=dict(origin='explicit unblinded coordinator ID-alias correction',aliases=mapping,prior_validation_error=prior_error,
        provider_response_sha256=sha((a.entry/'response.json').read_bytes()),task_sha256=sha(a.task.read_bytes()),aliases_sha256=sha(a.aliases.read_bytes()),
        roi_and_crop_coordinates_unchanged=True,original_provider_cache_unchanged=True,strict_validator_passed=True,
        human_rated=False,training_eligible=False,face_coverage_certified=False,paid_api_calls=0)
    (a.output/'repair.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
