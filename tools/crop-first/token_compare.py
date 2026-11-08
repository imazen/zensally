#!/usr/bin/env python3
"""Audit matched initial CLI image requests; do not claim API image-only billing."""
import argparse,hashlib,json
from pathlib import Path
from cli_records import read_records
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    cases=json.loads(a.cases.read_text());records=[];common=None
    for case in cases:
        rows=read_records(Path(case['rollout']))
        complete=[r for r in rows if r['type']=='event_msg' and r.get('payload',{}).get('type')=='task_complete']
        if len(complete)!=1 or complete[0]['payload']['last_agent_message'].strip()!='OK':raise ValueError('expected one OK-only probe turn')
        if any(r['type']=='response_item' and r.get('payload',{}).get('type') in ['custom_tool_call','function_call'] for r in rows):raise ValueError('probe called a tool')
        messages=[];images=0
        for r in rows:
            v=r.get('payload',{})
            if r['type']=='response_item' and v.get('type')=='message' and v.get('role') in ['system','developer','user']:
                images+=sum(c.get('type')=='input_image' for c in v['content'])
                messages.append({'role':v['role'],'content':[c for c in v['content'] if c.get('type')!='input_image']})
        if images!=1:raise ValueError('expected exactly one image attachment')
        normalized=json.dumps(messages,sort_keys=True).replace(case['image'],'INPUT_IMAGE_PATH')
        digest=hashlib.sha256(normalized.encode()).hexdigest()
        if common is None:common=digest
        if digest!=common:raise ValueError('non-image prompt changed beyond attachment path')
        usage=next(r['payload']['info']['total_token_usage'] for r in reversed(rows) if r['type']=='event_msg' and r.get('payload',{}).get('type')=='token_count')
        with Image.open(case['image']) as im:dimensions=list(im.size)
        records.append({'size':dimensions,'image_sha256':hashlib.sha256(Path(case['image']).read_bytes()).hexdigest(),'total_cli_usage':usage})
    result={'model':'gpt-6.1-terra','non_image_prompt_sha256_after_path_normalization':common,'measure':'total input tokens for matched fresh CLI requests; filenames differ, image-only usage meter unavailable','records':records,'billed_amount_known':False,'quality_or_face_stability_measured':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
