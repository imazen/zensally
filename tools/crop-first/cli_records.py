"""Read bounded CLI evidence without retaining base64 tool-image history."""
import json


def read_records(path):
    rows=[]
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):raise ValueError('CLI rollout has an incomplete trailing record')
            r=json.loads(line);v=r.get('payload',{});kind=r['type']
            if kind=='event_msg' and v.get('type') in ['token_count','task_complete','error']:
                rows.append(r)
            elif kind=='turn_context':
                rows.append({'type':kind,'payload':{'model':v['model']}})
            elif kind=='response_item' and v.get('type')=='message' and v.get('role') in ['system','developer','user']:
                # Preserve text for in-turn prompt selection and prompt matching;
                # retain only image markers where the probe audits attachments.
                content=[{'type':'input_image'} if c.get('type')=='input_image' else c for c in v['content']]
                rows.append({'type':kind,'payload':{**v,'content':content}})
    return rows
