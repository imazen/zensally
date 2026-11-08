#!/usr/bin/env python3
"""Download official candidate models and inspect metadata only, never run inference."""
import argparse,hashlib,io,json,zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator
from tflite.TensorType import TensorType

SELECT=[('face_detector','blaze_face_short_range.tflite','float16'),
        ('face_detector','blaze_face_full_range.tflite','float16'),
        ('face_detector','blaze_face_full_range_sparse.tflite','float16'),
        ('object_detector','efficientdet_lite0.tflite','int8'),
        ('object_detector','ssd_mobilenet_v2.tflite','float16'),
        ('image_segmenter','selfie_segmenter.tflite','float16'),
        ('image_segmenter','selfie_segmenter_landscape.tflite','float16'),
        ('image_segmenter','selfie_multiclass_256x256.tflite','float32'),
        ('image_segmenter','hair_segmenter.tflite','float32'),
        ('pose_landmarker','pose_landmarker_lite.task','float16'),
        ('face_landmarker','face_landmarker.task','float16')]


def capture_documents(root):
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.current=None
        def handle_starttag(self,tag,attrs):
            if tag=='a':
                attrs=dict(attrs)
                if 'href' in attrs:self.current={'text':'','href':attrs['href']}
        def handle_data(self,data):
            if self.current is not None:self.current['text']+=data
        def handle_endtag(self,tag):
            if tag=='a' and self.current is not None:self.links.append(self.current);self.current=None
    documents=[]
    for page in dict.fromkeys(p for p,_,_ in SELECT):
        url='https://developers.google.com/edge/mediapipe/solutions/vision/'+page
        with urlopen(Request(url,headers={'User-Agent':'zensally-model-inventory/1'}),timeout=60) as reply:
            data=reply.read(8*1024*1024+1)
        if len(data)>8*1024*1024:raise ValueError('documentation exceeds capture bound')
        parser=Links();parser.feed(data.decode())
        for link in parser.links:link['href']=urljoin(url,link['href'])
        for name,value in [(page+'.html',data),(page+'-links.json',(json.dumps(parser.links,indent=2)+'\n').encode())]:
            with (root/name).open('xb') as f:f.write(value)
        assets=[]
        for link in parser.links:
            if 'storage.googleapis.com' in link['href'] and any(x in link['href'] for x in ['.tflite','.task']) and not any(x['href']==link['href'] for x in assets):assets.append(link)
        documents.append({'name':page,'source':url,'assets':assets})
    with (root/'official-assets.json').open('x') as f:json.dump(documents,f,indent=2);f.write('\n')


def sha(b):return hashlib.sha256(b).hexdigest()


def inspect_model(data):
    if len(data)<8 or data[4:8]!=b'TFL3':raise ValueError('not a TFLite graph')
    model=Model.GetRootAs(data,0)
    builtins={v:k for k,v in vars(BuiltinOperator).items() if isinstance(v,int)}
    types={v:k for k,v in vars(TensorType).items() if isinstance(v,int)}
    codes=[]
    for i in range(model.OperatorCodesLength()):
        op=model.OperatorCodes(i);code=op.BuiltinCode();custom=op.CustomCode()
        codes.append((custom.decode() if custom else builtins.get(code,'BUILTIN_'+str(code)))+'@'+str(op.Version()))
    graphs=[]
    for i in range(model.SubgraphsLength()):
        g=model.Subgraphs(i)
        def tensor(index):
            t=g.Tensors(index)
            return {'shape':[t.Shape(j) for j in range(t.ShapeLength())],'type':types.get(t.Type(),str(t.Type()))}
        ops=Counter(codes[g.Operators(j).OpcodeIndex()] for j in range(g.OperatorsLength()))
        graphs.append({'inputs':[tensor(g.Inputs(j)) for j in range(g.InputsLength())],
                       'outputs':[tensor(g.Outputs(j)) for j in range(g.OutputsLength())],
                       'operators':dict(sorted(ops.items())),
                       'sparse_tensors':sum(g.Tensors(j).Sparsity() is not None for j in range(g.TensorsLength()))})
    return {'tflite_schema_version':model.Version(),'graphs':graphs}


def inspect_asset(data):
    # TFLite metadata may append a ZIP of associated files. Inspect the TFL3
    # graph first; ZIP recognition alone does not imply a task bundle.
    if len(data)>=8 and data[4:8]==b'TFL3':
        return inspect_model(data)
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise ValueError('neither TFLite model nor task bundle')
    components=[]
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.file_size>64*1024*1024:raise ValueError('oversized task component')
            if info.filename.endswith('.tflite'):
                b=archive.read(info)
                components.append({'name':info.filename,'bytes':len(b),'sha256':sha(b),**inspect_model(b)})
    if not components:raise ValueError('task bundle contains no model graphs')
    return {'components':components}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--inspect-only',action='store_true');p.add_argument('--capture-docs',action='store_true')
    p.add_argument('--capture-only',action='store_true')
    a=p.parse_args();root=a.directory
    if a.capture_docs:
        root.mkdir(parents=True,exist_ok=True);capture_documents(root)
    if a.capture_only:
        if not a.capture_docs:raise ValueError('--capture-only requires --capture-docs')
        return
    if a.inspect_only:
        import time
        target=root/'inventory.json';rows=json.loads(target.read_text())
        for row in rows:
            if row['status']!='downloaded':continue
            data=(root/row['file']).read_bytes()
            if sha(data)!=row['sha256'] or len(data)!=row['bytes']:raise ValueError('downloaded model changed')
            row.pop('graphs',None);row.pop('components',None);row.update(inspect_asset(data))
        target.rename(target.with_name(target.name+'.bak-'+str(time.time_ns())))
        with target.open('x') as f:json.dump(rows,f,indent=2);f.write('\n')
        print(json.dumps({'reinspected':sum(r['status']=='downloaded' for r in rows),'network_requests':0}))
        return
    documents=json.loads((root/'official-assets.json').read_text());assets=root/'models';assets.mkdir(exist_ok=False);rows=[]
    for page,filename,quant in SELECT:
        document=next(d for d in documents if d['name']==page)
        matches=[r for r in document['assets'] if r['href'].endswith('/'+filename) and '/'+quant+'/' in r['href']]
        if len(matches)!=1:raise ValueError('model URL must be unambiguous: '+filename)
        link=matches[0];row={'name':link['text'].strip(),'documentation':document['source'],'url':link['href'],'declared_quantization':quant,'inference_measured':False}
        try:
            with urlopen(Request(link['href'],headers={'User-Agent':'zensally-model-inventory/1'}),timeout=60) as reply:
                data=reply.read(64*1024*1024+1);row['http']={k:reply.headers.get(k) for k in ['ETag','Last-Modified','x-goog-generation']};row['resolved_url']=reply.url
            if len(data)>64*1024*1024:raise ValueError('model exceeds inventory capture bound')
            path=assets/(quant+'-'+filename)
            with path.open('xb') as f:f.write(data)
            row.update(status='downloaded',file=str(path.relative_to(root)),bytes=len(data),sha256=sha(data))
            row.update(inspect_asset(data))
        except HTTPError as error:
            row.update(status='http_error',http_status=error.code)
            (assets/(quant+'-'+filename+'.http-error')).write_bytes(error.read(1024*1024))
        rows.append(row)
        with (root/'inventory.json').open('w') as f:json.dump(rows,f,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ['components','graphs']}),flush=True)
    print(json.dumps({'requested':len(rows),'downloaded':sum(r['status']=='downloaded' for r in rows),'failed':[r['name'] for r in rows if r['status']!='downloaded']}))


if __name__=='__main__':main()
