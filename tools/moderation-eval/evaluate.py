#!/usr/bin/env python3
"""Pinned, offline-after-fetch moderation model probes and zenbench workers."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import pwd
import statistics
import subprocess
import sys
import time
import urllib.request


MODELS = {
    "swift-xs": ("OwenElliott/image-safety-classifier-xs", "54f4560bd9c5ee92d45dc30418a8f8680e80de6d"),
    "swift-s": ("OwenElliott/image-safety-classifier-s", "015042b0eab17f1b17f2986527386346fb0d94be"),
    "marqo": ("", "0c26ec22111b83f106d72a55f611ec35962bcb65"),
    "freepik": ("", "15b85477e4fd2000db76ae9aae0f89a72f95e2e3"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def fetch(root):
    repo_overrides = {}
    repos_file = root / "repositories.json"
    if repos_file.exists():
        repo_overrides = json.loads(repos_file.read_text())
    for key, (repo, rev) in MODELS.items():
        dest = root / "models" / key
        dest.mkdir(parents=True, exist_ok=True)
        manifest = dest / "FETCH.json"
        if not repo:
            repo = repo_overrides.get(key, "")
            if not repo and manifest.exists():
                repo = json.loads(manifest.read_text()).get("repository", "")
            if not repo:
                raise ValueError(f"supply repository slug for {key} in {repos_file}")
        url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
        with urllib.request.urlopen(url, timeout=60) as r:
            listing = json.load(r)
        selected = [x for x in listing if x["path"] in ("config.json", "model.safetensors", "README.md")
                    or x["path"].endswith(".onnx")]
        records = []
        for item in selected:
            name = item["path"]
            path = dest / name
            expected = item.get("lfs", {}).get("oid")
            source = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + ".partial-" + str(time.time_ns()))
                with urllib.request.urlopen(source, timeout=120) as r, partial.open("xb") as f:
                    while block := r.read(1024 * 1024):
                        f.write(block)
                if expected and digest(partial) != expected:
                    raise ValueError(f"hash mismatch: {partial}")
                partial.rename(path)
            sha = digest(path)
            if expected and sha != expected:
                raise ValueError(f"existing artifact hash mismatch: {path}")
            records.append(dict(path=str(path), source=source, sha256=sha, bytes=path.stat().st_size))
            print(f"fetched {key} {name} {path.stat().st_size} {sha}", flush=True)
        manifest = dest / "FETCH.json"
        value = dict(repository=repo, revision=rev, files=records)
        if manifest.exists():
            if json.loads(manifest.read_text()) != value:
                raise ValueError(f"changed fetch manifest: {manifest}")
        else:
            save(manifest, value)


def environment():
    packages = {}
    for name in ("torch", "torchvision", "timm", "onnx", "onnxruntime", "numpy", "Pillow", "safetensors"):
        packages[name] = importlib.metadata.version(name)
    import numpy, torch, onnxruntime
    actual = {m.__name__: dict(version=m.__version__, file=m.__file__) for m in (numpy,torch,onnxruntime)}
    return dict(host=platform.node(), platform=platform.platform(), python=sys.version,
                executable=sys.executable, packages=packages,
                actual_modules=actual, affinity=sorted(os.sched_getaffinity(0)), script_sha256=digest(__file__))


def runtime(threads):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return torch


def native(root, key):
    import timm
    from safetensors.torch import load_file
    if key == "freepik":
        # The CPU SDPA path cannot be represented by this legacy exporter in
        # torch 2.15 (is_causal becomes a Tensor). Use timm's explicit attention
        # for both the native reference and its export; preserve the weights.
        timm.layers.set_fused_attn(False)
    directory = root / "models" / key
    cfg = json.loads((directory / "config.json").read_text())
    model = timm.create_model(cfg["architecture"], pretrained=False,
                              num_classes=cfg["num_classes"], global_pool=cfg["global_pool"])
    state = load_file(directory / "model.safetensors", device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def source_images(manifest):
    from PIL import Image, ImageOps
    data = json.loads(manifest.read_text())
    images = []
    for item in data["items"]:
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"input changed: {path}")
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    if not images:
        raise ValueError("empty input manifest")
    return data, images


def preprocessing(cfg):
    from timm.data import create_transform, resolve_data_config
    return create_transform(**resolve_data_config(pretrained_cfg=cfg["pretrained_cfg"]), is_training=False)


def session(path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def probe(root, key, manifest, output, threads):
    import numpy as np
    import onnx
    from PIL import Image
    torch = runtime(threads)
    data, images = source_images(manifest)
    model, cfg = native(root, key)
    transform = preprocessing(cfg)
    inputs = [transform(im).unsqueeze(0) for im in images]
    witness = root / "models" / key / "witness-input.f32"
    if not witness.exists():
        with witness.open("xb") as f:
            f.write(inputs[0].numpy().astype("<f4").tobytes())
    with torch.inference_mode():
        native_scores = [model(x).softmax(-1).numpy().tolist()[0] for x in inputs]
    record = dict(model=key, source=MODELS[key], environment=environment(),
                  input_manifest_sha256=digest(manifest), native_scores=native_scores,
                  labels=cfg.get("label_names", cfg["pretrained_cfg"].get("label_names")),
                  native_preprocessing=cfg["pretrained_cfg"],
                  explicit_attention=(key == "freepik"))
    own_onnx = list((root / "models" / key / "onnx").glob("*.onnx"))
    if own_onnx:
        published = own_onnx[0]
        graph = onnx.load(str(published))
        record["publisher_onnx"] = dict(path=str(published), sha256=digest(published),
            operations=dict(Counter(n.op_type for n in graph.graph.node)),
            inputs=[dict(name=i.name, shape=[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in graph.graph.input])
        ort = session(published, threads)
        name = ort.get_inputs()[0].name
        raw = [np.asarray(im.resize((224,224), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2,0,1)[None] for im in images]
        results = [ort.run(None, {name: x})[0].tolist()[0] for x in raw]
        record["publisher_documented_scores"] = results
        # Match the exact native crop/resample pixels, while leaving normalization
        # to the published graph, to isolate the documented resize difference.
        mean = torch.tensor(cfg["pretrained_cfg"]["mean"])[None,:,None,None]
        std = torch.tensor(cfg["pretrained_cfg"]["std"])[None,:,None,None]
        matched = [((x * std + mean) * 255).numpy() for x in inputs]
        aligned = [ort.run(None, {name:x})[0].tolist()[0] for x in matched]
        record["publisher_aligned_scores"] = aligned
        record["publisher_aligned_max_abs_error"] = float(np.max(np.abs(np.array(aligned)-native_scores)))
        record["publisher_documented_max_abs_error"] = float(np.max(np.abs(np.array(results)-native_scores)))
    export = root / "models" / key / "native-fp32.onnx"
    if not export.exists():
        with torch.inference_mode():
            torch.onnx.export(model, inputs[0], str(export), input_names=["pixels"],
                              output_names=["logits"], opset_version=17, dynamo=False)
    ort = session(export, threads)
    ort_scores=[]
    for x in inputs:
        logits=ort.run(None, {"pixels":x.numpy()})[0]
        exp=np.exp(logits-logits.max(axis=-1,keepdims=True))
        ort_scores.append((exp/exp.sum(axis=-1,keepdims=True)).tolist()[0])
    error=float(np.max(np.abs(np.array(ort_scores)-native_scores)))
    record.update(onnx_path=str(export), onnx_sha256=digest(export), onnx_scores=ort_scores,
                  native_onnx_max_abs_error=error,
                  onnx_operations=dict(Counter(n.op_type for n in onnx.load(str(export)).graph.node)))
    save(output, record)
    if error > 0.0001:
        raise ValueError(f"native/export parity failed: {key} {error}; evidence: {output}")
    print(json.dumps(dict(model=key, parity_max_abs_error=error,
                         publisher_aligned_max_abs_error=record.get("publisher_aligned_max_abs_error")),allow_nan=False),flush=True)


def probes(root, manifest, output, threads):
    output.mkdir(parents=True, exist_ok=False)
    completed = {}
    progress = output / "progress.json"
    for key in MODELS:
        log = output / (key + ".log")
        command = [sys.executable, str(Path(__file__).resolve()), "probe", "--root", str(root),
                   "--model", key, "--manifest", str(manifest), "--output", str(output / (key + ".json")),
                   "--threads", str(threads)]
        with log.open("x") as f:
            child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
            progress.write_text(json.dumps(dict(stage="probe", model=key, pid=child.pid,
                supervisor_pid=os.getpid(), completed=completed, required=list(MODELS), log=str(log),
                checked_at=datetime.now(timezone.utc).isoformat(), script_sha256=digest(__file__)), indent=2)+"\n")
            print(f"probe {key} pid={child.pid} log={log}", flush=True)
            rc = child.wait()
        completed[key] = dict(returncode=rc, log=str(log))
        print(f"probe {key} returncode={rc}", flush=True)
    progress.write_text(json.dumps(dict(stage="finished", supervisor_pid=os.getpid(), completed=completed,
        checked_at=datetime.now(timezone.utc).isoformat()), indent=2)+"\n")
    if any(x["returncode"] for x in completed.values()):
        raise RuntimeError(f"probe failures retained in {output}")


def worker(root, key, manifest, output, threads, count, warmup, stage):
    import numpy as np
    from PIL import Image, ImageOps
    torch=runtime(threads)
    data,images=source_images(manifest)
    cfg=json.loads((root/"models"/key/"config.json").read_text())
    transform=preprocessing(cfg)
    inputs=[transform(im).unsqueeze(0).numpy() for im in images]
    encoded=[Path(item["path"]).read_bytes() for item in data["items"]]
    start=time.perf_counter_ns()
    ort=session(root/"models"/key/"native-fp32.onnx",threads)
    session_ns=time.perf_counter_ns()-start
    def run(i):
        if stage=="decode-preprocess-inference":
            with Image.open(io.BytesIO(encoded[i % len(encoded)])) as image:
                rgb=ImageOps.exif_transpose(image).convert("RGB")
                x=transform(rgb).unsqueeze(0).numpy()
        elif stage=="preprocess-inference":
            x=transform(images[i % len(images)]).unsqueeze(0).numpy()
        else:
            x=inputs[i % len(inputs)]
        return ort.run(None,{"pixels":x})[0]
    for i in range(warmup):run(i)
    samples=[]
    logits=[]
    for i in range(count):
        start=time.perf_counter_ns()
        value=run(i)
        samples.append((time.perf_counter_ns()-start)/1e6)
        if not np.isfinite(value).all():raise ValueError("non-finite output")
        logits.append(value.tolist())
    output.mkdir(parents=True,exist_ok=True)
    save(output/"worker.json",dict(model=key,stage=stage,threads=threads,count=count,warmup=warmup,
        input_manifest_sha256=digest(manifest),onnx_sha256=digest(root/"models"/key/"native-fp32.onnx"),
        session_ms=session_ns/1e6,samples_ms=samples,p50_ms=statistics.median(samples),
        p95_ms=float(np.percentile(samples,95)),logits=logits,environment=environment()))
    # One vector belongs to one fresh process; zenbench keeps that clustering.
    for sample in samples:print(f"elapsed_ms={sample:.9f}")


def check(before,after):
    import math
    a=json.loads((before/"worker.json").read_text())
    b=json.loads((after/"worker.json").read_text())
    for field in ("stage","threads","count","warmup","input_manifest_sha256"):
        if a[field]!=b[field]:raise ValueError(f"workload mismatch: {field}")
    for item in (a,b):
        if len(item["samples_ms"])!=item["count"]:raise ValueError("incomplete samples")
        if not all(math.isfinite(x) and x>0 for x in item["samples_ms"]):raise ValueError("invalid timing")
    if a["model"]==b["model"]:
        if a.get("backend")==b.get("backend"):
            if a["logits"]!=b["logits"]:
                raise ValueError("same-model A/A output changed")
        else:
            for la, lb in zip(a["logits"], b["logits"], strict=True):
                va = [math.exp(x - max(la[0])) for x in la[0]]
                vb = [math.exp(x - max(lb[0])) for x in lb[0]]
                sa, sb = sum(va), sum(vb)
                diff = max(abs(x / sa - y / sb) for x, y in zip(va, vb, strict=True))
                if diff > 0.0001:
                    raise ValueError(f"cross-backend probability parity failed: {diff}")


def plan(root, manifest, output, binary=None):
    script=str(Path(__file__).resolve())
    assets=dict(worker=script,manifest=str(manifest))
    data=json.loads(manifest.read_text())
    for i,item in enumerate(data["items"]):assets[f"image_{i}"]=item["path"]
    for key in MODELS:
        assets["graph_"+key.replace("-","_")]=str(root/"models"/key/"native-fp32.onnx")
        assets["config_"+key.replace("-","_")]=str(root/"models"/key/"config.json")
    def arm(key):
        # zenbench canonicalizes program symlinks. Keep the venv Python path as
        # an env argument so canonicalization cannot turn it into system Python.
        value=dict(id=key,program="/usr/bin/env",args=["/usr/bin/time","-v","-o","${output}/memory.txt",sys.executable,"${worker}","worker","--root",str(root),
            "--model",key,"--manifest","${manifest}","--output","${output}",
            "--stage","${stage}","--count","16","--warmup","4","--threads","1"],
            env=dict(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",RAYON_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",HF_HUB_OFFLINE="1"),
            revision=MODELS[key][1],timeout_seconds=180)
        if binary:
            cfg=json.loads((root/"models"/key/"config.json").read_text())
            value['args']=["/usr/bin/time","-v","-o","${output}/memory.txt",str(binary),"--bench",
                str(root/"models"/key/"native-fp32.onnx"),str(root/"models"/key/"witness-inputs.f32"),
                str(cfg["pretrained_cfg"]["input_size"][-1]),"${output}",key,digest(manifest)]
        return value
    value=dict(schema=1,baseline=arm("swift-xs"),candidates=[arm(x) for x in MODELS if x!="swift-xs"],
        cases=[dict(id=stage,inputs=assets,fields=dict(stage=stage)) for stage in
               ("inference","preprocess-inference","decode-preprocess-inference")],
        metric=dict(extract=dict(kind="prefix",prefix="elapsed_ms="),unit="ms",expected=16,divide_by=1),
        check=dict(kind="command",program="/usr/bin/env",args=[sys.executable,script,"check","--before","${baseline}","--after","${candidate}"],timeout_seconds=30),
        warmup="Four in-process forwards, then 16 calls cycling eight hashed photographs. FP32 CPU, one core. Session construction and disk I/O excluded. Separate boundaries: inference; decoded-RGB preprocess+inference; in-memory JPEG decode+preprocess+inference.",
        blocks=4,aa=True,deadline_seconds=1800)
    if binary:
        assets['rust_worker']=str(binary)
        for key in MODELS:assets['tensor_'+key.replace('-','_')]=str(root/"models"/key/"witness-inputs.f32")
        value['cases']=value['cases'][:1]
    save(output,value)


def witnesses(root,manifest):
    runtime(1)
    _,images=source_images(manifest)
    records=[]
    for key in MODELS:
        cfg=json.loads((root/"models"/key/"config.json").read_text())
        x=preprocessing(cfg)(images[0]).unsqueeze(0).numpy().astype("<f4")
        path=root/"models"/key/"witness-input.f32"
        if path.exists():
            if path.read_bytes()!=x.tobytes():raise ValueError(f"witness changed: {path}")
        else:
            with path.open("xb") as f:f.write(x.tobytes())
        batch_path=root/"models"/key/"witness-inputs.f32"
        batch=b"".join(preprocessing(cfg)(im).unsqueeze(0).numpy().astype("<f4").tobytes() for im in images)
        if batch_path.exists():
            if batch_path.read_bytes()!=batch:raise ValueError(f"witness batch changed: {batch_path}")
        else:
            with batch_path.open("xb") as f:f.write(batch)
        records.append(dict(model=key,path=str(path),sha256=digest(path),shape=list(x.shape),
                            manifest_sha256=digest(manifest)))
    identity=root/"witnesses.json"
    if identity.exists():
        if json.loads(identity.read_text())!=records:raise ValueError("witness manifest changed")
    else:save(identity,records)


def run_pair(root,plan_path,output,zenbench):
    # This host's user manager does not enable cpuset. A transient system scope
    # provides a real CPU boundary and a separate hard cap; inference keeps the
    # calling user's identity and the shared zenbench admission directory.
    user=pwd.getpwuid(os.getuid()).pw_name
    command=["sudo","-n","systemd-run","--scope","--quiet","-p","AllowedCPUs=0",
        "-p","MemoryMax=8G","-p","MemorySwapMax=0","runuser","-u",user,"--",
        "env","TMPDIR="+str(Path.home()/"tmp"/"zensally-moderation"),
        "ZENBENCH_COORDINATION_DIR="+str(Path.home()/".cache"/"zenbench"/"admission-v1"),
        "nice","-n19","ionice","-c3",str(zenbench),"pair",str(plan_path),"--output",str(output)]
    child=subprocess.Popen(command)
    progress=root/(output.name+"-progress.json")
    progress.write_text(json.dumps(dict(stage="running",pid=child.pid,supervisor_pid=os.getpid(),
        plan=str(plan_path),plan_sha256=digest(plan_path),output=str(output),command=command,
        zenbench_sha256=digest(zenbench),script_sha256=digest(__file__),
        checked_at=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
    rc=child.wait()
    record=json.loads(progress.read_text())
    record.update(stage="finished" if rc==0 else "failed",returncode=rc,
                  checked_at=datetime.now(timezone.utc).isoformat())
    progress.write_text(json.dumps(record,indent=2)+"\n")
    if rc:raise RuntimeError(f"zenbench returned {rc}: {output}")


def tract_probes(root,output,binary):
    import numpy as np
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for key in MODELS:
        directory=root/"models"/key
        cfg=json.loads((directory/"config.json").read_text())
        result=output/(key+".f32")
        command=["/usr/bin/time","-v","-o",str(output/(key+".time")),str(binary),
            str(directory/"native-fp32.onnx"),str(directory/"witness-input.f32"),
            str(cfg["pretrained_cfg"]["input_size"][-1]),str(result)]
        with (output/(key+".log")).open("x") as f:
            child=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=False)
        record=dict(model=key,returncode=child.returncode,command=command,binary_sha256=digest(binary))
        if child.returncode==0:
            logits=np.fromfile(result,dtype="<f4")
            probs=np.exp(logits-logits.max());probs/=probs.sum()
            reference_path=(root/"probe-freepik-v2.json") if key=="freepik" else root/"probes-v1"/(key+".json")
            reference=json.loads(reference_path.read_text())["native_scores"][0]
            error=float(np.max(np.abs(probs-reference)))
            record.update(probabilities=probs.tolist(),native_max_abs_error=error,parity_pass=(error<=0.0001))
        records.append(record)
        save(output/(key+".json"),record)
        print(json.dumps(record),flush=True)
    save(output/"summary.json",records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["fetch","probe","probes","worker","check","plan","witnesses","run","tract-probes"])
    p.add_argument("--root",type=Path)
    p.add_argument("--model",choices=MODELS)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--count",type=int,default=24)
    p.add_argument("--warmup",type=int,default=8)
    p.add_argument("--stage",choices=["inference","preprocess-inference","decode-preprocess-inference"],default="inference")
    p.add_argument("--before",type=Path)
    p.add_argument("--after",type=Path)
    p.add_argument("--plan",type=Path)
    p.add_argument("--zenbench",type=Path)
    p.add_argument("--binary",type=Path)
    a=p.parse_args()
    if a.command=="fetch":fetch(a.root)
    elif a.command=="probe":probe(a.root,a.model,a.manifest,a.output,a.threads)
    elif a.command=="probes":probes(a.root,a.manifest,a.output,a.threads)
    elif a.command=="worker":worker(a.root,a.model,a.manifest,a.output,a.threads,a.count,a.warmup,a.stage)
    elif a.command=="plan":plan(a.root,a.manifest,a.output,a.binary)
    elif a.command=="witnesses":witnesses(a.root,a.manifest)
    elif a.command=="run":run_pair(a.root,a.plan,a.output,a.zenbench)
    elif a.command=="tract-probes":tract_probes(a.root,a.output,a.binary)
    else:check(a.before,a.after)


if __name__=="__main__":
    main()
