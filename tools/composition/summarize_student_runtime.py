#!/usr/bin/env python3
"""Re-read retained timing/operator/trace evidence without running inference."""
import argparse
import gzip
import json
from pathlib import Path
import sqlite3
import statistics

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    runtime = args.base/'unsplash-student-runtime'
    timings = {}
    for label, path in [('original', runtime/'2026-11-08-eval/timing'),
                        ('gather', runtime/'2026-11-09-gather-static/timing'),
                        ('channels', runtime/'2026-11-09-channel-gather/timing'),
                        ('validation_graph', args.base/'graphed-rank-query/2026-11-08/timing')]:
        packet = path/'packet.json.gz'
        data = json.loads(gzip.decompress(packet.read_bytes()))
        if not data['complete'] or any(v['status'] != 'completed' for v in data['invocations']):
            raise ValueError('completed timing packet required')
        arms = [data['plan']['baseline']['id']]+[a['id'] for a in data['plan']['candidates']]+['__aa']
        timings[label] = dict(packet_sha256=sha(packet), placement=data['placement'],
            summaries=data['summaries'], invocations=len(data['invocations']),
            means_ms={case['id']:{arm:statistics.mean(v['values_ns'][0]/1e6
                for v in data['invocations'] if v['case'] == index and v['arm'] == a)
                for a,arm in enumerate(arms)} for index,case in enumerate(data['plan']['cases'])})
    profiles = {}
    for label, dirname in [('original', '2026-11-09-node-profile'), ('gather', '2026-11-09-gather-node-profile'),
                           ('channels', '2026-11-09-channel-node-profile')]:
        path = runtime/dirname/'summary.json'
        data = json.loads(path.read_text())
        profiles[label] = dict(summary_sha256=sha(path), requests=data['requests'],
            query_duration_us=sum(v['duration_us'] for v in data['reports']['query']['operations'].values()),
            query_operations=data['reports']['query']['operations'])
    training = {}
    for label, dirname in [('eager', '2026-11-08-scoped'), ('graph', '2026-11-09-graph')]:
        root = args.base/'unsplash-rank-training-profile'/dirname
        report = json.loads((root/'PROFILE_COMPLETION.json').read_text())
        scope = json.loads((root/'TARGET_SCOPE.json').read_text())
        if not report['completed'] or report['captured_optimizer_steps'] != 200 or scope['memory_max_bytes'] != 4294967296:
            raise ValueError('equal completed bounded training captures required')
        connection = sqlite3.connect('file:'+str(root/'trace.sqlite')+'?mode=ro', uri=True)
        kernels = connection.execute('SELECT count(*),sum(end-start),max(end)-min(start) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
        ranges = connection.execute('SELECT coalesce(n.text,s.value),count(*),sum(n.end-n.start) '
            'FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end>n.start '
            'GROUP BY coalesce(n.text,s.value)').fetchall()
        connection.close()
        training[label] = dict(trace_sha256=sha(root/'trace.nsys-rep'), kernels=kernels[0],
            kernel_total_ns=kernels[1], kernel_span_ns=kernels[2],
            nvtx={name:dict(calls=count, host_duration_ns=duration) for name,count,duration in ranges},
            cuda_peak_allocated=report['cuda_peak_allocated'], cuda_peak_reserved=report['cuda_peak_reserved'],
            full_training_speed_measured=False)
    result = dict(timings=timings, operator_profiles=profiles, training_profiles=training,
        profile_kernel_counts_do_not_include_memcpy_memset_events=True,
        profile_kernel_active_fraction_is_not_sm_occupancy=True, model_promoted=False)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(timing_packets=len(timings), operator_profiles=len(profiles), training_profiles=len(training))))


if __name__ == '__main__':
    main()
