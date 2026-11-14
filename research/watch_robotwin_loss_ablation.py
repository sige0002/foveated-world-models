"""Monitor the authorized four short runs and save paired reports automatically.

Does not start, stop, resume, or modify training jobs. Reads only their output.
"""
import argparse
import datetime
import json
from pathlib import Path
import time

from summarize_robotwin_loss_ablation import ARMS, read_json, read_jsonl, summarize


def save(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.runs_root.resolve()
    runs = {a: root/a for a in ARMS}
    jobs = {j['arm']: j for j in read_json(root/'training_launch.json')['jobs']}
    completed_reports = set()
    while True:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        warnings = []
        streams = {a: read_jsonl(p/'training_stream.jsonl', warnings) for a,p in runs.items()}
        common_step = min((s[-1]['step'] if s else 0) for s in streams.values())
        paired_stream = None
        if common_step:
            same = [next(x for x in s if x['step'] == common_step) for s in streams.values()]
            paired_stream = all(len({x['cumulative_'+k+'_sha256'] for x in same}) == 1
                                for k in ('data', 'aug', 'epsilon'))
        status = {}
        for a,p in runs.items():
            progress = read_json(p/'progress.json')
            summary = read_json(p/'run_summary.json')
            proc = Path(f'/proc/{jobs[a]["pid"]}')
            alive = False
            try:
                cmdline = (proc/'cmdline').read_bytes().decode().split('\0')
                stat = (proc/'stat').read_text()
                alive = str(p) in cmdline and stat[stat.rfind(')')+2:].split()[0] != 'Z'
            except FileNotFoundError:
                pass
            status[a] = {'pid': jobs[a]['pid'], 'alive': alive,
                         'last_step': progress['step'] if progress else None,
                         'seconds_per_update': progress['recent_seconds_per_update'] if progress else None,
                         'complete': bool(summary and summary.get('phase') == 'complete' and summary.get('step') == 3000)}
        done = all(s['complete'] for s in status.values())
        failed = [a for a,s in status.items() if not s['complete'] and not s['alive']]
        memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        report = summarize(runs, draws=0)
        step = report['selected_step']
        save(root/'live_comparison.json', report)
        if step is not None and step not in completed_reports:
            save(root/f'comparison_step{step:06d}.json', report)
            completed_reports.add(step)
            print(json.dumps({'at':now,'event':'comparison','step':step,'status':report['status']}),flush=True)
        state = {'at':now, 'arms':status, 'matched_stream_step':common_step,
                 'recorded_streams_match':paired_stream, 'available_memory_bytes':int(memory['MemAvailable'].split()[0])*1024,
                 'latest_common_validation':step, 'failed_arms':failed, 'warnings':warnings}
        save(root/'watch_status.json', state)
        if paired_stream is False:
            save(root/'pairing_alert.json', state)
        if failed:
            print(json.dumps({'at':now,'event':'failure','arms':failed}),flush=True)
            return
        if done:
            final = summarize(runs, step=3000, draws=4000)
            save(root/'final_comparison.json', final)
            print(json.dumps({'at':now,'event':'all_complete','status':final['status']}),flush=True)
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
