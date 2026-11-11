"""Prepare metadata only, or explicitly execute one preregistered Phase 12 job."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

from tarneeb.evaluation.phase12 import check_seed_blocks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/phase12_protocol_v1'
OLD = ROOT / 'results/phase11_a3_evaluation_v1'
COHORT = [f'phase9_gru_bc_s{i}' for i in range(3)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare():
    old = read(OLD / 'preregistration.json')
    selected = read(OLD / 'selection.json')
    exposure = read(OLD / 'exposure.json')
    pilot = read(ROOT/'results/balanced_seating_pilot_v1/plan.json')['evaluation']
    entries = exposure['entries'] + [('balanced pilot', 'arena', pilot['seed'], pilot['pairs_per_opponent'])]
    check_seed_blocks(entries, [(78000, 300), (80000, 1000), (82000, 1000), (85000, 100)])
    models = old['models']
    # Collapse byte-identical aliases before constructing all registered archive pairs.
    canonical = {}
    aliases = {}
    for name, model in models.items():
        canonical.setdefault(model['sha256'], name)
        aliases[name] = canonical[model['sha256']]
    completed = {}
    for key in selected['evidence']:
        a, b = key.split('__')
        if a in models and b in models:
            pair = tuple(sorted((aliases[a], aliases[b])))
            completed.setdefault(pair, []).append(str((OLD/'selection'/f'{key}.json').relative_to(ROOT)))
    jobs = {}
    for a, b in combinations(sorted(set(aliases.values())), 2):
        if (a, b) not in completed:
            jobs[f'e5__{a}__{b}'] = dict(suite='e5', a=[a, a], b=[b, b], pairs=300, seed=78000)
    for a, b in combinations(COHORT, 2):
        # E6 covers these historical edges at a larger sample size; do not duplicate.
        jobs.pop(f'e5__{a}__{b}', None)
        jobs[f'e6__{a}__{b}'] = dict(suite='e6', a=[a, a], b=[b, b], pairs=1000, seed=80000)
    for a in COHORT:
        jobs[f'e7__{a}__{a}'] = dict(suite='e7', a=[a, a], b=['strong', 'strong'], pairs=1000, seed=82000)
    for a, b in combinations(COHORT, 2):
        jobs[f'e7__{a}__{b}'] = dict(suite='e7', a=[a, b], b=['strong', 'strong'], pairs=1000, seed=82000)
    sources = [OLD/'preregistration.json', OLD/'selection.json', OLD/'exposure.json',
               ROOT/'results/balanced_seating_pilot_v1/plan.json',
               ROOT/'scripts/phase12_evaluate.py', ROOT/'tarneeb/evaluation/phase12.py',
               ROOT/'scripts/phase12_human.py', ROOT/'docs/PHASE12_PROTOCOL_V1.md']
    sources += [ROOT/name for name in (
        'tarneeb/evaluation/arena.py', 'tarneeb/evaluation/metrics.py',
        'tarneeb/evaluation/stats.py', 'tarneeb/agents/recurrent_agent.py',
        'tarneeb/app/cli.py', 'tarneeb/app/session.py')]
    # Inventory out-of-scope checkpoint paths, without loading or hashing large files.
    registered = {str(Path(m['path'])).lower() for m in models.values()}
    outside = [p.relative_to(ROOT).as_posix() for p in (ROOT/'checkpoints').rglob('*.pt')
               if str(p.relative_to(ROOT)).lower() not in registered]
    plan = dict(models=models, aliases=aliases, jobs=jobs,
                reused=[dict(policies=list(pair), paths=paths) for pair, paths in completed.items()],
                historical_cycles=selected['cross_play'], outside_registered_archive=outside,
                sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
                bootstrap_resamples=2000, human_seed_start=1000085000,
                human_seed_count=100, human_scored_deal_target=30)
    OUT.mkdir(exist_ok=False)
    write_new(OUT/'plan.json', plan)
    print({suite: sum(j['suite'] == suite for j in jobs.values()) for suite in ('e5', 'e6', 'e7')})


def verified_plan():
    plan = read(OUT/'plan.json')
    for name, digest in plan['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Plan source changed: {name}; do not rewrite the plan')
    return plan


def agent(name, plan):
    from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
    from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
    if name == 'strong':
        return StrongHeuristicAgent(name='strong')
    model = plan['models'][name]
    if sha(ROOT/model['path']) != model['sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {name}')
    result = RecurrentNeuralAgent.from_checkpoint(ROOT/model['path'], name=name, deterministic=True)
    result.actor.eval().requires_grad_(False)
    return result


def execute(job_id):
    import torch
    from tarneeb.evaluation.phase12 import Partnership, RecordingArena, calibration
    plan = verified_plan()
    job = plan['jobs'][job_id]
    directory = OUT/job_id
    directory.mkdir(exist_ok=False)  # interrupted jobs remain visible; never overwrite
    write_new(directory/'started.json', dict(job=job, plan_sha256=sha(OUT/'plan.json')))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    with (directory/'deals.jsonl').open('x', encoding='utf-8') as stream:
        def sink(batch):
            rows.extend(batch)
            for row in batch:
                stream.write(json.dumps(row)+'\n')
            stream.flush()
        arena = RecordingArena(sink)
        result = arena.evaluate(
            Partnership(*(agent(n, plan) for n in job['a'])),
            Partnership(*(agent(n, plan) for n in job['b'])),
            job['pairs'], job['seed'], bootstrap_resamples=plan['bootstrap_resamples'],
            replay_sample_rate=1.0,
        )
    if result.metrics['illegal_action_rate'] != 0 or result.metrics['replay_integrity_rate'] != 1:
        raise ValueError('Safety or scored-deal replay failure')
    write_new(directory/'result.json', dict(job=job, plan_sha256=sha(OUT/'plan.json'),
              metrics=result.metrics, calibration=calibration(rows),
              scores=[p.paired_score_difference for p in result.paired_outcomes],
              identities=[[p.match_seed, p.deal_index] for p in result.paired_outcomes]))


def gaps():
    from tarneeb.evaluation.phase12 import sp_xp_gap
    plan = verified_plan()
    output = {}
    for a, b in combinations(COHORT, 2):
        records = [read(OUT/f'e7__{x}__{y}'/'result.json') for x, y in [(a,a),(b,b),(a,b)]]
        if any(r['plan_sha256'] != sha(OUT/'plan.json') for r in records):
            raise ValueError('Result plan mismatch')
        if not records[0]['identities'] == records[1]['identities'] == records[2]['identities']:
            raise ValueError('Unmatched SP/XP deals')
        output[f'{a}__{b}'] = sp_xp_gap(*(r['scores'] for r in records), seed=82000,
                                       resamples=plan['bootstrap_resamples'])
    write_new(OUT/'sp_xp_gaps.json', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--job')
    group.add_argument('--gaps', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.job:
        execute(args.job)
    else:
        gaps()
