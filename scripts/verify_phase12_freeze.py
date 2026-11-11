"""Verify saved Phase 12 evidence only; no policy inference or new gameplay."""
import json
import math
import re
from collections import Counter
from dataclasses import fields
from itertools import combinations
from pathlib import Path

from scripts.phase12_evaluate import COHORT, OUT, ROOT, read, sha, verified_plan, write_new
from scripts.phase12_human_resume import check_saved
from tarneeb.core.recorder import DealRecord, ReplayEngine
from tarneeb.core.rules import RulesConfig
from tarneeb.evaluation.arena import evaluation_match_seed
from tarneeb.evaluation.metrics import DealOutcome, PairedOutcome, aggregate_metrics
from tarneeb.evaluation.phase12 import calibration, check_seed_blocks, deal_rows, sp_xp_gap


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equivalent(actual, expected, where='value'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f'{where}: keys')
        for key in expected:
            equivalent(actual[key], expected[key], f'{where}.{key}')
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f'{where}: length')
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            equivalent(left, right, f'{where}[{i}]')
    elif isinstance(expected, float):
        require(math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10),
                f'{where}: {actual} != {expected}')
    else:
        require(actual == expected, f'{where}: {actual} != {expected}')


def json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main():
    destination = ROOT/'results/phase12_freeze_v1'
    destination.mkdir(exist_ok=True)
    require(not (destination/'verification.json').exists(), 'Verification receipt already exists')
    plan = verified_plan()
    plan_hash = sha(OUT/'plan.json')
    checkpoint_hashes = {}
    for name in COHORT:
        model = plan['models'][name]
        require(sha(ROOT/model['path']) == model['sha256'], f'Checkpoint changed: {name}')
        checkpoint_hashes[model['path']] = model['sha256']
    exposure = read(ROOT/'results/phase11_a3_evaluation_v1/exposure.json')['entries']
    pilot = read(ROOT/'results/balanced_seating_pilot_v1/plan.json')['evaluation']
    check_seed_blocks(exposure + [('pilot', 'arena', pilot['seed'], pilot['pairs_per_opponent'])],
                      [(78000, 300), (80000, 1000), (82000, 1000), (85000, 100)])
    jobs = {key: value for key, value in plan['jobs'].items() if value['suite'] in ('e6', 'e7')}
    require(Counter(j['suite'] for j in jobs.values()) == {'e6': 3, 'e7': 6}, 'Job coverage')
    require(not list(OUT.glob('e5__*')), 'Unexpected E5 execution')
    reports, saved = {}, {}
    outcome_fields = [field.name for field in fields(DealOutcome)]
    for job_id, job in jobs.items():
        directory = OUT/job_id
        result = read(directory/'result.json')
        equivalent(read(directory/'started.json'), dict(job=job, plan_sha256=plan_hash), job_id)
        equivalent(result['job'], job, job_id)
        require(result['plan_sha256'] == plan_hash and job['pairs'] == 1000, job_id)
        identities = [[evaluation_match_seed(job['seed'], i), 0] for i in range(1000)]
        equivalent(result['identities'], identities, job_id)
        rows = [json.loads(line) for line in (directory/'deals.jsonl').read_text().splitlines()]
        require(len(rows) == 8000, f'{job_id}: row count')
        outcomes = []
        for offset in range(0, len(rows), 4):
            batch = rows[offset:offset+4]
            data = {name: batch[0][name] for name in outcome_fields}
            for key in ('score_change_by_team', 'tricks_won_by_team'):
                data[key] = tuple(data[key])
            outcome = DealOutcome(**data)
            pair_index = offset//8
            team = (offset//4) % 2
            require((outcome.match_seed, outcome.deal_index, outcome.dealer_seat, outcome.policy_a_team)
                    == (identities[pair_index][0], 0, pair_index % 4, team), f'{job_id}: seating')
            policies = [job['a'][seat//2] if seat % 2 == team else job['b'][seat//2]
                        for seat in range(4)]
            equivalent(batch, json_value(deal_rows(outcome, policies)), f'{job_id}: seat rows')
            require(outcome.passed_out or outcome.replay_verified is True, f'{job_id}: replay flag')
            if not outcome.passed_out:
                require(sum(outcome.tricks_won_by_team) == 13, f'{job_id}: tricks')
                require(outcome.contract_made == (outcome.tricks_won_by_team[outcome.declarer_seat % 2]
                                                   >= outcome.winning_bid), f'{job_id}: contract')
            outcomes.append(outcome)
        pairs = [PairedOutcome(identities[i][0], 0, outcomes[2*i], outcomes[2*i+1]) for i in range(1000)]
        equivalent(result['scores'], [p.paired_score_difference for p in pairs], job_id)
        metrics = aggregate_metrics(pairs, job['seed'], plan['bootstrap_resamples'],
                                    runtime_seconds=result['metrics']['runtime_seconds'])
        equivalent(result['metrics'], json_value(metrics), f'{job_id}: metrics')
        equivalent(result['calibration'], json_value(calibration(rows)), f'{job_id}: calibration')
        reports[job_id] = dict(metrics=metrics, calibration=result['calibration'])
        saved[job_id] = result
    gaps = read(OUT/'sp_xp_gaps.json')
    require(len(gaps) == 3, 'Gap count')
    for a, b in combinations(COHORT, 2):
        results = [saved[f'e7__{x}__{y}'] for x, y in [(a,a), (b,b), (a,b)]]
        require(results[0]['identities'] == results[1]['identities'] == results[2]['identities'], 'Gap pairing')
        equivalent(gaps[f'{a}__{b}'], sp_xp_gap(*(r['scores'] for r in results),
                   seed=82000, resamples=plan['bootstrap_resamples']), 'gap')

    human = OUT/'e8_human'
    count, match_count = check_saved(human, 'P01', plan)
    require(count == 30, 'Human count')
    equivalent(read(human/'complete.json'), dict(authenticity='operator attested, not independently proven',
                                               scored_deals=30, participant='P01'), 'completion')
    replay = ReplayEngine(RulesConfig())
    identities, summaries, match_reports = set(), [], []
    for index in range(match_count):
        match = read(human/f'match_{index}.json')
        score = [0, 0]
        scored_count = 0
        for attempt in match['attempts']:
            if attempt['passed_out']:
                continue
            key = (match['match_seed'], attempt['deal_index'])
            require(key not in identities, 'Duplicate human identity')
            identities.add(key)
            stem = f'{key[0]}_{key[1]}'
            row = read(human/f'{stem}.json')
            equivalent(row['summary'], attempt, stem)
            require(row['human_decisions_cumulative'] > 0, f'{stem}: human actions')
            record = DealRecord.from_json((human/f'{stem}.replay.json').read_text())
            require((record.match_seed, record.deal_index, record.dealer_seat)
                    == (*key, attempt['dealer_seat']), f'{stem}: replay identity')
            fresh = replay.engine.new_deal(record.dealer_seat, record.deal_index, record.match_seed)
            replay.engine.deal(fresh)
            require(tuple(fresh.original_hands) == record.original_hands, f'{stem}: seeded hands')
            state = replay.replay(record)  # replay recorded actions, never query a policy
            require(tuple(state.events) == record.events and state.deal_result == record.deal_result
                    and tuple(state.match_score) == record.final_match_score, f'{stem}: replay')
            policies = ['human', 'strong', COHORT[0], 'strong']
            require([p.policy_id for p in record.policies] == row['policies'] == policies, f'{stem}: policies')
            declared = state.declarer_seat
            result = state.deal_result
            delta = [0, 0]
            delta[declared % 2] = result.score_delta_bidding_team
            delta[1-declared % 2] = result.score_delta_defending_team
            for team in range(2):
                score[team] += delta[team]
            require(attempt['declarer_seat'] == declared and attempt['winning_bid'] == result.bid
                    and attempt['tricks_won_by_team'] == state.tricks_won
                    and attempt['score_change_by_team'] == delta
                    and attempt['match_score_by_team'] == score
                    and attempt['contract_made'] == (result.tricks_bidding_team >= result.bid), f'{stem}: summary')
            expected_rows = [dict(attempt, match_seed=key[0], seat=seat, policy=policy,
                                 role='declarer' if seat == declared else
                                      'partner' if seat % 2 == declared % 2 else 'defender',
                                 team_tricks=state.tricks_won[seat % 2]) for seat, policy in enumerate(policies)]
            equivalent(row['rows'], expected_rows, f'{stem}: roles')
            summaries.append(attempt)
            scored_count += 1
        match_reports.append(dict(index=index, seed=match['match_seed'], scored_deals=scored_count,
                                  score=score, target_reached=max(score) >= 31,
                                  human_decisions=match['human_decisions']))
    require(len(list(human.glob('*.replay.json'))) == len(identities) == 30, 'Human replay coverage')
    snapshots = {}
    for backup in sorted(ROOT.glob('phase12_e8*backup_*')):
        checked = 0
        for path in backup.rglob('*'):
            if path.is_file():
                current = human/path.relative_to(backup)
                require(current.is_file() and sha(current) == sha(path), f'Historical E8 snapshot differs: {path}')
                checked += 1
        snapshots[backup.name] = checked
    human_report = dict(scored_deals=30, replay_verified=30, participant='P01',
                        authenticity='operator attested, not independently proven',
                        matches=match_reports, preserved_snapshot_files=snapshots,
                        human_team_score=sum(s['score_change_by_team'][0] for s in summaries),
                        strong_team_score=sum(s['score_change_by_team'][1] for s in summaries),
                        declarer_counts=dict(Counter(s['declarer_seat'] for s in summaries)))
    artifacts = {}
    for folder in [OUT, ROOT/'results/phase12_evidence_v1', *sorted(ROOT.glob('phase12_e8*backup_*'))]:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    write_new(destination/'verification.json', dict(status='PASS', plan_sha256=plan_hash,
              checkpoint_sha256=checkpoint_hashes, verified_source_sha256=plan['sources'],
              jobs=reports, gaps=gaps, human=human_report, artifact_sha256=artifacts,
              limitations=['E6/E7 replay flags verified from saved rows; action traces were not persisted',
                           'Resume helper was not part of original plan source pins; frozen separately',
                           'Snapshot equality proves retained bytes, not absence of every possible past overwrite']))
    print(json.dumps(dict(status='PASS', jobs=len(jobs), played_deals=18000, human=human_report,
                          artifacts=len(artifacts)), indent=2))


if __name__ == '__main__':
    main()
