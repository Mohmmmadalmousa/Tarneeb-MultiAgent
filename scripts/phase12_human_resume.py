"""Resume authentic Phase 12 E8 collection after an interrupted, saved match.

Additive continuation: uses the next preregistered match seed, never resumes a
partially played match or modifies historical results. No automated human moves.
"""
import argparse
import json
import re
import sys
from dataclasses import asdict

from scripts.phase12_evaluate import COHORT, OUT, agent, sha, verified_plan, write_new
from scripts.phase12_human import RecordedHumanSession
from tarneeb.app.cli import play_match


def check_saved(directory, participant, plan):
    if not directory.is_dir():
        raise ValueError('Existing E8 session directory not found; do not start a new one')
    metadata = json.loads((directory / 'session.json').read_text(encoding='utf-8-sig'))
    if (metadata.get('participant') != participant
            or metadata.get('attestation') != 'HUMAN'
            or metadata.get('target') != 30
            or metadata.get('plan_sha256') != sha(OUT / 'plan.json')
            or metadata.get('seats') != ['human', 'strong', COHORT[0], 'strong']):
        raise ValueError('Session identity/plan mismatch: preserve data and stop')

    matches = {}
    for path in directory.glob('match_*.json'):
        match = re.fullmatch(r'match_(\d+)\.json', path.name)
        if match is None or int(match.group(1)) in matches:
            raise ValueError(f'Unexpected or duplicate match summary: {path.name}')
        matches[int(match.group(1))] = path
    if sorted(matches) != list(range(len(matches))):
        raise ValueError('Saved match-index gap; manual review required')
    if not matches:
        raise ValueError('No saved match summaries; manual review required')

    recorded = set()
    scored = 0
    for index in sorted(matches):
        record = json.loads(matches[index].read_text(encoding='utf-8-sig'))
        seed = plan['human_seed_start'] + index
        if record.get('match_seed') != seed or not isinstance(record.get('attempts'), list):
            raise ValueError(f'Match {index} has unexpected seed or attempts')
        for attempt in record['attempts']:
            if attempt['passed_out']:
                continue
            stem = f"{seed}_{attempt['deal_index']}"
            if stem in recorded:
                raise ValueError(f'Duplicate deal: {stem}')
            row_path = directory / f'{stem}.json'
            replay_path = directory / f'{stem}.replay.json'
            if not row_path.is_file() or not replay_path.is_file():
                raise ValueError(f'Missing deal or replay: {stem}')
            row = json.loads(row_path.read_text(encoding='utf-8-sig'))
            if (row.get('match_seed') != seed
                    or row.get('participant') != participant
                    or row.get('replay_verified') is not True):
                raise ValueError(f'Invalid saved deal identity/replay flag: {stem}')
            recorded.add(stem)
            scored += 1

    all_rows = {
        p.stem for p in directory.iterdir()
        if p.is_file() and re.fullmatch(r'\d+_\d+\.json', p.name)
    }
    if all_rows != recorded:
        raise ValueError('Saved deal records differ from match summaries; stop and review')
    if scored > 30:
        raise ValueError('Already beyond the 30-deal target; manual review required')
    return scored, len(matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', required=True, help='Original session pseudonym')
    parser.add_argument('--check', action='store_true', help='Read-only preflight: no games')
    args = parser.parse_args()
    plan = verified_plan()
    directory = OUT / 'e8_human'
    total, next_index = check_saved(directory, args.participant, plan)
    print(f'Verified {total}/30 scored deals; next match index: {next_index}')
    if (directory / 'complete.json').exists() or total >= 30:
        print('E8 already has its target; do not start more games. Review finalization.')
        return
    if next_index >= plan['human_seed_count']:
        raise ValueError('Preregistered E8 seed budget exhausted')
    print(f'Next unused match seed: {plan["human_seed_start"] + next_index}')
    if args.check:
        return

    if not sys.stdin.isatty():
        raise ValueError('Interactive human terminal required; no redirected input')
    if input('I will personally choose every human action (type HUMAN): ') != 'HUMAN':
        raise ValueError('Human attestation required')
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    for index in range(next_index, plan['human_seed_count']):
        session = RecordedHumanSession(
            directory, args.participant,
            agents_by_seat={1: agent('strong', plan), 2: agent(COHORT[0], plan),
                            3: agent('strong', plan)},
            human_seat=0,
            match_seed=plan['human_seed_start'] + index,
            dealer_seat=index % 4,
        )
        try:
            play_match(session, 30 - total, print, input)
        finally:
            write_new(directory / f'match_{index}.json', dict(
                match_seed=session.match_seed,
                attempts=[asdict(s) for s in session.completed_deals],
                human_decisions=session.human_decisions,
            ))
        total += sum(not s.passed_out for s in session.completed_deals)
        if total >= 30:
            write_new(directory / 'complete.json', dict(
                authenticity='operator attested, not independently proven',
                scored_deals=total,
                participant=args.participant,
            ))
            print(f'E8 complete: {total} scored deals preserved.')
            return
    raise ValueError('Seed budget exhausted; new reviewed allocation required')


if __name__ == '__main__':
    main()
