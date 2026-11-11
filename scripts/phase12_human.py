"""Authentic E8 collection using the frozen local CLI; never accepts auto-play."""
import argparse
import json
import sys
from dataclasses import asdict

from scripts.phase12_evaluate import COHORT, OUT, agent, sha, verified_plan, write_new
from tarneeb.app.cli import play_match
from tarneeb.app.session import MatchSession


class RecordedHumanSession(MatchSession):
    def __init__(self, directory, participant, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.participant = participant
        self.human_decisions = 0

    def apply_human_action(self, action_id):
        super().apply_human_action(action_id)
        self.human_decisions += 1

    def _finish_passed_out_deal(self):
        # Frozen CLI predates recurrent play: its automatic redeal does not reset
        # agents. The additive wrapper must zero GRU state at that boundary too.
        super()._finish_passed_out_deal()
        for policy in self.agents_by_seat.values():
            policy.reset()

    def finish_deal(self):
        summary = super().finish_deal()
        record = self.deal_records[-1]
        if not self.replay_engine.verify(record):
            raise ValueError('Human replay failed')
        stem = f'{self.match_seed}_{summary.deal_index}'
        with (self.directory/f'{stem}.replay.json').open('x', encoding='utf-8') as stream:
            stream.write(record.to_json())
        policies = [p.policy_id for p in self.seat_policies()]
        declarer = summary.declarer_seat
        rows = [dict(asdict(summary), match_seed=self.match_seed, seat=seat,
                     policy=policy,
                     role='declarer' if seat == declarer else
                          'partner' if seat % 2 == declarer % 2 else 'defender',
                     team_tricks=summary.tricks_won_by_team[seat % 2])
                for seat, policy in enumerate(policies)]
        write_new(self.directory/f'{stem}.json', dict(
            summary=asdict(summary), rows=rows, participant=self.participant,
            human_decisions_cumulative=self.human_decisions, replay_verified=True,
            match_seed=self.match_seed, policies=policies))
        return summary


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', required=True, help='Pseudonym, not a real name')
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise ValueError('An interactive human terminal is required; no redirected input')
    plan = verified_plan()
    if input('I will personally choose every human action (type HUMAN): ') != 'HUMAN':
        raise ValueError('Human attestation required')
    directory = OUT/'e8_human'
    directory.mkdir(exist_ok=False)
    write_new(directory/'session.json', dict(participant=args.participant,
              attestation='HUMAN', plan_sha256=sha(OUT/'plan.json'), target=30,
              seats=['human', 'strong', COHORT[0], 'strong']))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    total = 0
    for index in range(plan['human_seed_count']):
        session = RecordedHumanSession(directory, args.participant,
            agents_by_seat={1: agent('strong', plan), 2: agent(COHORT[0], plan),
                            3: agent('strong', plan)}, human_seat=0,
            match_seed=plan['human_seed_start']+index, dealer_seat=index % 4)
        try:
            play_match(session, 30-total, print, input)
        finally:
            # Save passed-out attempts too; scored deals have separate replay files.
            write_new(directory/f'match_{index}.json', dict(
                match_seed=session.match_seed,
                attempts=[asdict(s) for s in session.completed_deals],
                human_decisions=session.human_decisions))
        total += sum(not s.passed_out for s in session.completed_deals)
        if total >= 30:
            write_new(directory/'complete.json', dict(authenticity='operator attested, not independently proven',
                      scored_deals=total, participant=args.participant))
            return
    raise ValueError('Seed budget exhausted; new reviewed allocation required')


if __name__ == '__main__':
    main()
