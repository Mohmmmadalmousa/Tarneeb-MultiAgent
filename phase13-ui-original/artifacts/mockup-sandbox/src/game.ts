export type Summary = {
  deal_index: number; winning_bid: number | null; trump: string | null;
  tricks_won_by_team: number[]; score_change_by_team: number[];
  contract_made: boolean | null; passed_out: boolean;
};
export type Game = {
  revision: number;
  table: null | {
    phase: string; own_hand: string[]; seat_on_turn: number | null;
    dealer_seat: number; deal_index: number; current_high_bid: number | null;
    high_bidder_seat: number | null; seats_that_passed: number[];
    contract: number | null; trump: string | null; trick_number: number;
    current_trick: [number, string][]; tricks_won_by_team: number[];
    match_score_by_team: number[];
  };
  legal_actions: {action_id: number; label: string}[];
  hand_counts?: number[]; history: Summary[]; deal_complete?: boolean;
  match_complete: boolean;
  last_trick: null | {number: number; winner: number; cards: [number, string][]};
};
export async function requestGame(command?: string, revision?: number, action?: number): Promise<Game> {
  const response = await fetch(command ? '/api/command' : '/api/state', command ? {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({command, revision, action}),
  } : {cache: 'no-store'});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
