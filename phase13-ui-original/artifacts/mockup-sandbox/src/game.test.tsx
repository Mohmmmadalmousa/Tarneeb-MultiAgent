import { afterEach, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { TarneebAI } from './components/mockups/TarneebAI';
import { cardPress } from './cardPress';
import type { Game } from './game';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

test('selection changes and illegal/busy presses never submit', () => {
  expect(cardPress(null, 'A', true, false)).toEqual({selected: 'A', submit: false});
  expect(cardPress('A', 'B', true, false)).toEqual({selected: 'B', submit: false});
  expect(cardPress('B', 'B', true, false)).toEqual({selected: null, submit: true});
  expect(cardPress('A', 'A', false, false).submit).toBe(false);
  expect(cardPress('A', 'A', true, true).submit).toBe(false);
});

test('original card buttons submit on exactly second press and block duplicates', async () => {
  const game: Game = {
    revision: 8, table: {phase: 'PLAY', own_hand: ['AC', 'KC', '2H'], seat_on_turn: 0,
      dealer_seat: 2, deal_index: 0, current_high_bid: 7, high_bidder_seat: 0,
      seats_that_passed: [1, 2, 3], contract: 7, trump: 'clubs', trick_number: 0,
      current_trick: [], tricks_won_by_team: [0, 0], match_score_by_team: [0, 0]},
    legal_actions: [{action_id: 12, label: 'AC'}, {action_id: 11, label: 'KC'}],
    history: [], match_complete: false, last_trick: null,
  };
  const fetch = vi.fn().mockResolvedValueOnce({ok: true, json: async () => game})
    .mockImplementation(() => new Promise(() => {}));
  vi.stubGlobal('fetch', fetch);
  render(<TarneebAI />);
  const ace = await screen.findByRole('button', {name: /^A ♣/});
  const king = screen.getByRole('button', {name: /^K ♣/});
  const illegal = screen.getByRole('button', {name: /^2 ♥/});
  fireEvent.click(illegal);
  fireEvent.click(ace);
  expect(ace.getAttribute('aria-pressed')).toBe('true');
  expect(fetch).toHaveBeenCalledTimes(1);
  fireEvent.click(king);
  expect(fetch).toHaveBeenCalledTimes(1);
  fireEvent.click(king);
  fireEvent.click(king);
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({command: 'action', revision: 8, action: 11});
});
