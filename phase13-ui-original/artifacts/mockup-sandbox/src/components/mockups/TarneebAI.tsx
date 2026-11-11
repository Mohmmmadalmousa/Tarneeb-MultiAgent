import { useEffect, useRef, useState, type CSSProperties } from "react";

import { requestGame, type Game } from "../../game";
import { cardPress } from "../../cardPress";

type Language = "ar" | "en";
type Overlay = "menu" | "setup" | "settings" | "rules" | "scores" | "summary" | "results" | "trump" | null;
type Phase = "play" | "bidding" | "trump";
type Suit = "♠" | "♥" | "♦" | "♣";
type Card = { rank: string; suit: Suit };

const suits: Suit[] = ["♣", "♦", "♥", "♠"];
const suitNames = ["clubs", "diamonds", "hearts", "spades"];
const parseCard = (name: string): Card => ({rank: name.slice(0, -1), suit: suits["CDHS".indexOf(name.slice(-1))]});
const cardId = (card: Card) => suits.indexOf(card.suit) * 13 + ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"].indexOf(card.rank);

const text = {
  ar: {
    brand: "طرنيب", ai: "AI", score: "النتيجة", myTeam: "فريقك", rivals: "الخصوم",
    contract: "الطلب", tricks: "الأكلات", round: "الجولة", playerName: "دعبول", partner: "شريكك", left: "خصم ١",
    right: "خصم ٢", you: "إنت", dealer: "موزّع", turn: "دورك — اختار ورقة",
    hand: "ورقك", cards: "ورقة", play: "لعب", bid: "الطلب", trump: "الطرنيب",
    tapCard: "كبسة لاختيار الورقة، وكبسة ثانية عليها للعب", highest: "أعلى طلب", bidPrompt: "دورك للطلب", firstBid: "ابدأ من ٧", auctionStart: "الطلب يبدأ من ٧", pass: "باس",
    bidContinues: "باس — المزايدة مستمرة", finishAuction: "إنهاء الطلب", auctionDone: "انتهى الطلب", chooseTrumpTitle: "اختار الطرنيب", chooseTrumpDesc: "اختار نوع الطرنيب قبل ما يبدأ اللعب", playBegins: "هسا ببلّش اللعب بين الأربعة", firstTrick: "دورك — ابدأ الأكلة الأولى",
    chooseTrump: "اختار الطرنيب", menu: "القائمة", setup: "إعداد اللعبة", start: "ابدأ التوزيع",
    settings: "الإعدادات", rules: "كيف تلعب", close: "إغلاق", demo: "لعب محلي · ذكاء اصطناعي",
    classic: "طرنيب كلاسيك", target: "الهدف", points: "نقطة", sound: "الأصوات",
    motion: "الحركة", cardSize: "حجم الورق", confirm: "تأكيد لعب الورقة",
    hints: "تلميحات اللعب", enabled: "مفعّل", disabled: "مطفي", dealSummary: "ملخص التوزيعة",
    result: "نتيجة الجولة", next: "الجولة الجاي", made: "نجح الطلب",
    team: "فريقك", opponents: "الخصوم", total: "المجموع", closePanel: "رجوع للطاولة",
    rulesTitle: "قواعد الطرنيب", rulesLead: "أربع لاعبين، شراكة وشراكة، واللي بطلب لازم يجيب قدّ طلبه.",
    ruleOne: "كل لاعب معه ١٣ ورقة. اللعب بالدور، والشريك قاعد مقابلك.",
    ruleTwo: "اطلب من ٧ لـ١٣ أو احكي باس. أعلى طلب بختار الطرنيب.",
    ruleThree: "اتبع نفس النوع إذا معك. أعلى طرنيب أو أعلى ورقة من النوع اللي بلّش بتاخذ الأكلة.",
    ruleFour: "التوزيعة فيها ١٣ أكلة. النتيجة حسب الطلب والأكلات اللي جابها الفريق.",
    gameMode: "اللعب المحلي", modeDesc: "إنت وشريكك ضد فريق الذكاء الاصطناعي",
    newDeal: "توزيعة جديدة", match: "نتيجة المباراة", playAgain: "العب مرة ثانية",
    language: "العربية", English: "EN", demoNote: "طرنيب محلي · الهدف ٣١ نقطة",
    played: "اختيار تجريبي", startTitle: "قعدة طرنيب على السريع",
    startDesc: "طاولة محلية بواجهة عربية. العب، جرّب الطلب، أو لفّ على الإعدادات.",
    quickStart: "العب الآن", scoreSheet: "كشف النتيجة", winning: "أكلة",
  },
  en: {
    brand: "Tarneeb", ai: "AI", score: "Score", myTeam: "Your team", rivals: "Opponents",
    contract: "Bid", tricks: "Tricks", round: "Deal", playerName: "دعبول", partner: "Partner", left: "Opponent 1",
    right: "Opponent 2", you: "You", dealer: "Dealer", turn: "Your turn — choose a card",
    hand: "Your hand", cards: "cards", play: "Play", bid: "Bid", trump: "Trump",
    tapCard: "Press once to select, then press the same card to play", highest: "High bid", bidPrompt: "Your opening bid", firstBid: "Start at 7", auctionStart: "Bidding starts at 7", pass: "Pass",
    bidContinues: "Passed — bidding continues", finishAuction: "Finish bidding", auctionDone: "Bidding complete", chooseTrumpTitle: "Choose trump", chooseTrumpDesc: "Choose a suit before play begins", playBegins: "All four players begin the deal", firstTrick: "Your turn — lead the first trick",
    chooseTrump: "Choose trump", menu: "Menu", setup: "Game setup", start: "Start deal",
    settings: "Settings", rules: "How to play", close: "Close", demo: "Local game · AI",
    classic: "Classic Tarneeb", target: "Target", points: "points", sound: "Sound",
    motion: "Motion", cardSize: "Card size", confirm: "Confirm card play",
    hints: "Play hints", enabled: "On", disabled: "Off", dealSummary: "Deal summary",
    result: "Round result", next: "Next deal", made: "Contract made",
    team: "Your team", opponents: "Opponents", total: "Total", closePanel: "Back to table",
    rulesTitle: "Tarneeb rules", rulesLead: "Four players, two partnerships. The bidding team needs to make its contract.",
    ruleOne: "Each player gets 13 cards. Play goes around the table; your partner sits opposite.",
    ruleTwo: "Bid from 7 to 13 or pass. The highest bidder chooses trump.",
    ruleThree: "Follow suit if you can. Highest trump or highest card in the led suit takes the trick.",
    ruleFour: "A deal has 13 tricks. The score depends on the bid and the tricks your partnership wins.",
    gameMode: "Local table", modeDesc: "You and your AI partner against two AI opponents",
    newDeal: "New deal", match: "Match result", playAgain: "Play again",
    language: "EN", English: "العربية", demoNote: "Local Tarneeb · First to 31 points",
    played: "Demo selection", startTitle: "A quick game of Tarneeb",
    startDesc: "A local table with an Arabic-first interface. Try a bid, play a card, or explore settings.",
    quickStart: "Play now", scoreSheet: "Score sheet", winning: "tricks",
  },
};

function SuitMark({ suit, large = false }: { suit: Suit; large?: boolean }) {
  return <span className={`tai-suit ${suit === "♥" || suit === "♦" ? "is-red" : ""} ${large ? "is-large" : ""}`}>{suit}</span>;
}

function CardFace({ rank, suit, small = false }: { rank: string; suit: Suit; small?: boolean }) {
  return (
    <span className={`tai-card-face ${small ? "is-small" : ""} ${suit === "♥" || suit === "♦" ? "is-red" : ""}`}>
      <span className="tai-card-corner"><b>{rank}</b><i>{suit}</i></span>
      <span className="tai-card-center">{suit}</span>
      <span className="tai-card-corner bottom"><b>{rank}</b><i>{suit}</i></span>
    </span>
  );
}

function BackFan({ count = 5 }: { count?: number }) {
  return <span className="tai-back-fan" aria-label={`${count} hidden cards`}>
    {Array.from({ length: Math.min(count, 5) }).map((_, i) => <i key={i} style={{ "--card-i": i } as CSSProperties} />)}
    <b>{count}</b>
  </span>;
}

export function TarneebAI() {
  const [language, setLanguage] = useState<Language>("ar");
  const [overlay, setOverlay] = useState<Overlay>(null);
  const [game, setGame] = useState<Game | null>(null);
  const [selectedCardKey, setSelectedCardKey] = useState<string | null>(null);
  const selection = useRef<string | null>(null);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const pending = useRef(false);
  const [soundOn, setSoundOn] = useState(false);
  const [hintsOn, setHintsOn] = useState(true);
  const [motion, setMotion] = useState("normal");
  const [cardSize, setCardSize] = useState("regular");
  const t = text[language];
  const rtl = language === "ar";
  const table = game?.table;
  const phase: Phase = table?.phase === "BID" ? "bidding" : table?.phase === "DECLARE" ? "trump" : "play";
  const handCards = (table?.own_hand ?? []).map(parseCard);
  const selectedBid = table?.contract ?? table?.current_high_bid ?? 0;
  const scores = table?.match_score_by_team ?? [0, 0];
  const tricks = table?.tricks_won_by_team ?? [0, 0];
  const trump = table?.trump ? suits[suitNames.indexOf(table.trump)] : null;
  const summary = game?.history.filter(d => !d.passed_out).at(-1);
  const legal = (id: number) => !!game?.legal_actions.some(a => a.action_id === id);
  const getCardKey = (card: Card) => `${card.rank}-${card.suit}`;
  const seatName = (seat: number | null | undefined) => seat == null ? "—" : [t.you, t.right, t.partner, t.left][seat];
  const displayedTrick = table?.current_trick.length ? table.current_trick : game?.last_trick?.cards ?? [];
  const setupStarted = !!table;

  function accept(next: Game) {
    setGame(next);
    selection.current = null;
    setSelectedCardKey(null);
    if (!next.table) setOverlay("menu");
    else if (next.match_complete) setOverlay("results");
    else if (next.deal_complete) setOverlay("summary");
    else if (next.table.phase === "DECLARE" && next.table.seat_on_turn === 0) setOverlay("trump");
    else setOverlay(current => current === "trump" || current === "setup" || current === "summary" ? null : current);
  }

  async function refresh() {
    try { accept(await requestGame()); setNotice(""); }
    catch (error) { setNotice(String(error)); }
  }
  useEffect(() => { void refresh(); }, []);

  async function send(command: string, action?: number) {
    if (pending.current || !game) return;
    pending.current = true;
    setBusy(true);
    setNotice("");
    try {
      accept(await requestGame(command, game.revision, action));
      if (soundOn && command === "action") {
        const audio = new AudioContext();
        const tone = audio.createOscillator();
        const gain = audio.createGain();
        gain.gain.value = 0.025; tone.connect(gain); gain.connect(audio.destination);
        tone.start(); tone.stop(audio.currentTime + 0.07);
        tone.onended = () => { void audio.close(); };
      }
    } catch (error) {
      setNotice(String(error));
      // A lost HTTP response may follow a committed move. Reconcile, never retry it.
      try { accept(await requestGame()); } catch { /* Keep the last confirmed table. */ }
    } finally { pending.current = false; setBusy(false); }
  }

  useEffect(() => {
    if (!table || game?.deal_complete || table.seat_on_turn === 0 || busy || notice) return;
    const timer = window.setTimeout(() => { void send("advance"); }, 700);
    return () => window.clearTimeout(timer);
  }, [game, busy, notice]);

  function registerCardTap(card: Card) {
    const result = cardPress(selection.current, getCardKey(card), legal(cardId(card)), pending.current);
    selection.current = result.selected;
    setSelectedCardKey(result.selected);
    if (result.submit) void send("action", cardId(card));
  }
  const submitBid = (bid: number | null) => void send("action", bid === null ? 59 : bid + 45);
  const chooseTrump = (suit: Suit) => void send("action", 60 + suits.indexOf(suit));
  const resetDeal = () => void send("next");
  const startDemo = () => { setOverlay(null); if (!table || game?.match_complete) void send("start"); };

  return (
    <div className={`tai-root ${motion === "reduced" ? "reduce-motion" : ""} ${cardSize === "large" ? "large-cards" : ""}`} dir={rtl ? "rtl" : "ltr"} lang={language}>
      <style>{`
        .tai-root.reduce-motion *, .tai-root.reduce-motion *:before { animation-duration: .01ms !important; transition-duration: .01ms !important; }
        .tai-root.large-cards .tai-hand-card { zoom: 1.08; }
        .tai-seat.is-active .tai-avatar { box-shadow: 0 0 0 3px var(--gold); }
        .tai-hand-card:disabled { cursor: not-allowed; }
        .tai-hand-card.is-illegal { filter: brightness(.55); }
        .tai-root {
          --night: #171b1b; --night-soft: #202727; --edge: rgba(230,226,206,.13);
          --felt: #11110f; --felt-hi: #1d1a14; --felt-shadow: #080807;
          --ivory: #f4f0df; --muted: #b7c5b8; --gold: #d5bd82; --red: #c34c4b;
          position: relative; isolation: isolate; display: flex; flex-direction: column;
          width: 100%; height: 100dvh; min-height: 640px; overflow: hidden; box-sizing: border-box;
          padding: 10px clamp(12px, 2.6vw, 34px) 8px; gap: 7px; color: var(--ivory);
          background:
            radial-gradient(ellipse at 50% 54%, rgba(59,87,68,.14), transparent 57%),
            repeating-linear-gradient(135deg, rgba(255,255,255,.012) 0 1px, transparent 1px 5px),
            var(--night);
          font-family: "Tahoma", "Noto Sans Arabic", system-ui, sans-serif;
          -webkit-font-smoothing: antialiased;
        }
        .tai-root * { box-sizing: border-box; }
        .tai-root button { color: inherit; font: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent; }
        .tai-root button:focus-visible { outline: 2px solid var(--gold); outline-offset: 3px; }
        .tai-topbar { height: 45px; flex: 0 0 45px; display: flex; align-items: center; justify-content: space-between; gap: 10px; }
        .tai-brand-side, .tai-actions { display: flex; align-items: center; gap: 8px; }
        .tai-icon-btn, .tai-lang-btn {
          border: 1px solid var(--edge); background: #242b2a; color: var(--ivory);
          border-radius: 12px; display: inline-flex; align-items: center; justify-content: center;
          min-width: 38px; height: 36px; transition: background .18s ease, transform .18s ease;
        }
        .tai-icon-btn:hover, .tai-lang-btn:hover { background: #303a35; transform: translateY(-1px); }
        .tai-icon-btn { font-size: 17px; }
        .tai-lang-btn { padding: 0 11px; font-size: 12px; color: var(--muted); direction:ltr; }
        .tai-brand { display: flex; align-items: baseline; gap: 6px; white-space: nowrap; }
        .tai-brand-word { font-size: 20px; font-weight: 800; letter-spacing: -.7px; }
        .tai-brand-ai { font: 700 11px/1 ui-monospace, monospace; color: var(--gold); letter-spacing: .1em; }
        .tai-brand-kicker { color: #91a296; font-size: 10px; margin-inline-start: 3px; }
        .tai-scorebar {
          flex: 0 0 42px; width: min(100%, 560px); align-self: center; border: 1px solid var(--edge);
          background: rgba(32,40,37,.86); border-radius: 13px; display: flex; align-items: center;
          justify-content: space-between; padding: 0 13px; cursor: pointer;
          transition: border-color .2s ease, background .2s ease;
        }
        .tai-scorebar:hover { border-color: rgba(213,189,130,.48); background: #26302b; }
        .tai-score-team { display: flex; align-items: center; gap: 7px; white-space: nowrap; font-size: 11px; color: var(--muted); }
        .tai-score-team strong { color: var(--ivory); font: 700 18px/1 ui-monospace, monospace; }
        .tai-score-team.is-own strong { color: #d7c58f; }
        .tai-score-mid { display: flex; align-items: center; gap: 11px; color: #a6b3a8; font-size: 10px; white-space: nowrap; }
        .tai-score-mid b { color: var(--ivory); font-weight: 700; }
        .tai-stage {
          min-height: 260px; flex: 1 1 auto; position: relative; overflow: hidden; border-radius: 25px;
          border: 1px solid rgba(222,190,112,.34);
          background:
            radial-gradient(circle, rgba(222,190,112,.22) 1px, transparent 1.8px) 0 0 / 22px 22px,
            radial-gradient(circle, rgba(222,190,112,.12) .7px, transparent 1.7px) 11px 11px / 22px 22px,
            radial-gradient(ellipse at 50% 46%, rgba(150,116,53,.11), transparent 58%),
            linear-gradient(145deg, #201d17 0%, var(--felt) 48%, #090a09 100%);
          box-shadow: inset 0 0 0 5px rgba(12,11,9,.5), inset 0 0 52px rgba(0,0,0,.38), 0 13px 30px rgba(0,0,0,.2);
        }
        .tai-stage:before { content:""; position:absolute; inset: 12px; border: 1px solid rgba(231,218,173,.13); border-radius: 18px; pointer-events:none; }
        .tai-stage:after { content:""; position:absolute; inset:0; pointer-events:none; opacity:.18; background: radial-gradient(ellipse at center, transparent 48%, rgba(0,0,0,.5)); }
        .tai-stage-brand { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: rgba(235,225,185,.09); font: 800 clamp(38px, 7vw, 82px)/1 Georgia,serif; letter-spacing: .13em; white-space: nowrap; pointer-events:none; }
        .tai-stage-label { position:absolute; top:17px; left:50%; transform:translateX(-50%); z-index:2; padding:5px 10px; border:1px solid rgba(245,239,214,.19); border-radius:999px; background:rgba(9,47,35,.43); color:#e0e3d0; font-size:10px; white-space:nowrap; }
        .tai-seat { position:absolute; z-index:2; display:flex; align-items:center; gap:6px; }
        .tai-seat-top { top: 8%; left:50%; transform:translateX(-50%); flex-direction:column; gap:4px; }
        .tai-seat-left { top:49%; left:4.5%; transform:translateY(-50%); flex-direction:column; gap:5px; }
        .tai-seat-right { top:49%; right:4.5%; transform:translateY(-50%); flex-direction:column; gap:5px; }
        .tai-avatar {
          width:36px; height:36px; display:grid; place-items:center; border-radius:50%;
          border:1px solid rgba(237,223,180,.5); color:#efe8cf; font-size:13px; font-weight:800;
          background: radial-gradient(circle at 32% 25%, #879579, #40574a 55%, #273e36);
          box-shadow: 0 0 0 3px rgba(11,52,39,.65), 0 4px 12px rgba(0,0,0,.3);
        }
        .tai-avatar.is-own { background: radial-gradient(circle at 30% 24%, #c6a866, #76613d 58%, #403c2f); }
        .tai-seat-name { color:#f1eedf; font-size:10px; text-align:center; white-space:nowrap; text-shadow:0 1px 4px rgba(0,0,0,.6); }
        .tai-seat-role { color:#c4d0c2; font-size:8px; text-align:center; white-space:nowrap; text-shadow:0 1px 4px rgba(0,0,0,.6); }
        .tai-seat-meta { display:flex; align-items:center; gap:4px; color:#c4d0c2; font-size:9px; }
        .tai-dealer { border-radius:5px; background:rgba(219,197,137,.2); color:#f2d99e; padding:1px 4px; font:700 9px ui-monospace,monospace; }
        .tai-back-fan { position:relative; display:flex; align-items:center; height:23px; direction:ltr; padding-inline:3px; }
        .tai-back-fan i {
          width:16px; height:22px; display:block; margin-left:-8px; border-radius:4px; border:1px solid #dbc9a2;
          background-color:#8d4848;
          background-image: repeating-linear-gradient(45deg, transparent 0 3px, rgba(246,226,193,.48) 3px 4px), repeating-linear-gradient(-45deg, transparent 0 4px, rgba(246,226,193,.27) 4px 5px);
          box-shadow:0 2px 4px rgba(0,0,0,.26); transform:translateY(calc(var(--card-i) * -1px));
        }
        .tai-back-fan i:first-child { margin-left:0; }
        .tai-back-fan b { position:relative; z-index:4; font:700 9px ui-monospace,monospace; color:#e8ddc0; margin-inline-start:4px; }
        .tai-seat-left .tai-back-fan, .tai-seat-right .tai-back-fan { transform:scale(.86); }
        .tai-trick { position:absolute; left:50%; top:51%; width:160px; height:156px; transform:translate(-50%,-50%); z-index:3; }
        .tai-trick-card { position:absolute; width:48px; height:68px; filter:drop-shadow(0 5px 5px rgba(0,0,0,.28)); }
        .tai-trick-card.top { top:0; left:50%; transform:translateX(-50%) rotate(-3deg); }
        .tai-trick-card.right { right:0; top:50%; transform:translateY(-50%) rotate(5deg); }
        .tai-trick-card.bottom { bottom:0; left:50%; transform:translateX(-50%) rotate(4deg); }
        .tai-trick-card.left { left:0; top:50%; transform:translateY(-50%) rotate(-5deg); }
        .tai-card-face { position:relative; display:block; width:100%; height:100%; overflow:hidden; border:1px solid #e7dfcb; border-radius:7px; background:#f6f2e6; color:#202522; box-shadow:inset 0 0 0 2px rgba(255,255,255,.65); }
        .tai-card-face.is-red { color:#b94344; }
        .tai-card-corner { position:absolute; top:4px; left:5px; display:flex; flex-direction:column; align-items:center; gap:0; font-style:normal; font-size:11px; line-height:1; }
        .tai-card-corner b { font-size:12px; font-weight:800; }
        .tai-card-corner i { font-style:normal; font-size:10px; }
        .tai-card-corner.bottom { top:auto; left:auto; right:5px; bottom:4px; transform:rotate(180deg); }
        .tai-card-center { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:24px; }
        .tai-card-face.is-small .tai-card-corner { top:3px; left:3px; font-size:9px; }
        .tai-card-face.is-small .tai-card-corner b { font-size:10px; }
        .tai-card-face.is-small .tai-card-corner i { font-size:9px; }
        .tai-card-face.is-small .tai-card-corner.bottom { left:auto; right:3px; top:auto; bottom:3px; }
        .tai-card-face.is-small .tai-card-center { font-size:19px; }
        .tai-trick-center { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:31px; height:31px; border-radius:50%; background:rgba(10,55,40,.54); border:1px solid rgba(241,225,179,.19); display:grid; place-items:center; color:#d9d0af; font-size:10px; }
        .tai-table-info { position:absolute; z-index:3; left:50%; top:calc(51% + 91px); transform:translateX(-50%); display:flex; align-items:center; gap:7px; white-space:nowrap; }
        .tai-auction-center { position:absolute; z-index:3; left:50%; top:50%; transform:translate(-50%,-50%); display:flex; align-items:center; gap:8px; white-space:nowrap; color:#e4e0cc; font-size:11px; }
        .tai-auction-center b { display:grid; place-items:center; width:38px; height:38px; border:1px solid rgba(235,216,160,.38); border-radius:50%; background:rgba(11,53,40,.76); color:#f0d99c; font:800 13px ui-monospace,monospace; box-shadow:0 0 0 5px rgba(11,53,40,.24); }
        .tai-info-chip { border:1px solid rgba(234,228,202,.2); border-radius:8px; background:rgba(12,52,40,.68); padding:5px 8px; color:#dce2d0; font-size:10px; box-shadow:0 3px 8px rgba(0,0,0,.12); }
        .tai-info-chip strong { color:#f1dfae; font-weight:700; }
        .tai-notice { position:absolute; z-index:8; left:50%; bottom:14px; transform:translateX(-50%); border:1px solid rgba(238,218,165,.36); border-radius:10px; padding:8px 12px; background:rgba(18,30,26,.95); color:#f6f0dc; font-size:11px; text-align:center; box-shadow:0 7px 22px rgba(0,0,0,.24); max-width:calc(100% - 28px); animation:tai-pop .2s ease-out; }
        .tai-hand-area { flex:0 0 auto; display:flex; flex-direction:column; align-items:center; gap:6px; padding-top:1px; }
        .tai-hand-head { width:min(100%, 660px); display:flex; align-items:center; justify-content:space-between; padding:0 3px; min-height:18px; }
        .tai-hand-title { display:flex; align-items:center; gap:7px; font-size:11px; color:#e8e4d4; }
        .tai-hand-title strong { font-size:11px; }
        .tai-hand-count { color:#9eada0; font-size:9px; letter-spacing:.05em; }
        .tai-turn-label { display:flex; align-items:center; gap:5px; color:#dcca9b; font-size:10px; }
        .tai-turn-dot { width:6px; height:6px; background:#d2bb7b; border-radius:50%; box-shadow:0 0 0 3px rgba(210,187,123,.12); }
        .tai-hand { width:min(100%, 680px); height:108px; display:flex; justify-content:center; align-items:flex-start; direction:ltr; padding:3px 1px 0; }
        .tai-hand-card { flex:0 0 clamp(35px, 4.1vw, 56px); width:clamp(35px, 4.1vw, 56px); height:104px; min-width:0; padding:0; margin-left:-23px; border:0; background:transparent; position:relative; transition:transform .18s ease, filter .18s ease; border-radius:7px; touch-action:none; user-select:none; }
        .tai-hand-card:first-child { margin-left:0; }
        .tai-hand-card .tai-card-face { width:100%; height:100%; box-shadow:0 4px 10px rgba(0,0,0,.25), inset 0 0 0 2px rgba(255,255,255,.7); border-radius:7px; }
        .tai-hand-card:hover, .tai-hand-card:focus-visible { transform:translateY(-9px); z-index:30 !important; }
        .tai-hand-card.is-selected { transform:translateY(-7px); z-index:29 !important; }
        .tai-hand-card.is-dragging { z-index:60 !important; opacity:.94; transition:none; }
        .tai-phase-bar { display:flex; justify-content:center; align-items:center; gap:6px; width:100%; min-height:38px; }
        .tai-phase-tab { min-width:83px; min-height:34px; border:1px solid rgba(235,229,205,.13); border-radius:10px; padding:0 11px; background:rgba(36,44,40,.86); color:#b8c4b8; font-size:11px; transition:all .18s ease; }
        .tai-phase-tab:hover { border-color:rgba(215,197,145,.4); color:var(--ivory); }
        .tai-phase-tab.active { color:#17241d; background:#d1bd85; border-color:#e2d2a6; font-weight:800; box-shadow:0 3px 10px rgba(0,0,0,.16); }
        .tai-action-state { display:flex; align-items:center; justify-content:center; gap:6px; min-height:34px; }
        .tai-auction-popup { width:min(100%, 410px); flex-direction:column; align-items:stretch; gap:6px; padding:7px 9px 8px; border:1px solid rgba(232,216,172,.25); border-radius:12px; background:linear-gradient(145deg,rgba(40,54,46,.97),rgba(27,39,33,.97)); box-shadow:0 8px 22px rgba(0,0,0,.24); animation:tai-rise .2s ease-out; }
        .tai-auction-heading { display:flex; align-items:center; justify-content:space-between; gap:8px; color:#f0e5bf; font-size:9px; }
        .tai-auction-heading strong { font-size:10px; }
        .tai-auction-start { border:1px solid rgba(219,197,137,.19); border-radius:999px; padding:3px 7px; color:#c8c9b5; background:rgba(255,255,255,.03); white-space:nowrap; }
        .tai-auction-options { display:flex; align-items:center; justify-content:center; gap:4px; }
        .tai-auction-continue { min-height:31px; border:1px solid rgba(219,197,137,.28); border-radius:9px; padding:0 12px; background:rgba(204,177,111,.16); color:#f0e5bf; font-size:10px; white-space:nowrap; }
        .tai-auction-continue:hover { background:rgba(204,177,111,.26); }
        .tai-bid-number { min-width:38px; height:34px; border-radius:9px; border:1px solid rgba(240,232,207,.23); background:#26332d; color:#f4efdd; font:700 13px ui-monospace,monospace; }
        .tai-bid-number:hover { background:#405548; transform:translateY(-1px); }
        .tai-pass { min-width:62px; height:34px; border-radius:9px; border:1px solid rgba(201,115,104,.52); background:rgba(112,49,47,.55); color:#f2d4ce; font-size:11px; }
        .tai-trump-choice { display:flex; gap:6px; }
        .tai-suit-choice { width:43px; height:35px; border:1px solid rgba(234,228,204,.2); border-radius:9px; background:#26332d; color:#f4f0df; font-size:21px; }
        .tai-suit-choice:hover { background:#42564a; transform:translateY(-1px); }
        .tai-trump-options { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:18px; direction:ltr; }
        .tai-trump-option { min-height:76px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:7px; border:1px solid rgba(221,194,125,.24); border-radius:12px; background:rgba(245,239,220,.04); color:#eee7d1; font-size:11px; transition:background .16s ease, transform .16s ease, border-color .16s ease; }
        .tai-trump-option .tai-suit { font-size:27px; line-height:1; }
        .tai-trump-option .tai-suit.is-red { color:#d5655d; }
        .tai-trump-option:hover { border-color:rgba(221,194,125,.62); background:rgba(221,194,125,.12); transform:translateY(-2px); }
        .tai-trump-note { margin-top:12px; color:#aab8ad; font-size:10px; text-align:center; line-height:1.6; }
        .tai-suit { color:inherit; }
        .tai-suit.is-red { color:#c94e4d; }
        .tai-suit.is-large { font-size:19px; }
        .tai-bottom-tools { display:flex; justify-content:center; align-items:center; gap:5px; flex-wrap:wrap; }
        .tai-tool { min-height:27px; border:0; padding:0 8px; border-radius:8px; background:transparent; color:#9daa9f; font-size:9px; }
        .tai-tool:hover { background:rgba(235,233,215,.08); color:var(--ivory); }
        .tai-prototype-tag { position:absolute; left:50%; bottom:5px; transform:translateX(-50%); color:rgba(202,211,199,.56); font-size:8px; white-space:nowrap; }
        .tai-card-face.is-small { width:100%; height:100%; }
        .tai-overlay { position:absolute; z-index:20; inset:0; display:grid; place-items:center; padding:20px; background:rgba(9,15,13,.63); backdrop-filter:blur(6px); animation:tai-fade .18s ease-out; }
        .tai-dialog { position:relative; width:min(100%, 440px); max-height:min(82dvh, 690px); overflow:auto; border:1px solid rgba(224,210,168,.25); border-radius:20px; background:linear-gradient(160deg,#28332e,#1e2723 70%); box-shadow:0 22px 65px rgba(0,0,0,.48); padding:23px; animation:tai-rise .22s ease-out; }
        .tai-dialog.is-wide { width:min(100%, 550px); }
        .tai-dialog-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:18px; }
        .tai-dialog-title { font-size:19px; line-height:1.35; font-weight:800; color:#f3efdf; margin:0; }
        .tai-dialog-subtitle { margin:5px 0 0; color:#aab8ad; font-size:11px; line-height:1.7; }
        .tai-close { width:32px; height:32px; flex:0 0 32px; border:1px solid var(--edge); border-radius:10px; background:rgba(255,255,255,.04); color:#ccd5c8; }
        .tai-close:hover { background:rgba(255,255,255,.11); }
        .tai-menu-logo { display:flex; align-items:baseline; gap:7px; margin-bottom:4px; }
        .tai-menu-logo b { font-size:29px; letter-spacing:-1px; }
        .tai-menu-logo span { color:var(--gold); font:700 12px ui-monospace,monospace; }
        .tai-menu-art { height:134px; position:relative; margin:15px 0 17px; display:grid; place-items:center; border-radius:15px; border:1px solid rgba(228,217,181,.14); overflow:hidden; background:radial-gradient(ellipse at center,#2d7652,#174b39 68%,#153d31); }
        .tai-menu-art:before { content:""; position:absolute; width:210px; height:100px; border:1px solid rgba(232,218,173,.17); border-radius:50%; }
        .tai-art-card { position:absolute; width:61px; height:88px; box-shadow:0 8px 18px rgba(0,0,0,.3); }
        .tai-art-card.one { transform:translate(-25px,2px) rotate(-13deg); }
        .tai-art-card.two { transform:translate(22px,-3px) rotate(11deg); }
        .tai-art-mark { position:absolute; color:rgba(239,225,178,.2); font:24px Georgia,serif; }
        .tai-primary { min-height:46px; width:100%; border:1px solid #e6d5a4; border-radius:12px; background:#d1bb7c; color:#253129 !important; font-size:14px !important; font-weight:800 !important; transition:transform .16s ease, background .16s ease; }
        .tai-primary:hover { background:#e3cf96; transform:translateY(-1px); }
        .tai-secondary-actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
        .tai-secondary { min-height:39px; border:1px solid rgba(230,226,206,.15); border-radius:11px; background:rgba(250,247,227,.04); color:#d8ddce; font-size:11px !important; }
        .tai-secondary:hover { background:rgba(250,247,227,.1); }
        .tai-demo-line { display:flex; align-items:center; justify-content:space-between; gap:10px; color:#aab7aa; font-size:10px; margin-top:15px; }
        .tai-status-pill { border-radius:999px; border:1px solid rgba(214,192,136,.24); color:#d5c38e; padding:5px 8px; white-space:nowrap; }
        .tai-setting-row { min-height:48px; display:flex; justify-content:space-between; align-items:center; gap:14px; border-top:1px solid rgba(234,230,210,.09); color:#e3e6d8; font-size:12px; }
        .tai-setting-row small { display:block; color:#87978b; font-size:9px; margin-top:3px; }
        .tai-toggle { min-width:52px; height:28px; border:1px solid rgba(230,226,206,.2); border-radius:999px; background:#303d35; color:#bfc8ba; font-size:9px !important; }
        .tai-toggle.on { background:#627b58; border-color:#8a9c68; color:#fff4d5; }
        .tai-select { min-width:88px; min-height:30px; border-radius:8px; border:1px solid rgba(230,226,206,.16); background:#29332e; color:#dfe4d8; padding:0 8px; font-size:10px !important; }
        .tai-rules-list { display:grid; gap:9px; margin-top:15px; }
        .tai-rule { display:flex; gap:10px; padding:11px; border:1px solid rgba(234,230,210,.09); border-radius:11px; color:#d3dbce; font-size:11px; line-height:1.7; }
        .tai-rule i { flex:0 0 23px; height:23px; display:grid; place-items:center; border-radius:7px; background:rgba(214,192,136,.13); color:#dfca8e; font-style:normal; font-size:10px; }
        .tai-score-table { width:100%; border-collapse:collapse; margin-top:5px; font-size:11px; }
        .tai-score-table th, .tai-score-table td { padding:11px 8px; text-align:start; border-bottom:1px solid rgba(234,230,210,.09); }
        .tai-score-table th { color:#aebbad; font-weight:500; }
        .tai-score-table td { color:#e9e8d9; }
        .tai-score-table td:first-child { color:#d8c58f; }
        .tai-result-hero { text-align:center; margin:7px 0 18px; }
        .tai-result-mark { width:48px; height:48px; margin:0 auto 9px; display:grid; place-items:center; border-radius:50%; border:1px solid rgba(227,205,145,.4); color:#e1cd91; background:rgba(213,189,130,.11); font:22px Georgia,serif; }
        .tai-result-score { display:flex; align-items:center; justify-content:center; gap:14px; margin:17px 0; }
        .tai-result-score div { min-width:100px; border:1px solid rgba(234,230,210,.12); border-radius:12px; padding:11px; background:rgba(255,255,255,.03); }
        .tai-result-score strong { display:block; font:700 26px ui-monospace,monospace; margin-bottom:4px; color:#e0cb8d; }
        .tai-result-score span { color:#aab8ad; font-size:10px; }
        .tai-modal-actions { display:flex; gap:8px; margin-top:16px; }
        .tai-modal-actions > * { flex:1; }
        @keyframes tai-pop { from { opacity:0; transform:translate(-50%,6px) } to { opacity:1; transform:translate(-50%,0) } }
        @keyframes tai-fade { from { opacity:0 } to { opacity:1 } }
        @keyframes tai-rise { from { opacity:.4; transform:translateY(8px) } to { opacity:1; transform:translateY(0) } }
        @media (min-width: 900px) {
          .tai-root { padding-top:13px; padding-bottom:11px; gap:9px; }
          .tai-topbar { height:49px; flex-basis:49px; }
          .tai-stage { width:min(100%, 1320px); align-self:center; }
          .tai-avatar { width:41px; height:41px; font-size:14px; }
          .tai-seat-top { top:7%; }
          .tai-seat-left { left:7%; }
          .tai-seat-right { right:7%; }
          .tai-trick { width:192px; height:182px; }
          .tai-trick-card { width:56px; height:79px; }
          .tai-table-info { top:calc(51% + 107px); }
          .tai-hand { height:121px; }
          .tai-hand-card { flex-basis:58px; width:58px; height:116px; margin-left:-24px; }
          .tai-card-corner { top:5px; left:6px; font-size:12px; }
          .tai-card-corner b { font-size:14px; }
          .tai-card-corner i { font-size:12px; }
          .tai-card-corner.bottom { right:6px; bottom:5px; }
        }
        @media (max-width: 600px) {
          .tai-root { padding:7px 10px 5px; gap:5px; min-height:640px; }
          .tai-topbar { height:39px; flex-basis:39px; gap:5px; }
          .tai-icon-btn, .tai-lang-btn { height:33px; min-width:34px; border-radius:10px; }
          .tai-icon-btn { font-size:15px; }
          .tai-brand-word { font-size:18px; }
          .tai-brand-kicker { display:none; }
          .tai-brand-side, .tai-actions { gap:5px; }
          .tai-scorebar { flex-basis:38px; padding:0 9px; border-radius:11px; }
          .tai-score-team { font-size:9px; gap:5px; }
          .tai-score-team strong { font-size:16px; }
          .tai-score-mid { font-size:9px; gap:6px; }
          .tai-stage { border-radius:20px; min-height:235px; }
          .tai-stage:before { inset:8px; border-radius:14px; }
          .tai-stage-label { top:12px; font-size:8px; padding:4px 8px; }
          .tai-seat-top { top:8%; }
          .tai-seat-left { left:2%; top:50%; }
          .tai-seat-right { right:2%; top:50%; }
          .tai-avatar { width:30px; height:30px; font-size:11px; box-shadow:0 0 0 2px rgba(11,52,39,.65),0 3px 8px rgba(0,0,0,.3); }
          .tai-seat-name { font-size:8px; }
          .tai-seat-role { font-size:7px; }
          .tai-seat-meta { font-size:8px; }
          .tai-back-fan { transform:scale(.86); transform-origin:top center; height:19px; }
          .tai-seat-left .tai-back-fan, .tai-seat-right .tai-back-fan { transform:scale(.72); }
          .tai-trick { width:130px; height:130px; top:53%; }
          .tai-trick-card { width:40px; height:57px; }
          .tai-card-center { font-size:19px; }
          .tai-card-corner { top:3px; left:4px; font-size:9px; }
          .tai-card-corner b { font-size:10px; }
          .tai-card-corner i { font-size:8px; }
          .tai-card-corner.bottom { right:4px; bottom:3px; }
          .tai-trick-center { width:25px; height:25px; font-size:9px; }
          .tai-table-info { top:calc(53% + 75px); gap:4px; }
          .tai-info-chip { padding:4px 6px; font-size:8px; border-radius:7px; }
          .tai-hand-area { gap:3px; }
          .tai-hand-head { min-height:17px; }
          .tai-hand-title, .tai-hand-title strong { font-size:10px; }
          .tai-turn-label { font-size:9px; }
          .tai-hand { height:94px; padding-top:2px; }
          .tai-hand-card { flex-basis:clamp(35px, 12vw, 47px); width:clamp(35px, 12vw, 47px); height:90px; margin-left:-22px; }
          .tai-hand-card .tai-card-corner { top:4px; left:4px; font-size:10px; }
          .tai-hand-card .tai-card-corner b { font-size:11px; }
          .tai-hand-card .tai-card-corner i { font-size:9px; }
          .tai-hand-card .tai-card-corner.bottom { left:auto; right:4px; top:auto; bottom:4px; }
          .tai-hand-card .tai-card-center { font-size:18px; }
          .tai-phase-bar { min-height:32px; gap:5px; }
          .tai-phase-tab { min-width:74px; min-height:30px; padding:0 9px; font-size:10px; }
          .tai-auction-popup { gap:5px; padding:6px 7px 7px; }
          .tai-auction-heading { font-size:8px; }
          .tai-auction-heading strong { font-size:9px; }
          .tai-bid-number { min-width:33px; height:31px; }
          .tai-pass { min-width:53px; height:31px; }
          .tai-bid-number, .tai-pass, .tai-suit-choice { height:31px; min-height:31px; }
          .tai-bid-number { min-width:34px; }
          .tai-trump-options { gap:5px; margin-top:14px; }
          .tai-trump-option { min-height:64px; font-size:10px; }
          .tai-trump-option .tai-suit { font-size:24px; }
          .tai-bottom-tools { min-height:21px; gap:1px; }
          .tai-tool { min-height:22px; padding:0 6px; font-size:8px; }
          .tai-prototype-tag { bottom:2px; font-size:7px; }
          .tai-dialog { padding:18px; border-radius:17px; }
          .tai-overlay { padding:13px; }
        }
        @media (max-height: 720px) and (max-width: 600px) {
          .tai-root { min-height:0; padding-top:5px; gap:4px; }
          .tai-stage { min-height:205px; }
          .tai-hand { height:80px; }
          .tai-hand-card { height:77px; }
          .tai-stage-label { top:8px; }
          .tai-seat-top { top:6%; }
          .tai-trick { top:54%; }
          .tai-table-info { top:calc(54% + 72px); }
        }
        @media (prefers-reduced-motion: reduce) {
          .tai-root *, .tai-root *:before, .tai-root *:after { animation-duration:.01ms !important; transition-duration:.01ms !important; }
        }
      `}</style>

      <header className="tai-topbar">
        <div className="tai-brand-side">
          <button className="tai-icon-btn" aria-label={t.menu} onClick={() => setOverlay("menu")}>☰</button>
          <div className="tai-brand">
            <span className="tai-brand-word">{t.brand}</span>
            <span className="tai-brand-ai">AI</span>
            <span className="tai-brand-kicker">TARNEEB</span>
          </div>
        </div>
        <div className="tai-actions">
          <button className="tai-lang-btn" onClick={() => setLanguage(rtl ? "en" : "ar")} aria-label="Switch language">
            <span dir="rtl">العربية</span><span aria-hidden="true"> / </span><span dir="ltr">EN</span>
          </button>
          <button className="tai-icon-btn" aria-label={t.settings} onClick={() => setOverlay("settings")}>⚙</button>
        </div>
      </header>

      <button className="tai-scorebar" onClick={() => setOverlay("scores")} aria-label={t.scoreSheet}>
        <span className="tai-score-team is-own"><span>{t.myTeam}</span><strong>{scores[0]}</strong></span>
        <span className="tai-score-mid">
          {phase === "bidding"
            ? <b>{t.bid} · ٧–١٣</b>
            : phase === "trump"
              ? <b>{t.contract} {selectedBid} · {t.chooseTrump}</b>
              : <><b>{t.contract} {selectedBid || "—"} {trump && <SuitMark suit={trump} />}</b><span>·</span><span>{t.tricks} {tricks.join("–")}</span></>}
        </span>
        <span className="tai-score-team"><strong>{scores[1]}</strong><span>{t.rivals}</span></span>
      </button>

      <main className="tai-stage" aria-label="Tarneeb table">
        <div className="tai-stage-brand">TARNEEB</div>
        <div className="tai-stage-label">{t.round} {(table?.deal_index ?? 0) + 1} <span aria-hidden="true">·</span> {t.classic}</div>

        <div className={`tai-seat tai-seat-top ${table?.seat_on_turn === 2 ? "is-active" : ""}`}>
          <span className="tai-avatar">د</span>
          <span className="tai-seat-name">{t.playerName}</span>
          <span className="tai-seat-role">{t.partner} · AI</span>
          <span className="tai-seat-meta"><BackFan count={game?.hand_counts?.[2] ?? 0} /></span>
        </div>
        <div className={`tai-seat tai-seat-left ${table?.seat_on_turn === 3 ? "is-active" : ""}`}>
          <span className="tai-avatar">د</span>
          <span className="tai-seat-name">{t.playerName}</span>
          <span className="tai-seat-role">{t.left}</span>
          <span className="tai-seat-meta"><BackFan count={game?.hand_counts?.[3] ?? 0} /></span>
        </div>
        <div className={`tai-seat tai-seat-right ${table?.seat_on_turn === 1 ? "is-active" : ""}`}>
          <span className="tai-avatar">د</span>
          <span className="tai-seat-name">{t.playerName}</span>
          <span className="tai-seat-role">{t.right}</span>
          <span className="tai-seat-meta"><BackFan count={game?.hand_counts?.[1] ?? 0} /></span>
        </div>

        {phase === "play" ? (
          <>
            <div className="tai-trick" aria-label="Current trick">
              {displayedTrick.map(([seat, name]) => <div key={`${seat}-${name}`} className={`tai-trick-card ${["bottom", "right", "top", "left"][seat]}`}><CardFace {...parseCard(name)} small /></div>)}
              <span className="tai-trick-center">{(table?.trick_number ?? 0) + (game?.deal_complete ? 0 : 1)}</span>
            </div>
            <div className="tai-table-info">
              <span className="tai-info-chip">{t.contract} {selectedBid || "—"} {trump && <SuitMark suit={trump} />}</span>
              <span className="tai-info-chip">{t.tricks} {tricks.join("–")}</span>
              <span className="tai-info-chip">{t.dealer}: {seatName(table?.dealer_seat)}</span>
              {game?.last_trick && <span className="tai-info-chip">{rtl ? "آخر أكلة" : "Last trick"}: {seatName(game.last_trick.winner)}</span>}
            </div>
          </>
        ) : (
          <div className="tai-auction-center">
            <b>{selectedBid || "7+"}</b>
            <span>{phase === "bidding" ? t.highest : t.chooseTrump} · {seatName(table?.high_bidder_seat)}</span>
            <span>{t.dealer}: {seatName(table?.dealer_seat)}</span>
            <span>{t.pass}: {table?.seats_that_passed.map(seatName).join("، ") || "—"}</span>
          </div>
        )}
      </main>
      {notice && <div className="tai-notice" style={{position: "fixed", top: 60, bottom: "auto", zIndex: 100}} role="alert">{notice} <button disabled={busy} onClick={() => void refresh()}>{rtl ? "تحديث" : "Refresh"}</button></div>}

      <section className="tai-hand-area" aria-label={t.hand}>
        <div className="tai-hand-head">
          <span className="tai-hand-title"><strong>{t.hand}</strong><span className="tai-hand-count">{handCards.length.toLocaleString(rtl ? "ar-JO" : "en-US")} {t.cards}</span></span>
          <span className="tai-turn-label"><i className="tai-turn-dot" />{game?.deal_complete ? t.dealSummary : `${seatName(table?.seat_on_turn)} · ${phase === "play" ? t.play : phase === "bidding" ? t.bid : t.chooseTrump}`}</span>
        </div>
        <div className="tai-hand">
          {handCards.map((card, index) => {
            const key = getCardKey(card);
            return (
              <button
                key={key}
                className={`tai-hand-card ${selectedCardKey === key ? "is-selected" : ""} ${hintsOn && !legal(cardId(card)) ? "is-illegal" : ""}`}
                style={{
                  "--n": index,
                  zIndex: index + 1,
                } as CSSProperties}
                aria-label={`${card.rank} ${card.suit}. ${t.tapCard}`}
                title={`${card.rank} ${card.suit}`}
                disabled={busy || !legal(cardId(card)) || phase !== "play"}
                aria-pressed={selectedCardKey === key}
                onClick={() => registerCardTap(card)}
              >
                <CardFace rank={card.rank} suit={card.suit} />
              </button>
            );
          })}
        </div>
        <div className="tai-phase-bar" role="group" aria-label="Game phase">
          {(["play", "bidding", "trump"] as Phase[]).map(p => <span key={p} className={`tai-phase-tab ${phase === p ? "active" : ""}`}>{p === "play" ? t.play : p === "bidding" ? t.bid : t.trump}</span>)}
        </div>
        {phase === "bidding" && table?.seat_on_turn === 0 && (
          <div className="tai-action-state tai-auction-popup" role="group" aria-label={t.bidPrompt}>
            <div className="tai-auction-heading"><strong>{t.bidPrompt}</strong><span className="tai-auction-start">{t.highest}: {selectedBid || "—"}</span></div>
            <div className="tai-auction-options">
              {[7, 8, 9, 10, 11, 12, 13].map(bid => <button disabled={busy || !legal(bid + 45)} className="tai-bid-number" key={bid} onClick={() => submitBid(bid)}>{bid}</button>)}
              <button disabled={busy || !legal(59)} className="tai-pass" onClick={() => submitBid(null)}>{t.pass}</button>
            </div>
          </div>
        )}
        {phase === "trump" && <div className="tai-action-state"><span className="tai-hand-count">{t.chooseTrumpDesc}</span></div>}
        {phase === "play" && (
          <div className="tai-action-state"><span className="tai-hand-count">{t.tapCard}</span></div>
        )}
        <div className="tai-bottom-tools">
          <button className="tai-tool" onClick={() => setOverlay("summary")}>{t.dealSummary}</button>
          <button className="tai-tool" onClick={() => setOverlay("results")}>{t.match}</button>
          <button className="tai-tool" onClick={() => setOverlay("rules")}>{t.rules}</button>
          <button className="tai-tool" onClick={() => setOverlay("settings")}>{t.settings}</button>
        </div>
      </section>
      <div className="tai-prototype-tag">{t.demoNote}</div>

      {overlay && (
        <div className="tai-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget && overlay !== "trump") setOverlay(null); }}>
          {overlay === "menu" && (
            <section className="tai-dialog">
              <div className="tai-dialog-head">
                <div><div className="tai-menu-logo"><b>{t.brand}</b><span>AI</span></div><p className="tai-dialog-subtitle">{t.startTitle}</p></div>
                <button className="tai-close" onClick={() => setOverlay(null)} aria-label={t.close}>×</button>
              </div>
              <p className="tai-dialog-subtitle">{t.startDesc}</p>
              <div className="tai-menu-art" aria-hidden="true">
                <span className="tai-art-mark" style={{ left: "18%", top: "19%" }}>♠</span>
                <span className="tai-art-mark" style={{ right: "17%", bottom: "16%", color: "rgba(239,119,108,.34)" }}>♥</span>
                <span className="tai-art-card one"><CardFace rank="A" suit="♠" small /></span>
                <span className="tai-art-card two"><CardFace rank="K" suit="♥" small /></span>
              </div>
              <button className="tai-primary" onClick={() => setOverlay("setup")}>{t.quickStart}</button>
              <div className="tai-secondary-actions">
                <button className="tai-secondary" onClick={() => setOverlay("rules")}>{t.rules}</button>
                <button className="tai-secondary" onClick={() => setOverlay("settings")}>{t.settings}</button>
              </div>
              <div className="tai-demo-line"><span>{t.gameMode}</span><span className="tai-status-pill">{t.demo}</span></div>
              <div className="tai-demo-line"><span>{t.modeDesc}</span><button className="tai-tool" onClick={() => setLanguage(rtl ? "en" : "ar")}><span dir="rtl">العربية</span><span aria-hidden="true"> / </span><span dir="ltr">EN</span></button></div>
            </section>
          )}

          {overlay === "setup" && (
            <section className="tai-dialog">
              <div className="tai-dialog-head">
                <div><h2 className="tai-dialog-title">{t.setup}</h2><p className="tai-dialog-subtitle">{t.classic} · {t.modeDesc}</p></div>
                <button className="tai-close" onClick={() => setOverlay(null)} aria-label={t.close}>×</button>
              </div>
              <div className="tai-setting-row"><span>{t.gameMode}<small>{t.demo}</small></span><span className="tai-status-pill">Local</span></div>
              <div className="tai-setting-row"><span>{t.target}<small>{t.classic}</small></span><b>٣١ {t.points}</b></div>
              <div className="tai-setting-row"><span>{t.partner}<small>{t.you} · {t.partner} / {t.rivals}</small></span><span>٤</span></div>
              <button className="tai-primary" style={{ marginTop: 17 }} disabled={busy || !game} onClick={startDemo}>{setupStarted && !game?.match_complete ? t.closePanel : t.start}</button>
              <p className="tai-dialog-subtitle" style={{ textAlign: "center", marginTop: 12 }}>{t.demoNote}</p>
            </section>
          )}

          {overlay === "trump" && (
            <section className="tai-dialog" role="dialog" aria-modal="true" aria-labelledby="tai-trump-title">
              <div className="tai-dialog-head">
                <div>
                  <h2 className="tai-dialog-title" id="tai-trump-title">{t.chooseTrumpTitle}</h2>
                  <p className="tai-dialog-subtitle">{t.chooseTrumpDesc}</p>
                </div>
              </div>
              <div className="tai-trump-options">
                {(["♣", "♦", "♥", "♠"] as Suit[]).map((suit) => (
                  <button className="tai-trump-option" disabled={busy || !legal(60 + suits.indexOf(suit))} key={suit} onClick={() => chooseTrump(suit)} aria-label={`${t.chooseTrumpTitle} ${suit}`}>
                    <SuitMark suit={suit} />
                    <span>{suit}</span>
                  </button>
                ))}
              </div>
              <p className="tai-trump-note">{t.playBegins} · {t.demo}</p>
            </section>
          )}

          {overlay === "settings" && (
            <section className="tai-dialog">
              <div className="tai-dialog-head">
                <div><h2 className="tai-dialog-title">{t.settings}</h2><p className="tai-dialog-subtitle">{t.demoNote}</p></div>
                <button className="tai-close" onClick={() => setOverlay(null)} aria-label={t.close}>×</button>
              </div>
              <div className="tai-setting-row"><span>{t.language}<small>Language</small></span><button className="tai-select" onClick={() => setLanguage(rtl ? "en" : "ar")}><span dir="rtl">العربية</span><span aria-hidden="true"> / </span><span dir="ltr">EN</span></button></div>
              <div className="tai-setting-row"><span>{t.sound}</span><button className={`tai-toggle ${soundOn ? "on" : ""}`} onClick={() => setSoundOn(!soundOn)}>{soundOn ? t.enabled : t.disabled}</button></div>
              <div className="tai-setting-row"><span>{t.motion}</span><select className="tai-select" value={motion} onChange={e => setMotion(e.target.value)} aria-label={t.motion}><option value="normal">{rtl ? "عادي" : "Normal"}</option><option value="reduced">{rtl ? "خفيف" : "Reduced"}</option></select></div>
              <div className="tai-setting-row"><span>{t.cardSize}</span><select className="tai-select" value={cardSize} onChange={e => setCardSize(e.target.value)} aria-label={t.cardSize}><option value="regular">{rtl ? "عادي" : "Regular"}</option><option value="large">{rtl ? "كبير" : "Large"}</option></select></div>
              <div className="tai-setting-row"><span>{t.confirm}</span><span>{rtl ? "كبستين" : "Two presses"}</span></div>
              <div className="tai-setting-row"><span>{t.hints}</span><button className={`tai-toggle ${hintsOn ? "on" : ""}`} onClick={() => setHintsOn(!hintsOn)}>{hintsOn ? t.enabled : t.disabled}</button></div>
              <div className="tai-modal-actions"><button className="tai-secondary" onClick={() => { setOverlay("menu"); }}>{t.menu}</button><button className="tai-secondary" onClick={() => setOverlay(null)}>{t.closePanel}</button></div>
            </section>
          )}

          {overlay === "rules" && (
            <section className="tai-dialog is-wide">
              <div className="tai-dialog-head">
                <div><h2 className="tai-dialog-title">{t.rulesTitle}</h2><p className="tai-dialog-subtitle">{t.rulesLead}</p></div>
                <button className="tai-close" onClick={() => setOverlay(null)} aria-label={t.close}>×</button>
              </div>
              <div className="tai-rules-list">
                {[t.ruleOne, t.ruleTwo, t.ruleThree, t.ruleFour].map((rule, index) => <div className="tai-rule" key={rule}><i>{["٠١", "٠٢", "٠٣", "٠٤"][index]}</i><span>{rule}</span></div>)}
              </div>
              <div className="tai-modal-actions"><button className="tai-primary" onClick={() => setOverlay(null)}>{t.closePanel}</button></div>
            </section>
          )}

          {overlay === "scores" && (
            <section className="tai-dialog is-wide">
              <div className="tai-dialog-head">
                <div><h2 className="tai-dialog-title">{t.scoreSheet}</h2><p className="tai-dialog-subtitle">{t.demoNote}</p></div>
                <button className="tai-close" onClick={() => setOverlay(null)} aria-label={t.close}>×</button>
              </div>
              <table className="tai-score-table">
                <thead><tr><th>{t.round}</th><th>{t.contract}</th><th>{t.winning}</th><th>{t.team}</th><th>{t.opponents}</th></tr></thead>
                <tbody>{game?.history.map((d, i) => <tr key={i}><td>{d.deal_index + 1}</td><td>{d.passed_out ? t.pass : `${d.winning_bid} ${suits[suitNames.indexOf(d.trump ?? "")] ?? ""}`}</td><td>{d.tricks_won_by_team.join("–")}</td><td>{d.score_change_by_team[0]}</td><td>{d.score_change_by_team[1]}</td></tr>)}</tbody>
                <tfoot><tr><td colSpan={3}>{t.total}</td><td><b>{scores[0]}</b></td><td><b>{scores[1]}</b></td></tr></tfoot>
              </table>
              <div className="tai-modal-actions"><button className="tai-primary" onClick={() => setOverlay(null)}>{t.closePanel}</button></div>
            </section>
          )}

          {overlay === "summary" && (
            <section className="tai-dialog">
              <div className="tai-dialog-head">
                <div><h2 className="tai-dialog-title">{t.dealSummary}</h2><p className="tai-dialog-subtitle">{t.demoNote}</p></div>
                <button className="tai-close" onClick={() => setOverlay(null)} aria-label={t.close}>×</button>
              </div>
              <div className="tai-result-hero"><div className="tai-result-mark">♥</div><h3 className="tai-dialog-title">{summary ? summary.contract_made ? t.made : (rtl ? "فشل الطلب" : "Contract failed") : "—"}</h3><p className="tai-dialog-subtitle">{t.contract}: {summary?.winning_bid ?? "—"} {summary?.trump && <SuitMark suit={suits[suitNames.indexOf(summary.trump)]} />} · {t.tricks}: {summary?.tricks_won_by_team.join("–") ?? "—"}</p></div>
              <div className="tai-result-score">
                <div><strong>{summary?.score_change_by_team[0] ?? "—"}</strong><span>{t.team}</span></div><span>—</span><div><strong>{summary?.score_change_by_team[1] ?? "—"}</strong><span>{t.opponents}</span></div>
              </div>
              <button className="tai-primary" disabled={busy || !game?.deal_complete || game.match_complete} onClick={resetDeal}>{t.next}</button>
            </section>
          )}

          {overlay === "results" && (
            <section className="tai-dialog">
              <div className="tai-dialog-head">
                <div><h2 className="tai-dialog-title">{t.match}</h2><p className="tai-dialog-subtitle">{t.demoNote}</p></div>
                <button className="tai-close" onClick={() => setOverlay(null)} aria-label={t.close}>×</button>
              </div>
              <div className="tai-result-hero"><div className="tai-result-mark">♠</div><h3 className="tai-dialog-title">{game?.match_complete ? (scores[0] >= 31 ? t.team : t.opponents) : t.match}</h3></div>
              <div className="tai-result-score"><div><strong>{scores[0]}</strong><span>{t.team}</span></div><span>—</span><div><strong>{scores[1]}</strong><span>{t.opponents}</span></div></div>
              <div className="tai-modal-actions"><button className="tai-primary" onClick={() => { setOverlay("setup"); }}>{t.playAgain}</button><button className="tai-secondary" onClick={() => setOverlay("menu")}>{t.menu}</button></div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
