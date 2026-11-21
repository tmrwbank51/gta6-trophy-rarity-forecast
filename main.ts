// GTA VI trophy-rarity forecaster — earners/players decomposition in log-time.
//
// The rate of a trophy is earners divided by players tracked. The program
// models the two counts separately, in log-time, and lets dilution fall out
// of the decomposition: when players join faster than earners accumulate,
// the rate falls; when the existing cohort progresses through the game, the
// rate of early trophies rises. The page Exophase prints is a snapshot from
// its last refresh, so the program also corrects for staleness: the counts
// at the checkpoint are higher than the stale page shows, and the correction
// uses the model's own growth rates, never a hard-coded refresh jump.
//
// Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes one
// /output/<id>.out per case: {"trophies": [{"id": n, "q": [q05, q25, q50,
// q75, q95]}, ...]}. Node 22 stdlib only, no network, fully deterministic.
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const RELEASE = Date.parse("2026-11-19T00:00:00Z");

// Prior slopes for log-count growth against u = ln(1 + hours since release).
// Players tracked on launch week grow roughly like (1 + t)^1 while the
// launch saturates; earners of early (bronze) trophies outpace players as
// the cohort progresses, late (gold/platinum) trophies lag behind.
const PRIOR_PLAYER_SLOPE = 1.0;
const TIER_EARNER_OFFSET = { bronze: 0.8, silver: 0.3, gold: -0.3, platinum: -0.8 };

function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}
function logit(pct) {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}
function invLogit(x) {
  return 100 / (1 + Math.exp(-x));
}
function round4(x) {
  return Math.round(x * 10000) / 10000;
}
// Least-squares slope/intercept of y on x; returns null when underdetermined.
function fitLine(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i];
  }
  const den = n * sxx - sx * sx;
  if (Math.abs(den) < 1e-12) return null;
  const slope = (n * sxy - sx * sy) / den;
  return { slope, intercept: (sy - slope * sx) / n };
}

function forecastCase(kase) {
  const trophies = kase.trophies;
  const history = kase.history || [];
  const checkpointMs = Date.parse(kase.at);
  const targetMs = Date.parse(kase.target);
  const horizonDays = (kase.horizon_minutes || 0) / 1440;
  const releaseMs = kase.game && kase.game.release_date
    ? Date.parse(kase.game.release_date + "T00:00:00Z")
    : (history.length ? Date.parse(history[0].at) : RELEASE);
  const hoursSince = (ms) => Math.max(0, (ms - releaseMs) / 3600000);
  const u = (ms) => Math.log(1 + hoursSince(ms));

  // The refresh series: reads where the page actually changed, plus the
  // first and the last read. Exophase refreshes every few hours, so the
  // history is a staircase; fitting the staircase's steps (not the flat
  // treads) is what estimates growth.
  const series = [];
  let lastKey = null;
  let lastChangeMs = history.length ? Date.parse(history[0].at) : checkpointMs;
  let distinct = 0;
  for (const read of history) {
    const key = read.players_tracked + "|" + (read.rates || []).join(",");
    if (key !== lastKey) {
      series.push(read);
      lastChangeMs = Date.parse(read.at);
      lastKey = key;
      distinct++;
    }
  }
  if (history.length && series[series.length - 1] !== history[history.length - 1]) {
    series.push(history[history.length - 1]);
  }
  const last = history.length ? history[history.length - 1] : null;
  const stalenessH = Math.max(0, (checkpointMs - lastChangeMs) / 3600000);

  const uCkpt = u(checkpointMs);
  const uTarget = u(targetMs);
  const dU = Math.max(0, uTarget - uCkpt);
  const tCkptH = hoursSince(checkpointMs);

  // --- players tracked: log-linear fit on the refresh series, shrunk
  // toward the launch prior when the series carries little signal. The
  // blend weight counts distinct snapshots, not reads: a flat page
  // (practice) has one distinct snapshot and no usable signal.
  let playerSlope = PRIOR_PLAYER_SLOPE;
  if (distinct >= 2) {
    const fit = fitLine(series.map((r) => u(Date.parse(r.at))),
      series.map((r) => Math.log(Math.max(1, r.players_tracked))));
    if (fit) {
      const w = Math.min(1, (distinct - 1) / 4);
      playerSlope = (1 - w) * PRIOR_PLAYER_SLOPE + w * clamp(fit.slope, 0, 3);
    }
  }
  const playersLast = last ? last.players_tracked : 1;

  // --- staleness correction: the true counts at the checkpoint exceed the
  // stale page. Players grow at the model's own rate; earners grow at most
  // half that fast over a short stale window (new joiners have earned
  // almost nothing yet), so a refresh dilutes the rates.
  const gPlayer = playerSlope / (1 + tCkptH);
  const playersCorr = playersLast * Math.min(10, Math.exp(gPlayer * stalenessH));
  const playersTarget = playersCorr * Math.exp(playerSlope * dU);

  // --- uncertainty: grows with horizon (pinball punishes overconfidence),
  // plus a refresh term while the page is stale, plus a no-signal term
  // when the history never moved (the model is then running on priors,
  // and an honest spread says so instead of collapsing to the fit).
  const spread = clamp(
    0.15 + 0.35 * Math.sqrt(horizonDays) + (stalenessH > 1 ? 0.2 : 0)
      + (distinct < 2 ? 0.15 : 0),
    0.2, 1.5);

  const rows = [];
  for (let i = 0; i < trophies.length; i++) {
    const trophy = trophies[i];
    const lastRate = last && last.rates && last.rates[i] != null ? last.rates[i] : null;

    // A trophy every tracked player already holds stays held: forecast a
    // tight band at the top instead of letting the logit math run away.
    if (lastRate != null && lastRate >= 99.5) {
      rows.push({ id: trophy.id, q: [98, 99.5, 100, 100, 100] });
      continue;
    }

    // Earners of this trophy across the refresh series.
    const xs = [];
    const ys = [];
    for (const read of series) {
      const r = read.rates ? read.rates[i] : null;
      if (r == null) continue;
      const earners = (r * read.players_tracked) / 100;
      if (earners > 0) {
        xs.push(u(Date.parse(read.at)));
        ys.push(Math.log(earners));
      }
    }
    const tierOffset = TIER_EARNER_OFFSET[trophy.tier] ?? 0;
    const priorSlope = PRIOR_PLAYER_SLOPE + tierOffset;
    let earnerSlope = priorSlope;
    if (xs.length >= 2) {
      const fit = fitLine(xs, ys);
      if (fit) {
        const w = Math.min(1, (xs.length - 1) / 4);
        earnerSlope = (1 - w) * priorSlope + w * clamp(fit.slope, 0, 3);
      }
    }

    const earnersLast = lastRate != null
      ? Math.max(0.5, (lastRate * playersLast) / 100)
      : 0.5;
    const gEarner = Math.min(0.5 * gPlayer, earnerSlope / (1 + tCkptH));
    const earnersCorr = earnersLast * Math.min(10, Math.exp(gEarner * stalenessH));
    const earnersTarget = earnersCorr * Math.exp(earnerSlope * dU);

    const median = clamp((100 * earnersTarget) / playersTarget, 0.01, 99.99);
    const medLogit = logit(median);
    const q = Z.map((z) => round4(clamp(invLogit(medLogit + z * spread), 0.01, 99.99)));
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    rows.push({ id: trophy.id, q });
  }
  return { trophies: rows };
}

function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  mkdirSync(outDir, { recursive: true });
  const files = readdirSync(inDir).filter((f) => f.endsWith(".in")).sort();
  for (const file of files) {
    const kase = JSON.parse(readFileSync(join(inDir, file), "utf8"));
    const out = forecastCase(kase);
    const id = file.replace(/\.in$/, "");
    writeFileSync(join(outDir, id + ".out"), JSON.stringify(out), "utf8");
  }
}

main();
