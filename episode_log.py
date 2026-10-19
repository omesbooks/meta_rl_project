"""Training episode log: where every training episode started on the Train slice.

TradingEnv starts each training episode at a random bar and plays up to `ep_len` bars, so
which part of Train the agent has actually learned from is random. This callback records
every episode (start bar, length, how it ended, reward) to `<run>/episodes.json` after
every rollout, so coverage can be checked while training runs (Monitor: TRAINING DATA
COVERAGE) and afterwards.

Decision bars only: episode e covers bars [start, start + len). Each observation also looks
back `window - 1` bars, which this does not count.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

SCHEMA = "metafxclub.episodes.v1"


def coverage(episodes, n_rows):
    """Times each Train bar was played (decision bars), from a list of {start, len}."""
    diff = np.zeros(int(n_rows) + 1, dtype=np.int64)
    for e in episodes:
        s, n = int(e["start"]), int(e["len"])
        if n <= 0:
            continue
        diff[max(0, s)] += 1
        diff[min(int(n_rows), s + n)] -= 1
    return np.cumsum(diff)[:int(n_rows)]


def coverage_summary(episodes, n_rows, start_min=0):
    """Share of the playable bars (from `start_min`) played at least once, bars never played,
    mean plays per bar, and the coverage of the first and last 10% (the edges are played least)."""
    c = coverage(episodes, n_rows)[int(start_min):]
    if not len(c):
        return {}
    k = max(1, len(c) // 10)
    return {"covered": round(float((c > 0).mean()), 4), "never": int((c == 0).sum()),
            "passes": round(float(c.sum() / len(c)), 3), "max_plays": int(c.max()),
            "first10": round(float((c[:k] > 0).mean()), 4), "last10": round(float((c[-k:] > 0).mean()), 4),
            "episodes": len(episodes), "blowups": sum(1 for e in episodes if e.get("end") == "blowup")}


def _atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def make_episode_callback(path, train_df, window, ep_len, total_steps=0, model_name=""):
    """SB3 callback that logs every training episode of a single-env DummyVecEnv over TradingEnv."""
    from stable_baselines3.common.callbacks import BaseCallback

    path = Path(path)
    n_rows = int(len(train_df))
    ts = train_df["timestamp"].astype(str).tolist() if "timestamp" in train_df.columns else None
    ticks = [[int(i), ts[int(i)][:16]] for i in np.linspace(0, n_rows - 1, 11)] if ts and n_rows else []
    start_min = int(window) - 1

    class EpisodeLog(BaseCallback):
        def _env(self):
            env = self.training_env.envs[0]
            return getattr(env, "unwrapped", env)

        def _begin(self):
            start = getattr(self._env(), "start_t", None)
            self.cur = None if start is None else {
                "i": len(self.eps), "start": int(start), "len": 0, "step0": int(self.num_timesteps),
                "ts0": ts[int(start)][:16] if ts and 0 <= int(start) < n_rows else None, "reward": 0.0}

        def _on_training_start(self):
            self.eps, self.cur, self.t0 = [], None, time.time()
            self._begin()

        def _on_step(self):
            if self.cur is None:
                return True
            self.cur["len"] += 1
            rewards = self.locals.get("rewards")
            if rewards is not None:
                self.cur["reward"] += float(np.asarray(rewards).ravel()[0])
            dones = self.locals.get("dones")
            if dones is not None and bool(np.asarray(dones).ravel()[0]):
                c = self.cur
                c["end"] = ("length" if c["len"] >= ep_len
                            else "data_end" if c["start"] + c["len"] >= n_rows - 2 else "blowup")
                c["reward"] = round(c["reward"], 4)
                self.eps.append(c)
                self._begin()                     # the VecEnv already reset: start_t is the next episode's
            return True

        def _on_rollout_end(self):
            self.write()

        def _on_training_end(self):
            s = self.write(final=True)
            if s:
                print(f"\n[coverage] played {100 * s['covered']:.1f}% of the Train bars at least once "
                      f"({s['never']:,} never) · first 10% {100 * s['first10']:.0f}% · last 10% {100 * s['last10']:.0f}% · "
                      f"{s['passes']:.2f} plays/bar · {s['episodes']} episodes, {s['blowups']} blow-ups -> {path}", flush=True)

        def write(self, final=False):
            eps = list(self.eps)
            if self.cur is not None and self.cur["len"]:
                eps.append(dict(self.cur, end="open", reward=round(self.cur["reward"], 4)))
            summary = coverage_summary(eps, n_rows, start_min)
            doc = {"schema": SCHEMA, "model": model_name, "train_rows": n_rows, "window": int(window),
                   "ep_len": int(ep_len), "start_min": start_min, "start_max": n_rows - int(ep_len) - 2,
                   "ticks": ticks, "total_steps": int(total_steps or 0), "step": int(self.num_timesteps),
                   "final": bool(final), "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "summary": summary, "episodes": eps}
            try:
                _atomic_write(path, json.dumps(doc, separators=(",", ":")))
            except OSError as exc:
                print(f"\n[coverage] episodes.json not written: {exc}", flush=True)
            return summary

    return EpisodeLog()
