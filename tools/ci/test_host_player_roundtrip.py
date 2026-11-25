#!/usr/bin/env python3
"""Public host+player round-trip verification for Daycare.

Tests the real compiled `daycare` server + `daycare-player` binaries against a
local stub Coworld LLM sidecar across:
  1. Valid bare + fenced JSON replies (including >80-rune hunch & >240-rune
     notes clipping) -> episode completes exit 0 with genuine numeric scores,
     model-produced actions executed, no player_failure.json, and private
     trajectory marks all model turns accepted with raw text preserved.
  2. Invalid / ambiguous reply (surrounding prose + extra object) -> episode
     terminates unfinished unscored (exit != 0, player_failure.json with closed
     GamePlayerFailure schema kind == "player_error" and actual failing seat,
     NO retries, NO results.json, NO replay.json).
  3. Provider failure (HTTP 503 / HTTP 500) -> episode terminates unfinished
     unscored (exit != 0, player_failure.json with closed GamePlayerFailure
     schema kind == "episode_inconclusive" and actual interrupted seat,
     NO retries, NO results.json, NO replay.json).
  4. Missing credentials (prompt seat with no LLM endpoint/keys) -> episode
     terminates unfinished unscored (exit != 0, player_failure.json with
     kind == "episode_inconclusive", NO results.json, NO replay.json).
  5. Purely scripted offline certification (caretaker vs stubborn, no LLM env)
     -> completes exit 0 with genuine numeric scores and replay.
"""

import argparse
import http.server
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_healthz(port: int, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"game server on port {port} did not become healthy")


def make_stub_server(mode: str, on_call=None):
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            req = json.loads(raw.decode("utf-8"))
            slot = int(self.headers.get("X-Coworld-Player-Slot", "0"))
            calls.append({"slot": slot, "mode": mode})
            if on_call is not None:
                on_call(len(calls), slot)
            if mode in ("provider_500", "provider_503"):
                status_code = 503 if mode == "provider_503" else 500
                body = json.dumps({"error": f"upstream {status_code}"}).encode()
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if mode == "ambiguous":
                text = (
                    "Here are two candidate orders:\n"
                    '{"job":"watch","guess":"apple"}\n'
                    '{"job":"idle","guess":"banana"}'
                )
            else:
                if slot == 0:
                    # Fenced JSON with >80 hunch and >240 notes (matching live Nov 25 shape)
                    obj = {
                        "job": "provide",
                        "fruit": "banana",
                        "guess": "banana",
                        "hunch": "H" * 110,
                        "notes": "N" * 265,
                    }
                    text = "```json\n" + json.dumps(obj) + "\n```"
                else:
                    # Bare JSON with optional defaults
                    obj = {
                        "job": "show",
                        "fruit": "banana",
                        "hunch": None,
                        "notes": None,
                    }
                    text = json.dumps(obj)

            body = json.dumps(
                {
                    "id": f"msg_{len(calls)}",
                    "type": "message",
                    "role": "assistant",
                    "model": "local-stub-model",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 50, "output_tokens": 30},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


def run_scenario(game_bin: Path, player_bin: Path, name: str, mode: str,
                 scripted_seats: tuple[str, str] = ("", ""), use_llm: bool = True,
                 connect_slots: tuple[int, ...] = (0, 1),
                 connect_timeout_s: int = 10, min_turn_s: int = 0,
                 disconnect_slot: int | None = None,
                 disconnect_on_call: int | None = None,
                 disconnect_delay_s: float = 0.0,
                 expected_seat: int | None = None):
    with tempfile.TemporaryDirectory(prefix=f"daycare-rt-{name}-") as td:
        work = Path(td)
        port = free_port()
        cfg = {
            "seed": 7,
            "turns": 2,
            "ticksPerTurn": 60,
            "minTurnSeconds": min_turn_s,
            "llmTimeoutSeconds": 5,
            "shutdownGraceSeconds": 0,
            "playerConnectTimeoutSeconds": connect_timeout_s,
            "num_agents": 2,
            "players": [{"name": "Alder"}, {"name": "Bramble"}],
            "tokens": ["tok0", "tok1"],
        }
        (work / "config.json").write_text(json.dumps(cfg))

        players_by_slot: dict[int, subprocess.Popen] = {}

        def on_call(call_num: int, _slot: int):
            if disconnect_slot is not None and disconnect_on_call == call_num:
                proc = players_by_slot.get(disconnect_slot)
                if proc is not None and proc.poll() is None:
                    if disconnect_delay_s > 0:
                        def _drop():
                            time.sleep(disconnect_delay_s)
                            if proc.poll() is None:
                                proc.terminate()
                                proc.wait(timeout=5)
                        threading.Thread(target=_drop, daemon=True).start()
                    else:
                        proc.terminate()
                        proc.wait(timeout=5)
                        time.sleep(0.25)

        srv, calls = (None, [])
        if use_llm:
            srv, calls = make_stub_server(mode, on_call=on_call)

        base_env = {
            k: os.environ[k]
            for k in ("PATH", "HOME", "TMPDIR", "LANG")
            if k in os.environ
        }
        game_env = base_env | {
            "COGAME_HOST": "127.0.0.1",
            "COGAME_PORT": str(port),
            "COGAME_CONFIG_URI": (work / "config.json").as_uri(),
            "COGAME_RESULTS_URI": (work / "results.json").as_uri(),
            "COGAME_SAVE_REPLAY_URI": (work / "replay.json").as_uri(),
            "COGAME_PLAYER_FAILURE_URI": (work / "player_failure.json").as_uri(),
            "COGAME_LOG_URI": (work / "game.log").as_uri(),
            "COGAME_SAVE_TRAJECTORY_URI": (work / "trajectory.jsonl").as_uri(),
            "COWORLD_EPISODE_ID": f"rt-{name}",
            "COWORLD_GAME_VERSION": "0.1.7",
            "COWORLD_SOURCE_REVISION": "test-rev",
            "COWORLD_GAME_NAME": "daycare",
        }
        if srv is not None:
            game_env["COWORLD_LLM_ENDPOINT"] = f"http://127.0.0.1:{srv.server_port}"
            game_env["COWORLD_LLM_MODEL"] = "local-stub-model"

        game = subprocess.Popen(
            [str(game_bin)],
            env=game_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        players = []
        try:
            wait_healthz(port)
            for slot in connect_slots:
                penv = base_env | {
                    "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{port}/player?slot={slot}&token=tok{slot}"
                }
                if scripted_seats[slot]:
                    penv["PLAYER_SCRIPTED"] = scripted_seats[slot]
                else:
                    penv["COWORLD_LLM_ENABLED"] = "true"
                p = subprocess.Popen(
                    [str(player_bin)],
                    env=penv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                players.append(p)
                players_by_slot[slot] = p

            game_out, _ = game.communicate(timeout=30)
            for slot, p in players_by_slot.items():
                p.wait(timeout=10)
                if slot != disconnect_slot:
                    assert p.returncode == 0, f"{name}: player {slot} exited {p.returncode}"

            results_exists = (work / "results.json").exists()
            replay_exists = (work / "replay.json").exists()
            failure_exists = (work / "player_failure.json").exists()
            traj_path = work / "trajectory.jsonl"

            if mode == "valid" and disconnect_slot is None and len(connect_slots) == 2:
                assert game.returncode == 0, f"{name}: game failed: {game_out}"
                assert results_exists and replay_exists and not failure_exists
                results = json.loads((work / "results.json").read_text())
                replay = json.loads((work / "replay.json").read_text())
                assert results["reason"] == "complete"
                assert results["ending"] == "turn_limit"
                assert results["turns"] == 2
                assert isinstance(results["scores"], list) and len(results["scores"]) == 2
                if use_llm:
                    llm_seats = sum(1 for s in scripted_seats if not s)
                    expected_calls = 2 * llm_seats
                    assert len(calls) == expected_calls, f"expected {expected_calls} model calls, got {len(calls)}"
                    order_events = [e for e in replay["events"] if e.get("k") == "order"]
                    assert len(order_events) == 4
                    llm_orders = [e for e in order_events if e["source"] == "llm"]
                    assert len(llm_orders) == expected_calls
                    traj_rows = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
                    dec_rows = [r for r in traj_rows if r["event_type"] == "decision"]
                    assert len(dec_rows) == 4
                    llm_dec_rows = [r for r in dec_rows if not scripted_seats[int(r["seat"])]]
                    assert len(llm_dec_rows) == expected_calls
                    assert all(r["action_status"] == "accepted" for r in llm_dec_rows), llm_dec_rows
                    assert llm_dec_rows[0]["observation"]["native_execution"]["ancillary_normalized"] == {
                        "hunch_clipped": True,
                        "notes_clipped": True,
                    }
                    assert "H" * 110 in llm_dec_rows[0]["attempts"][0]["response"]
                    assert "N" * 265 in llm_dec_rows[0]["attempts"][0]["response"]
            else:
                assert game.returncode != 0, f"{name}: expected non-zero exit, got 0\n{game_out}"
                assert not results_exists, f"{name}: results.json must NOT be written on failure"
                assert not replay_exists, f"{name}: replay.json must NOT be written on failure"
                if disconnect_slot is not None or len(connect_slots) < 2:
                    expected_kind = "player_error"
                    expected_cause = "player_error"
                elif mode == "ambiguous":
                    expected_kind = "player_error"
                    expected_cause = "invalid_action"
                else:
                    expected_kind = "episode_inconclusive"
                    expected_cause = "infra_error"
                if expected_seat is not None and expected_seat < 0:
                    assert not failure_exists, f"{name}: must not invent a seat in player_failure.json"
                else:
                    assert failure_exists, f"{name}: player_failure.json missing\n{game_out}"
                    failure = json.loads((work / "player_failure.json").read_text())
                    assert set(failure.keys()) == {"failed_policy_index", "message", "kind"}, f"{name}: {failure}"
                    assert failure["kind"] == expected_kind, f"{name}: {failure}"
                    assert isinstance(failure["failed_policy_index"], int) and failure["failed_policy_index"] >= 0, f"{name}: {failure}"
                    assert isinstance(failure["message"], str) and 1 <= len(failure["message"]) <= 2000, f"{name}: {failure}"
                    if expected_seat is not None:
                        assert failure["failed_policy_index"] == expected_seat, f"{name}: expected seat {expected_seat}, got {failure}"
                if use_llm and disconnect_slot is None and len(connect_slots) == 2:
                    llm_seats = sum(1 for s in scripted_seats if not s)
                    assert len(calls) == llm_seats, f"{name}: expected {llm_seats} calls (no retries), got {len(calls)}"
                log_path = work / "game.log"
                assert log_path.exists(), f"{name}: game.log missing"
                diag = json.loads(log_path.read_text().strip().splitlines()[-1])
                assert diag["cause"] == expected_cause, f"{name}: {diag}"
                if expected_seat is not None:
                    assert diag["seat"] == expected_seat, f"{name}: expected diag seat {expected_seat}, got {diag}"
                if traj_path.exists():
                    traj_rows = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
                    ep = [r for r in traj_rows if r["event_type"] == "episode"][-1]
                    assert ep["status"] == "failed", ep
                    assert ep["participant_outcomes"] is None, ep
            print(f"PASS round-trip scenario: {name}")
        finally:
            if srv is not None:
                srv.shutdown()
            for proc in [game, *players]:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--player", type=Path, required=True)
    args = parser.parse_args()

    run_scenario(args.game, args.player, "valid_bare_and_fenced_clipped", "valid", use_llm=True)
    run_scenario(
        args.game,
        args.player,
        "cert_mix_model_vs_caretaker",
        "valid",
        scripted_seats=("", "caretaker"),
        use_llm=True,
    )
    run_scenario(args.game, args.player, "invalid_ambiguous_object", "ambiguous", use_llm=True, expected_seat=0)
    run_scenario(args.game, args.player, "provider_http_503", "provider_503", scripted_seats=("", "caretaker"), use_llm=True, expected_seat=0)
    run_scenario(args.game, args.player, "provider_http_500", "provider_500", use_llm=True, expected_seat=0)
    run_scenario(args.game, args.player, "missing_credentials", "missing_creds", use_llm=False, expected_seat=0)
    run_scenario(
        args.game,
        args.player,
        "startup_zero_seats_missing",
        "valid",
        use_llm=False,
        connect_slots=(),
        connect_timeout_s=1,
        expected_seat=-1,
    )
    run_scenario(
        args.game,
        args.player,
        "startup_one_seat_missing",
        "valid",
        scripted_seats=("caretaker", "stubborn"),
        use_llm=False,
        connect_slots=(0,),
        connect_timeout_s=1,
        expected_seat=1,
    )
    run_scenario(
        args.game,
        args.player,
        "midgame_model_seat_disconnect",
        "valid",
        scripted_seats=("", "caretaker"),
        use_llm=True,
        disconnect_slot=0,
        disconnect_on_call=2,
        expected_seat=0,
    )
    run_scenario(
        args.game,
        args.player,
        "midgame_between_turns_disconnect",
        "valid",
        scripted_seats=("", "caretaker"),
        use_llm=True,
        min_turn_s=1,
        disconnect_slot=1,
        disconnect_on_call=1,
        disconnect_delay_s=0.35,
        expected_seat=1,
    )
    run_scenario(
        args.game,
        args.player,
        "scripted_offline_cert",
        "valid",
        scripted_seats=("caretaker", "stubborn"),
        use_llm=False,
    )
    print("ALL host+player round-trip scenarios PASS")


if __name__ == "__main__":
    main()
