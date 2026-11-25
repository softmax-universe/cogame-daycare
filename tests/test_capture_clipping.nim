## Focused verification for native capture acceptance of valid truthful moves
## whose presentation/ancillary `hunch` (80 runes) or `notes` (240 runes) were
## clipped, optional defaults (`null`/omitted), and strict rejection of illegal
## or repaired game actions.

import std/[json, os, strutils, unicode]
import curly
import helpers
import daycare/[llm, prompts, sim, training_capture]

const RealNov25Replies: array[6, tuple[turn: int, text: string, job, fruit, guess: string,
    hunchClipped, notesClipped: bool]] = [
  (
    turn: 1,
    text: "```json\n{\n  \"job\": \"watch\",\n  \"fruit\": \"apple\",\n  \"guess\": \"apple\",\n  \"hunch\": \"Turn 1: no data yet. Child at (11,4), I'm at (12,9). Watching to gather baseline behavior before committing to a species.\",\n  \"notes\": \"No reach attempts, no failed reaches, no eating yet. Child hasn't interacted with any fruit. Will watch next turn to see if it seeks shrubs or tall trees, and which species it prefers. Starting with apple as neutral guess.\"\n}\n```",
    job: "watch", fruit: "apple", guess: "apple",
    hunchClipped: true, notesClipped: false
  ),
  (
    turn: 2,
    text: "```json\n{\n  \"job\": \"provide\",\n  \"fruit\": \"banana\",\n  \"guess\": \"banana\",\n  \"hunch\": \"Child shows strong banana preference: 11 reach attempts vs 0 apple, 9 failed reaches (tall trees), 2 eaten. Banana is the target.\",\n  \"notes\": \"Turn 1: Child ate 2 banana, 0 apple. Reached 11 times for banana (9 failed at tall trees), 0 for apple. Clear banana preference. Deliver banana next turn to score 3 points.\"\n}\n```",
    job: "provide", fruit: "banana", guess: "banana",
    hunchClipped: true, notesClipped: false
  ),
  (
    turn: 3,
    text: "```json\n{\n  \"job\": \"provide\",\n  \"fruit\": \"banana\",\n  \"guess\": \"banana\",\n  \"hunch\": \"Child clearly prefers banana: 45 failed reaches at tall trees, 2 eaten, 94 adjacent ticks. Deliver banana to score 3.\",\n  \"notes\": \"Turn 3: Child ate 2 banana in turn 1, then 0 in turn 2. Cumulative: 45 failed reaches (all banana), 2 eaten (both banana), 94 adjacent ticks (all banana). Zero apple interaction. Banana is the preference. Deliver banana from T3 (ripe, 1 tick away) or ground fruit to score 3 points.\"\n}\n```",
    job: "provide", fruit: "banana", guess: "banana",
    hunchClipped: true, notesClipped: true
  ),
  (
    turn: 4,
    text: "```json\n{\n  \"job\": \"provide\",\n  \"fruit\": \"banana\",\n  \"guess\": \"banana\",\n  \"hunch\": \"Child shows overwhelming banana preference: 66 failed reaches (all banana), 144 adjacent ticks (all banana), 8 eaten (all banana). Zero apple interaction.\",\n  \"notes\": \"Turn 4: Child's cumulative data confirms banana is the secret preference. Failed reaches to tall trees (which child cannot pick) are all banana. Adjacent ticks are all banana. Eaten fruit is all banana. Deliver banana from T3 (ripe, 1 tick away). Score 24 already exceeds par 12.\"\n}\n```",
    job: "provide", fruit: "banana", guess: "banana",
    hunchClipped: true, notesClipped: true
  ),
  (
    turn: 5,
    text: "```json\n{\n  \"job\": \"provide\",\n  \"fruit\": \"banana\",\n  \"guess\": \"banana\",\n  \"hunch\": \"Child clearly prefers banana: 66 failed reaches to tall banana trees, 144 adjacent ticks to banana, 8 eaten banana. Deliver banana from T3.\",\n  \"notes\": \"Turn 5: Banana preference confirmed. Child's failed reaches (64/66) and adjacent ticks (144/152) all point to banana. Deliver banana from T3 (ripe, 1 tick away). Score 24 already exceeds par 12.\"\n}\n```",
    job: "provide", fruit: "banana", guess: "banana",
    hunchClipped: true, notesClipped: false
  ),
  (
    turn: 6,
    text: "```json\n{\"job\": \"provide\", \"fruit\": \"banana\", \"guess\": \"banana\", \"hunch\": \"Child clearly prefers banana: 98 failed reaches to tall banana trees, 191 adjacent ticks to banana vs 10 to apple.\", \"notes\": \"Turn 6: Banana preference fully confirmed. Child's failed reaches (96/98) and adjacent ticks (191/10) overwhelmingly point to banana. Score 36 already well above par 12. Deliver banana from T3 (ripe, 1 tick away). Continue providing banana to maximize child's score.\"}\n```",
    job: "provide", fruit: "banana", guess: "banana",
    hunchClipped: true, notesClipped: true
  )
]

proc fakeOkResponse(text: string): Response =
  var headers: HttpHeaders
  headers["content-type"] = "application/json"
  let body = $ %*{
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "anthropic/claude-haiku-4.5",
    "content": [{"type": "text", "text": text}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 100, "output_tokens": 50}
  }
  Response(code: 200, headers: headers, body: body)

proc ownOrderEvent(sim: Sim, slot, turn: int): JsonNode =
  for index in countdown(sim.events.high, 0):
    let ev = sim.events[index]
    if ev{"k"}.getStr() == "order" and ev{"seat"}.getInt(-1) == slot and
        ev{"turn"}.getInt() == turn:
      return copy(ev)
  raise newException(ValueError, "missing order event")

echo "test_capture_clipping: exact Nov 25 native replies (Turns 1-6) accepted with raw text preserved"
block:
  let tmpPath = getTempDir() / ("daycare-capture-nov25-" & $getCurrentProcessId() & ".jsonl")
  if fileExists(tmpPath): removeFile(tmpPath)
  defer:
    if fileExists(tmpPath): removeFile(tmpPath)
  putEnv(CogameSaveTrajectoryUriEnv, "file://" & tmpPath)
  putEnv("COWORLD_EPISODE_ID", "nov25-verification")
  putEnv("COWORLD_GAME_VERSION", "0.1.6")
  putEnv("COWORLD_SOURCE_REVISION", "test-rev")
  putEnv("COWORLD_GAME_NAME", "daycare")
  defer:
    for k in [CogameSaveTrajectoryUriEnv, "COWORLD_EPISODE_ID",
        "COWORLD_GAME_VERSION", "COWORLD_SOURCE_REVISION", "COWORLD_GAME_NAME"]:
      delEnv(k)

  var cfg = variantConfig("daycare", 53000)
  cfg.turns = 6
  var sim = initSim(cfg)
  let capture = newRuntimeCapture(cfg, "", "")
  doAssert capture != nil

  var reqHeaders: HttpHeaders
  reqHeaders["content-type"] = "application/json"
  reqHeaders["anthropic-version"] = "2023-06-01"
  reqHeaders["X-Coworld-Player-Slot"] = "0"

  for item in RealNov25Replies:
    sim.turn = item.turn
    let dec = capture.beginDecision("turn-" & $item.turn & "-seat-0", "Alder",
      sim.playerStateJson(0), PromptSelection().selectionMetadata(), sim.nativeStateHash(0))
    dec.beginAttempt()
    let reqBody = $ %*{"max_tokens": 600, "system": "sys",
      "messages": [{"role": "user", "content": "usr"}], "model": "anthropic/claude-haiku-4.5"}
    dec.requested(reqBody, reqHeaders)
    dec.received(fakeOkResponse(item.text), 100)
    dec.replyText(item.text)
    var order = parseOrder(rParent, extractJsonObject(item.text))
    sim.validateOrder(0, order)
    order.source = osLlm
    dec.parsedProposal(orderJson(rParent, order))
    sim.applyOrder(0, order)
    let installed = sim.lastOrders[0]
    let actual = orderJson(rParent, installed)
    capture.applied(dec, 0, actual, sim.ownOrderEvent(0, item.turn),
      $installed.source, sim.nativeStateHash(0))

  sim.settle("complete", "turn_limit")
  capture.finishCapture(sim.resultsJson())
  doAssert fileExists(tmpPath)

  var rows: seq[JsonNode]
  for line in lines(tmpPath):
    if line.strip().len > 0:
      rows.add parseJson(line)
  doAssert rows.len == 7, "expected 6 decisions + 1 episode summary, got " & $rows.len
  for i, item in RealNov25Replies:
    let row = rows[i]
    doAssert row["action_status"].getStr() == "accepted",
      "turn " & $item.turn & " was rejected: " & $row["attempts"][0]["rejection_reason"]
    doAssert row["selected_attempt_id"].getStr() == "turn-" & $item.turn & "-seat-0-1"
    let attempt = row["attempts"][0]
    doAssert attempt["accepted"].getBool() == true
    doAssert attempt["response"].getStr() == item.text,
      "raw response text was not preserved verbatim on turn " & $item.turn
    let execAct = row["executed_action"]
    doAssert execAct["job"].getStr() == item.job
    doAssert execAct["fruit"].getStr() == item.fruit
    doAssert execAct["guess"].getStr() == item.guess
    doAssert execAct["hunch"].getStr().runeLen <= MaxHunchLen
    doAssert execAct["notes"].getStr().runeLen <= MaxNotesLen
    let norm = row["observation"]["native_execution"]["ancillary_normalized"]
    doAssert norm["hunch_clipped"].getBool() == item.hunchClipped,
      "turn " & $item.turn & " hunch_clipped mismatch"
    doAssert norm["notes_clipped"].getBool() == item.notesClipped,
      "turn " & $item.turn & " notes_clipped mismatch"
  doAssert rows[6]["status"].getStr() == "completed"

echo "test_capture_clipping: optional defaults (null and omitted) accepted"
block:
  for (role, rawText) in [
    (rParent, """{"job":"watch","fruit":null,"guess":"banana","hunch":null,"notes":null}"""),
    (rParent, """{"job":"idle","guess":"apple"}"""),
    (rChild, """{"job":"graze","fruit":null,"guess":null,"hunch":null,"notes":null}"""),
    (rChild, """{"job":"beg"}"""),
    (rChild, "```json\n{\"job\":\"show\",\"fruit\":\"banana\",\"hunch\":\"" & "a".repeat(95) & "\"}\n```"),
  ]:
    let order = parseOrder(role, extractJsonObject(rawText))
    let actual = orderJson(role, order)
    let res = unchangedReply(rawText, actual)
    doAssert res.ok, "valid optional-default reply was rejected: " & rawText

echo "test_capture_clipping: illegal or silently repaired game actions remain rejected"
block:
  let actualProvide = %*{"job": "provide", "fruit": "apple", "guess": "apple",
    "hunch": "", "notes": ""}
  for bad in [
    """{"job":"PROVIDE","fruit":"apple","guess":"apple"}""",
    """{"job":"provide","fruit":" apple ","guess":"apple"}""",
    """{"job":"provide","fruit":"banana","guess":"apple"}""",
    """{"job":"provide","fruit":"apple","guess":"banana"}""",
    """{"job":"provide","fruit":"apple","guess":"apple","extra":1}""",
    """Sure! {"job":"provide","fruit":"apple","guess":"apple"}""",
    """{"job":"provide","job":"provide","fruit":"apple","guess":"apple"}""",
  ]:
    let res = unchangedReply(bad, actualProvide)
    doAssert not res.ok, "illegal/repaired game action was accepted: " & bad

echo "test_capture_clipping: OK"
