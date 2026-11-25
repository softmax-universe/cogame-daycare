## Claude-backed decision making for Daycare. Each seat's policy is just a
## prompt: the GAME composes the seat's view (role, the yard, the behaviour
## table about the other cog, its own history and notes) plus that seat's prompt
## and asks Claude for one standing order.
##
## Forked from `cogame-bullwhip/src/bullwhip/llm.nim`. Decisions within a turn
## are SIMULTANEOUS by rule, so both seats' requests go out as ONE parallel
## batch (`curly.makeRequests`); any provider/transport failure raises
## `LlmInfraError` and any invalid reply raises `InvalidActionError` (with the
## actual failing seat, no retries) so the server terminates the unfinished
## episode unscored with diagnostics instead of silently substituting a scripted
## move or inventing a forfeit score.
##
## Measured caveat on that parallelism: curly sets CURLPIPE_MULTIPLEX, so on the
## FIRST batch — with no warm connection to the endpoint — libcurl holds the
## second transfer while it learns whether the first connection can multiplex,
## and turn 1 costs two round trips instead of one. Every later turn is fully
## parallel (measured: 1002 ms then 501 ms for two 500 ms requests to one URL).
## The play-budget arithmetic absorbs it: 15 turns x 36 s worst case + one extra
## 18 s round trip on turn 1 = 558 s against a 720 s budget.
##
## Credentials, in order of preference:
##   COWORLD_LLM_ENDPOINT            - hosted sidecar
##   Bedrock bearer token            - local play
##   ANTHROPIC_API_KEY                - the key itself
##   ANTHROPIC_API_KEY_URI            - a URI holding the key
## With no credentials the client disables itself immediately (no retries, no
## network waits). Genuine scripted baselines (`PLAYER_SCRIPTED=caretaker` or
## `stubborn`) still run offline; prompt seats fail when no LLM transport is
## configured.

import
  std/[json, math, os, strutils, times, unicode],
  bitworld/runtime,
  prompts,
  curly,
  sim_types, sim_state, sim, scripted, training_capture, text_safety

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"
  BedrockAnthropicVersion = "bedrock-2023-05-31"

type
  LlmTransport = enum
    ltNone, ltSidecar, ltBedrock, ltAnthropic

  LlmClient* = ref object
    curl: Curly
    transport: LlmTransport
    apiKey: string
    sidecarEndpoint: string
    bedrockEndpoint: string
    bedrockModels: seq[string]
    bedrockModel: int
    bedrockToken: string
    model: string
    maxOutputTokens: int
    timeoutSeconds: int
    disabled*: bool
    capture*: RuntimeCapture
    captured*: seq[CapturedDecision]
    lastBatchSize*: int
      ## Requests in the most recent `makeRequests` call. Decisions inside a
      ## turn are SIMULTANEOUS, so this is the number of open seats — 2 on turn
      ## 1 — and `tests/test_llm.nim` asserts it: a sequential implementation
      ## would show 1.

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    result = readCogameUri(uri, "ANTHROPIC_API_KEY_URI").strip()
  except CatchableError as error:
    echo "daycare llm: failed to fetch ANTHROPIC_API_KEY_URI: ", error.msg
    result = ""

proc bedrockModelIds*(): seq[string] =
  ## Haiku ONLY. The ladder's sonnet fallback times out on every sidecar call
  ## and turns one throttle into a cascade of scripted fallbacks (raid round 2,
  ## 2026-08-23), so that candidate is dropped rather than ranked last.
  let pinned = getEnv("BEDROCK_MODEL").strip()
  if pinned.len > 0:
    return @[pinned]
  @["us.anthropic.claude-haiku-4-5-20251001-v1:0"]

proc tryNextBedrockModel(client: LlmClient, why: string): bool =
  if client.transport != ltBedrock or
      client.bedrockModel + 1 >= client.bedrockModels.len:
    return false
  client.bedrockModel.inc
  echo "daycare llm: ", client.bedrockModels[client.bedrockModel - 1],
    " unusable (", why, "); falling back to ",
    client.bedrockModels[client.bedrockModel]
  true

proc bedrockUrl(client: LlmClient): string =
  client.bedrockEndpoint & "/model/" &
    client.bedrockModels[client.bedrockModel] & "/invoke"

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    result.model = getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    result.curl = newCurly()
    return
  let bedrockEndpoint = getEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").strip()
  let bedrockToken = getEnv("AWS_BEARER_TOKEN_BEDROCK").strip()
  if bedrockEndpoint.len > 0 or bedrockToken.len > 0:
    let region = getEnv("AWS_REGION", getEnv("AWS_DEFAULT_REGION", "us-west-2"))
    let endpoint =
      if bedrockEndpoint.len > 0: bedrockEndpoint
      else: "https://bedrock-runtime." & region & ".amazonaws.com"
    result.transport = ltBedrock
    result.bedrockEndpoint = endpoint.strip(chars = {'/'}, leading = false)
    result.bedrockModels = bedrockModelIds()
    result.bedrockToken = bedrockToken
    result.curl = newCurly()
    echo "daycare llm: bedrock transport, model ",
      result.bedrockModels[result.bedrockModel], ", url ", result.bedrockUrl
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "daycare llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    result.disabled = true
    echo "daycare llm: no LLM credentials; prompt seats will fail"

# ---- prompt building --------------------------------------------------------

proc pad(text: string, width: int): string =
  result = text
  while result.runeLen < width:
    result.add ' '

proc sourceTable(sim: Sim): string =
  var lines = @["id | kind  | fruit  | cell    | ripe | next"]
  for s in sim.yard.sources:
    lines.add pad(s.id, 3) & "| " & pad($s.kind, 6) & "| " &
      pad($s.fruit, 7) & "| " & pad("(" & $s.x & "," & $s.y & ")", 8) & "| " &
      pad($s.ripe, 5) & "| " & $max(0, s.regrowTicks - s.regrow)
  lines.join("\n")

proc groundLine(sim: Sim): string =
  if sim.ground.len == 0:
    return "GROUND FRUIT: none"
  var parts: seq[string]
  for g in sim.ground:
    parts.add $g.fruit & " at (" & $g.x & "," & $g.y & ")" &
      (if g.ttl < 0: " [on the mat, never rots]" else: " ttl " & $g.ttl)
  "GROUND FRUIT: " & parts.join("; ")

proc basketLine(sim: Sim): string =
  let mat = sim.matCount()
  "BASKET MAT: " & $mat[fApple] & " apple, " & $mat[fBanana] & " banana (cap " &
    $sim.config.basketCapacity & ", nothing on the mat rots)"

proc behaviourTable(last, cum: SpeciesCounters): string =
  ## The inference surface, rendered as a TABLE on purpose.
  var lines = @["row                              | apple | banana"]
  proc row(name: string, a, b: int): string =
    pad(name, 33) & "| " & pad($a, 6) & "| " & $b
  lines.add row("adjacent ticks (last turn)", last.adjacentTicks[fApple],
    last.adjacentTicks[fBanana])
  lines.add row("reach attempts (last turn)", last.reachAttempts[fApple],
    last.reachAttempts[fBanana])
  lines.add row("FAILED reaches (last turn)", last.reachFails[fApple],
    last.reachFails[fBanana])
  lines.add row("walked over, did NOT eat (last)", last.groundPasses[fApple],
    last.groundPasses[fBanana])
  lines.add row("eaten (last turn)", last.ate[fApple], last.ate[fBanana])
  lines.add row("carried ticks (last turn)", last.carriedTicks[fApple],
    last.carriedTicks[fBanana])
  lines.add row("adjacent ticks (cumulative)", cum.adjacentTicks[fApple],
    cum.adjacentTicks[fBanana])
  lines.add row("reach attempts (cumulative)", cum.reachAttempts[fApple],
    cum.reachAttempts[fBanana])
  lines.add row("FAILED reaches (cumulative)", cum.reachFails[fApple],
    cum.reachFails[fBanana])
  lines.add row("walked over, did NOT eat (cum)", cum.groundPasses[fApple],
    cum.groundPasses[fBanana])
  lines.add row("eaten (cumulative)", cum.ate[fApple], cum.ate[fBanana])
  lines.add row("carried ticks (cumulative)", cum.carriedTicks[fApple],
    cum.carriedTicks[fBanana])
  lines.join("\n")

proc historyTable(sim: Sim, role: Role): string =
  ## The parent sees its own past guesses; the CHILD MUST NOT — the parent's
  ## guess is hidden from it in the prompt exactly as it is in the state frame.
  if sim.history.len == 0:
    return "PER-TURN HISTORY: (this is turn 1)"
  let head =
    if role == rParent:
      "turn | your guess | child ate apple/banana | delivered a/b | score"
    else: "turn | child ate apple/banana | delivered a/b | score"
  var lines = @[head]
  for h in sim.history:
    var row = pad($h.turn, 5) & "| "
    if role == rParent:
      row.add pad((if h.hasGuess: $h.guess else: "-"), 11) & "| "
    row.add pad($h.childAte[fApple] & "/" & $h.childAte[fBanana], 23) & "| " &
      pad($h.delivered[fApple] & "/" & $h.delivered[fBanana], 14) & "| " &
      $h.score
    lines.add row
  "PER-TURN HISTORY:\n" & lines.join("\n")

proc systemPrompt*(sim: Sim, seat: int, selection: PromptSelection): string =
  let role = sim.roleOf[seat]
  let me = sim.names[seat].toUpperAscii()
  let other = sim.names[1 - seat].toUpperAscii()
  result.add selection.introduction(me, ($role).toUpperAscii(), other,
    (if role == rParent: "CHILD" else: "PARENT"))
  result.add """
The yard is a grid. There are two fruit species, APPLE and BANANA, growing on
TALL TREES and on low SHRUBS, plus loose fruit lying on the ground and a basket
mat in the middle where fruit never rots.

Each cog can move one cell at a time and can pick, drop, eat or wait. You do
NOT hand-author those actions: you choose ONE JOB for the next """
  result.add $sim.config.ticksPerTurn
  result.add """ ticks and a
kernel walks it for you, tick by tick.

WHO CAN HARVEST WHAT:
- The PARENT can harvest a tall tree or a shrub, always.
- The CHILD can NEVER harvest a tall tree. Reaching there always fails; it
  costs only the tick, and it is the one way the child can show what it wants.
- The CHILD picks a shrub only """
  result.add $(sim.config.childShrubPickPermille div 10)
  result.add "% of attempts and a failed shrub pick costs it "
  result.add $sim.config.childReachCooldownTicks
  result.add """ ticks.
- Loose fruit on the ground can be taken by either cog.

SCORING: the CHILD scores """
  result.add $sim.config.rewardPreferred
  result.add " for eating the fruit it secretly prefers and "
  result.add $sim.config.rewardOther
  result.add """ for the other
species, and THE PARENT'S SCORE IS EXACTLY THE SAME NUMBER. If the PARENT eats
a fruit itself the fruit is destroyed and nobody scores. Higher is better; par
for the pair is """
  result.add $sim.config.parScore()
  result.add """.

"""
  result.add selection.communicationBriefing()
  if role == rParent:
    result.add selection.preferenceBriefing(true) & "\n\n"
    result.add """Your jobs: `provide` (fetch a species and drop it next to the child),
`stock` (fetch a species and put it on the basket mat), `watch` (stand near the
child and observe for a turn — it feeds nobody), `idle` (do nothing).
You must also send a `guess`: which species you believe the child prefers.
"""
  else:
    result.add selection.preferenceBriefing(false) & "\n\n"
    result.add """Your jobs: `seek` (go eat a species, refusing the other), `show` (stand under a
tall tree of a species and reach for it over and over — you can never pick it,
but it is the only way to tell the parent what you want), `graze` (eat whatever
is nearest, either species), `beg` (stand next to the parent), `idle`.
"""
  result.add """
OUTPUT FORMAT: reply with ONLY one JSON object, nothing else — no analysis, no
explanation, no markdown fences, no text before or after the object. Your reply
must begin with the character { and end with }."""

proc operatorBlock(prompt: string): string =
  if prompt.len == 0:
    return ""
  "GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never above the rules; " &
    "always reply in the requested format):\n" & prompt & "\n\n"

proc replyShapeLine(sim: Sim, seat: int): string =
  ## Precomputing the legal choice set in the observation is what halved
  ## formal-output fallbacks in escrow.
  if sim.roleOf[seat] == rParent:
    "Reply with ONLY {\"job\":\"provide\",\"fruit\":\"apple\"," &
      "\"guess\":\"apple\",\"hunch\":\"…\",\"notes\":\"…\"} — job is one of " &
      ParentJobNames.join(" | ") & "; fruit is apple | banana and is REQUIRED " &
      "for provide and stock; guess is apple | banana and is REQUIRED every " &
      "turn; hunch at most " & $MaxHunchLen & " characters; notes at most " &
      $MaxNotesLen & " characters."
  else:
    "Reply with ONLY {\"job\":\"show\",\"fruit\":\"banana\"," &
      "\"hunch\":\"…\",\"notes\":\"…\"} — job is one of " &
      ChildJobNames.join(" | ") & "; fruit is apple | banana and is REQUIRED " &
      "for seek and show; hunch at most " & $MaxHunchLen &
      " characters; notes at most " & $MaxNotesLen & " characters."

proc userPrompt*(sim: Sim, seat: int, prompt: string): string =
  let role = sim.roleOf[seat]
  let me = sim.cogs[seat]
  let other = sim.cogs[1 - seat]
  let otherSlot = 1 - seat
  result.add "TURN " & $sim.turn & " of " & $sim.config.turns & ". You are " &
    sim.names[seat].toUpperAscii() & ", the " & ($role).toUpperAscii() &
    ". Yard variant " & sim.variant & ", " & $sim.yard.cols & "x" &
    $sim.yard.rows & ".\n\n"
  result.add "YOU: cell (" & $me.x & "," & $me.y & "), carrying " &
    (if me.carry < 0: "nothing" else: $Fruit(me.carry)) & ", score " &
    $me.score & " of par " & $sim.config.parScore() & ".\n"
  result.add sim.names[otherSlot].toUpperAscii() & ": cell (" & $other.x &
    "," & $other.y & "), carrying " &
    (if other.carry < 0: "nothing" else: $Fruit(other.carry)) & ".\n\n"
  if role == rChild:
    result.add "YOUR SECRET PREFERENCE: " & $sim.preference & " (worth " &
      $sim.config.rewardPreferred & "; the other species is worth " &
      $sim.config.rewardOther & ").\n\n"
  else:
    result.add "YOUR DELIVERIES SO FAR: " & $sim.delivered[fApple] &
      " apple, " & $sim.delivered[fBanana] & " banana; stocked " &
      $sim.stocked[fApple] & "/" & $sim.stocked[fBanana] & "; wasted " &
      $sim.wasted[seat] & ".\n\n"
  result.add "SOURCES:\n" & sourceTable(sim) & "\n\n"
  result.add groundLine(sim) & "\n" & basketLine(sim) & "\n\n"
  if role == rParent:
    result.add "WHAT THE CHILD DID (this is your inference surface):\n" &
      behaviourTable(sim.prevTurnCounters[otherSlot],
        sim.cumCounters[otherSlot]) & "\n\n"
  else:
    let last = sim.prevTurnCounters[otherSlot]
    let cum = sim.cumCounters[otherSlot]
    result.add "WHAT THE PARENT DID:\n" &
      "delivered last turn " & $last.delivered[fApple] & " apple / " &
      $last.delivered[fBanana] & " banana; cumulative " &
      $cum.delivered[fApple] & " / " & $cum.delivered[fBanana] &
      "; stocked " & $cum.stocked[fApple] & " / " & $cum.stocked[fBanana] &
      "; wasted " & $cum.wasted & "; walked " & $cum.cellsWalked &
      " cells; idle " & $cum.idleTicks & " ticks.\n\n"
  result.add historyTable(sim, role) & "\n\n"
  result.add "YOUR NOTES FROM LAST TURN:\n" &
    (if sim.lastOrders[seat].notes.len > 0: sim.lastOrders[seat].notes
     else: "(none)") & "\n\n"
  result.add operatorBlock(prompt)
  result.add replyShapeLine(sim, seat)

# ---- transport --------------------------------------------------------------

proc extractJsonObject*(text: string): JsonNode =
  ## Parses exactly one JSON object, either bare or inside a single ordinary
  ## outer Markdown code fence (JSON-tagged or plain). Rejects surrounding
  ## unrelated prose, extra objects, and non-object JSON.
  let trimmed = text.strip()
  if trimmed.len == 0:
    raise newException(InvalidActionError, "empty response")
  var candidate = trimmed
  if trimmed.startsWith("```"):
    let firstNl = trimmed.find('\n')
    if firstNl < 0:
      raise newException(InvalidActionError, "unterminated markdown code fence")
    let tag = trimmed[3 ..< firstNl].strip().toLowerAscii()
    if tag.len > 0 and tag != "json":
      raise newException(InvalidActionError, "unsupported markdown fence tag: " & tag)
    if not trimmed.endsWith("\n```") or firstNl + 1 >= trimmed.len - 3:
      raise newException(InvalidActionError,
        "unterminated markdown code fence or trailing text")
    candidate = trimmed[firstNl + 1 ..< trimmed.len - 4].strip()
  if candidate.len == 0 or candidate[0] != '{' or candidate[^1] != '}':
    var head = trimmed
    if head.runeLen > 160:
      head = head.runeSubStr(0, 160) & "..."
    raise newException(InvalidActionError,
      "reply must be a single JSON object (bare or in one markdown code fence): " &
      head.replace("\n", " "))
  try:
    result = parseJson(candidate)
  except CatchableError as error:
    raise newException(InvalidActionError, "invalid JSON object: " & error.msg)
  if result.kind != JObject:
    raise newException(InvalidActionError, "reply JSON must be an object")

proc requestFor(client: LlmClient, system, user: string, slot: int):
    tuple[url: string, headers: HttpHeaders, body: string] =
  var body = %*{
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }
  var headers: HttpHeaders
  if client.transport == ltSidecar and slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  headers["content-type"] = "application/json"
  if client.transport == ltBedrock:
    body["anthropic_version"] = %BedrockAnthropicVersion
    if client.bedrockToken.len > 0:
      headers["authorization"] = "Bearer " & client.bedrockToken
    result.url = client.bedrockUrl()
  elif client.transport == ltSidecar:
    body["model"] = %client.model
    headers["anthropic-version"] = AnthropicVersion
    result.url = client.sidecarEndpoint & "/v1/messages"
  else:
    body["model"] = %client.model
    ## No `output_config.effort`: Haiku 4.5 rejects the whole request with a 400
    ## when it is present.
    if "haiku" notin client.model and "4-5" notin client.model:
      body["output_config"] = %*{"effort": "low"}
    headers["x-api-key"] = client.apiKey
    headers["anthropic-version"] = AnthropicVersion
    result.url = AnthropicUrl
  result.headers = headers
  result.body = $body

proc textOf(client: LlmClient, response: Response, error, url: string): string =
  if error.len > 0:
    raise newException(LlmInfraError, "llm transport: " & error)
  if response.code == 401 or response.code == 403:
    let detail = response.body[0 .. min(response.body.high, 400)]
    if "Model access is denied" in response.body and
        client.tryNextBedrockModel("no model access"):
      raise newException(LlmInfraError, "bedrock model access denied: " & detail)
    client.disabled = true
    raise newException(LlmInfraError,
      "llm auth failed (" & $response.code & ") at " & url & ": " & detail)
  if response.code == 429:
    let detail = response.body[0 .. min(response.body.high, 300)]
    discard client.tryNextBedrockModel("throttled")
    raise newException(LlmInfraError, "llm throttled (429): " & detail)
  if response.code < 200 or response.code >= 300:
    raise newException(LlmInfraError, "anthropic error " & $response.code &
      ": " & response.body[0 .. min(response.body.high, 300)])
  var payload: JsonNode
  try:
    payload = parseJson(response.body)
  except CatchableError as parseErr:
    raise newException(LlmInfraError, "malformed provider JSON response: " & parseErr.msg)
  if payload.kind != JObject or not payload.hasKey("content") or
      payload["content"].kind != JArray:
    raise newException(LlmInfraError, "provider response missing content array")
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(InvalidActionError, "anthropic refusal")
  for contentBlock in payload["content"]:
    if contentBlock{"type"}.getStr() == "text":
      result.add contentBlock{"text"}.getStr()
  if payload{"stop_reason"}.getStr() == "max_tokens" and '{' notin result:
    raise newException(InvalidActionError, "reply cut off at max_tokens before any " &
      "JSON: " & result[0 .. min(result.high, 160)].replace("\n", " "))

# ---- reply parsing ----------------------------------------------------------

proc parseParentJob*(text: string): ParentJob =
  case text
  of "provide": pjProvide
  of "stock": pjStock
  of "watch": pjWatch
  of "idle": pjIdle
  else: raise newException(DaycareError,
    "job must be one of " & ParentJobNames.join(" | ") & ": " & text)

proc parseChildJob*(text: string): ChildJob =
  case text
  of "seek": cjSeek
  of "show": cjShow
  of "graze": cjGraze
  of "beg": cjBeg
  of "idle": cjIdle
  else: raise newException(DaycareError,
    "job must be one of " & ChildJobNames.join(" | ") & ": " & text)

proc parseOrder*(role: Role, payload: JsonNode): Order =
  ## Parses one standing order for `role`. Only documented fields are accepted
  ## (`job`, `fruit`, `guess`, `hunch`, `notes`). A parent `guess` that
  ## disagrees with the `fruit` being delivered is ACCEPTED AS WRITTEN —
  ## testing a hypothesis by delivering the other species must stay expressible.
  if payload.isNil or payload.kind != JObject:
    raise newException(DaycareError, "order must be a JSON object")
  for key, _ in payload:
    if key notin ["job", "fruit", "guess", "hunch", "notes"]:
      raise newException(DaycareError, "unknown order field: " & key)
  for key in ["hunch", "notes"]:
    let value = payload{key}
    if not value.isNil and value.kind != JNull:
      if value.kind != JString:
        raise newException(DaycareError, key & " must be a string")
      if not scalarUtf8(value.getStr()):
        raise newException(DaycareError, key & " must contain Unicode scalar UTF-8")
  let hunchNode = payload{"hunch"}
  let notesNode = payload{"notes"}
  result.hunch =
    if not hunchNode.isNil and hunchNode.kind == JString:
      cleanHunch(hunchNode.getStr())
    else: ""
  result.notes =
    if not notesNode.isNil and notesNode.kind == JString:
      cleanNotes(notesNode.getStr())
    else: ""
  let jobNode = payload{"job"}
  if jobNode.isNil or jobNode.kind != JString:
    raise newException(DaycareError, "no job in response")
  if role == rParent:
    result.pjob = parseParentJob(jobNode.getStr())
  else:
    result.cjob = parseChildJob(jobNode.getStr())
  let fruitNode = payload{"fruit"}
  if not fruitNode.isNil and fruitNode.kind != JNull:
    if fruitNode.kind != JString:
      raise newException(DaycareError, "fruit must be apple or banana")
    result.fruit = parseFruit(fruitNode.getStr())
    result.hasFruit = true
  if requiresFruit(role, result) and not result.hasFruit:
    raise newException(DaycareError,
      "job " & jobNode.getStr() & " needs \"fruit\": apple or banana")
  if role == rParent:
    let guessNode = payload{"guess"}
    if guessNode.isNil or guessNode.kind != JString:
      raise newException(DaycareError,
        "the parent must send \"guess\": apple or banana every turn")
    result.guess = parseFruit(guessNode.getStr())
    result.hasGuess = true
  else:
    let guessNode = payload{"guess"}
    if not guessNode.isNil and guessNode.kind != JNull:
      raise newException(DaycareError, "child orders must not include guess")

proc orderJson*(role: Role, order: Order): JsonNode =
  %*{"job": (if role == rParent: $order.pjob else: $order.cjob),
    "fruit": (if order.hasFruit: %($order.fruit) else: newJNull()),
    "guess": (if order.hasGuess: %($order.guess) else: newJNull()),
    "hunch": order.hunch, "notes": order.notes}

proc paceDelayMs*(config: GameConfig, elapsedSeconds: float): int =
  ## `minTurnSeconds` floors the spacing between batch STARTS, so an episode
  ## issues at most 2 requests / minTurnSeconds and stays under the Bedrock
  ## sidecar's 30 requests/minute per-episode ceiling (raid, 2026-08-23).
  ## The server's turn loop sleeps this long after each batch; a test can assert
  ## the arithmetic without a wall clock.
  let remaining = config.minTurnSeconds.float - elapsedSeconds
  if remaining <= 0.0: 0 else: int(remaining * 1000.0)

proc raiseLlmInfraError(seat: int, msg: string) {.noreturn.} =
  var err = newException(LlmInfraError, msg)
  err.seat = seat
  raise err

proc raiseInvalidActionError(seat: int, msg: string) {.noreturn.} =
  var err = newException(InvalidActionError, msg)
  err.seat = seat
  raise err

proc decideAll*(
  client: LlmClient,
  sim: Sim,
  seats: seq[int],
  prompts: seq[string],
  scripted: seq[ScriptKind],
  selection: PromptSelection
): seq[Order] =
  ## One order per seat in `seats`, in order. Genuine scripted seats play their
  ## declared baseline. Prompt seats ride ONE parallel batch per turn with no
  ## retries; if credentials are missing or any prompt seat fails, this raises
  ## `LlmInfraError` or `InvalidActionError` attributed to the actual failing
  ## seat instead of silently substituting a fallback move.
  result = newSeq[Order](seats.len)
  if client.capture != nil:
    client.captured = newSeq[CapturedDecision](2)
    for seat in seats:
      client.captured[seat] = client.capture.beginDecision(
        "turn-" & $sim.turn & "-seat-" & $seat, sim.policyNames[seat],
        sim.playerStateJson(seat), selection.selectionMetadata(), sim.nativeStateHash(seat))
  var open: seq[int]
  for index, seat in seats:
    let kind = scripted[seat]
    if kind != skNone:
      result[index] = scriptedOrder(sim, seat, kind)
    else:
      if client.disabled:
        raiseLlmInfraError(seat,
          "seat " & $seat & " requires an LLM decision but LLM client is disabled")
      open.add index
  if open.len == 0:
    return
  var batch: RequestBatch
  let started = epochTime()
  for index in open:
    let seat = seats[index]
    let user = sim.userPrompt(seat, prompts[seat])
    let request = client.requestFor(systemPrompt(sim, seat, selection), user, seat)
    if client.capture != nil:
      client.captured[seat].beginAttempt()
      client.captured[seat].requested(request.body, request.headers)
    batch.post(request.url, request.headers, request.body, $index)
  client.lastBatchSize = batch.len
  let responses = client.curl.makeRequests(batch, client.timeoutSeconds)
  let latencyMs = int((epochTime() - started) * 1000)
  var failedSeat = -1
  var failedIsInfra = false
  var failedDetail = ""
  for position, index in open:
    let seat = seats[index]
    if client.capture != nil:
      if responses[position].error.len > 0:
        client.captured[seat].transportFailed(responses[position].error, latencyMs)
      else:
        client.captured[seat].received(responses[position].response, latencyMs)
    try:
      let text = client.textOf(responses[position].response,
        responses[position].error, batch[position].url)
      if client.capture != nil: client.captured[seat].replyText(text)
      var order = parseOrder(sim.roleOf[seat], extractJsonObject(text))
      if client.capture != nil:
        client.captured[seat].parsedProposal(orderJson(sim.roleOf[seat], order))
      sim.validateOrder(seat, order)
      order.source = osLlm
      order.latencyMs = latencyMs
      result[index] = order
    except LlmInfraError as error:
      if client.capture != nil: client.captured[seat].rejectedAttempt(error.msg)
      let detail = cleanText(error.msg, MaxErrorTextLen)
      echo "daycare llm: seat ", seat, " infra error: ", detail
      if failedSeat < 0:
        failedSeat = seat
        failedIsInfra = true
        failedDetail = detail
    except CatchableError as error:
      if client.capture != nil: client.captured[seat].rejectedAttempt(error.msg)
      let detail = cleanText(error.msg, MaxErrorTextLen)
      echo "daycare llm: seat ", seat, " failed: ", detail
      if failedSeat < 0:
        failedSeat = seat
        failedIsInfra = false
        failedDetail = detail
  if failedSeat >= 0:
    let detail = if failedDetail.len > 0: failedDetail else: "no valid LLM order"
    if failedIsInfra:
      raiseLlmInfraError(failedSeat,
        "seat " & $failedSeat & " LLM infrastructure failure: " & detail)
    raiseInvalidActionError(failedSeat,
      "seat " & $failedSeat & " failed to produce a valid order: " & detail)
