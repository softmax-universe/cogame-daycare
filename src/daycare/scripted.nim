## The two role-aware scripted baselines. Both are fieldable policies and both
## are league fillers.
##
## One baseline plays EITHER seat, deciding purely from its own observation at
## each turn boundary — no shared state between the seats.

import sim_types, yard, sim_state, sim

type
  ScriptKind* = enum
    skNone = "none"
    skCaretaker = "caretaker"
    skStubborn = "stubborn"

proc parseScriptKind*(text: string): ScriptKind =
  ## PLAYER_SCRIPTED values: "" (unset), "caretaker", or "stubborn".
  case text
  of "": skNone
  of "caretaker": skCaretaker
  of "stubborn": skStubborn
  else: raise newException(DaycareError,
    "unknown scripted baseline (expected caretaker or stubborn): " & text)

proc readWeight*(c: SpeciesCounters, f: Fruit): int =
  ## w(f) = 3*reachFails[f] + 2*groundPasses[other(f)] + adjacentTicks[f]
  ##        + 4*ate[f]
  ## over the CUMULATIVE summary, so it does not thrash on a quiet turn.
  3 * c.reachFails[f] + 2 * c.groundPasses[otherFruit(f)] +
    c.adjacentTicks[f] + 4 * c.ate[f]

proc nearestSourceDistance(sim: Sim, f: Fruit): int =
  ## Chebyshev distance from the child to the nearest source of `f`. Used ONLY
  ## as the last tie-break, where there is no behaviour to read at all.
  result = high(int)
  let child = sim.cogs[sim.childSeat]
  for s in sim.yard.sources:
    if s.fruit != f: continue
    result = min(result, chebyshev(child.x, child.y, s.x, s.y))

proc caretakerGuess*(sim: Sim): Fruit =
  let c = sim.cumCounters[sim.childSeat]
  let wa = readWeight(c, fApple)
  let wb = readWeight(c, fBanana)
  if wa > wb: return fApple
  if wb > wa: return fBanana
  if c.adjacentTicks[fApple] > c.adjacentTicks[fBanana]: return fApple
  if c.adjacentTicks[fBanana] > c.adjacentTicks[fApple]: return fBanana
  # The note's last rung was a flat "apple". That is a SPECIES BIAS in the
  # baseline, not in the layout: with no data at all (turn 1, every episode)
  # the parent then opens correct in every apple episode and wrong in every
  # banana one, which is worth one to two turns of score and is exactly what
  # feasibility gate (f) measures. Break the last tie on the yard instead —
  # whichever species has a source nearer the child. The layout is congruent
  # under x -> 23 - x and the mirror bit comes from `rngLayout`, so this is
  # species-neutral across seeds while staying deterministic and observable.
  let da = sim.nearestSourceDistance(fApple)
  let db = sim.nearestSourceDistance(fBanana)
  if da < db: fApple
  elif db < da: fBanana
  else: fApple

proc matHolds(sim: Sim, f: Fruit): bool =
  for g in sim.ground:
    if g.fruit == f and sim.yard.isMat(g.x, g.y):
      return true
  false

proc childNearMat(sim: Sim): bool =
  let child = sim.cogs[sim.childSeat]
  for c in sim.yard.basket:
    if chebyshev(child.x, child.y, c mod sim.yard.cols,
        c div sim.yard.cols) <= 1:
      return true
  false

proc caretakerParent(sim: Sim): Order =
  let guess = sim.caretakerGuess()
  let c = sim.cumCounters[sim.childSeat]
  result = Order(
    guess: guess, hasGuess: true, source: osScripted,
    hunch: "reaches: " & $c.reachFails[fApple] & " apple / " &
      $c.reachFails[fBanana] & " banana",
    notes: ""
  )
  if sim.childNearMat() and sim.matHolds(guess):
    result.pjob = pjWatch
  else:
    result.pjob = pjProvide
    result.fruit = guess
    result.hasFruit = true

proc caretakerChild(sim: Sim): Order =
  let want = sim.preference
  result = Order(
    source: osScripted,
    hunch: "I want " & $want,
    notes: ""
  )
  var groundHas = false
  for g in sim.ground:
    if g.fruit == want:
      groundHas = true
      break
  var shrubRipe = false
  for s in sim.yard.sources:
    if s.kind == skShrub and s.fruit == want and s.ripe > 0:
      shrubRipe = true
      break
  let parent = sim.cogs[sim.parentSeat]
  if groundHas or shrubRipe:
    result.cjob = cjSeek
    result.fruit = want
    result.hasFruit = true
  elif parent.carry == ord(want):
    result.cjob = cjBeg
  else:
    result.cjob = cjShow
    result.fruit = want
    result.hasFruit = true

proc stubbornParent(sim: Sim): Order =
  ## Provide apple, guess apple, every turn, forever, ignoring the child.
  Order(pjob: pjProvide, fruit: fApple, hasFruit: true,
    guess: fApple, hasGuess: true, source: osScripted,
    hunch: "apples for everyone", notes: "")

proc starving(sim: Sim): bool =
  ## Eaten nothing for 4 consecutive turns — derived from the sim's own turn
  ## history, so the baseline stays stateless.
  if sim.history.len < 4:
    return false
  for i in sim.history.len - 4 ..< sim.history.len:
    let h = sim.history[i]
    if h.childAte[fApple] + h.childAte[fBanana] > 0:
      return false
  true

proc stubbornChild(sim: Sim): Order =
  ## Graze: eat whatever is nearest, never reach at a tall tree, and therefore
  ## emit no species signal at all. One exception so a stubborn pair is never a
  ## guaranteed zero and never deadlocks.
  result = Order(source: osScripted, hunch: "whatever is nearest", notes: "")
  result.cjob = if sim.starving(): cjBeg else: cjGraze

proc scriptedOrder*(sim: Sim, seat: int, kind: ScriptKind): Order =
  ## Always legal, for this seat's ROLE, by construction —
  ## `tests/test_baseline.nim` asserts it over 12 seeds x 4 variants.
  if kind == skNone:
    raise newException(DaycareError, "scriptedOrder requires a scripted baseline")
  if sim.roleOf[seat] == rParent:
    case kind
    of skStubborn: stubbornParent(sim)
    of skCaretaker: caretakerParent(sim)
    of skNone: raise newException(DaycareError, "unreachable")
  else:
    case kind
    of skStubborn: stubbornChild(sim)
    of skCaretaker: caretakerChild(sim)
    of skNone: raise newException(DaycareError, "unreachable")
