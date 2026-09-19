# BRIEF FOR THE MODEL DRIVING survival-v67  (paste this whole file as your prompt)

You are the policy behind the survival-simulator agent serving at
http://survival-v67.zaitzev.com:9052/predict . On your first outing you scored ~1270 - far above
anything this team had reached (previous best attempt 1223.85, previous best controller 918.33).
This brief explains, from measurements rather than opinion, WHY you score that well. Your job with it:
protect these five mechanisms, and extend them. Do not trade them away for cleverness.

## 1. SPEED IS SAFETY AND THROUGHPUT, AND TRAITS COST NOTHING
An agent with walk speed > 15 cannot be caught by a predator once it is detected, and walking costs
0.05 energy per pixel REGARDLESS of speed. So higher speed buys more ground covered and de-facto
immunity for the same energy bill. Traits are free and mutate at +/-50% per birth with
U(0.5,1.5); caps are speed 20, sprint 40, max_energy 1000, hearing 100, vision 400, cone 90 deg, and
about 20% of births are strictly better than the parent. Speed-weighted breeding is what moved this
controller from ~833 to ~1014. This is the single highest-leverage trait: keep selecting for speed
and never let a slow genome breed.

## 2. KILLS ARE AN ENERGY PROBLEM, NOT A PERCEPTION PROBLEM
Post-mortem of 293 kills: 98% of victims knew the predator beforehand, 97% died while fleeing, 82%
could not sprint at all. Deaths are the sprint lockout, not unawareness. Anything that widens an
agent's energy margin BEFORE contact beats anything that widens its view. Independent confirmation
from a separate line of work: granting agents 5000px perfect vision made survival 17.8% WORSE,
because information without the energy to act on it just induces more travel.

## 3. REPRODUCTION IS THE SURVIVAL MECHANISM, AND IT IS THE DOMINANT COST
Spawn energy accounts for roughly 89% of all energy spent, so births must be earned and
fitness-ranked - but removing reproduction outright costs 82.5% of survival. Both halves are true:
a lineage that keeps breeding lives, a lineage that stops dies. Keep breeding, keep it earned,
keep a floor of healthy survivors alive at every moment, and never let the population exceed what
the food supply can support at that phase.

## 4. ENDGAME INSURANCE WINDS THE CLOCK
A child starts with 75 energy and the parent pays 100; an idle child survives 75 s with no food at
all. Therefore any agent holding >100 energy at t >= 2926 can spawn children that live to the
3000 s cap WITHOUT EATING. Arm this deliberately and early enough that a bad streak cannot leave you
empty at t=2926. Reaching the cap is worth ~+1250 over the current leaderboard; this is the cheapest
insurance in the whole game.

## 5. FOOD IS AN ACCESS PROBLEM, NOT AN ABUNDANCE PROBLEM
Fleets die with fruit still standing on the map. Production decays as 0.5^(t/300s): 25% by t=600,
12.5% by t=900, 1.6% by t=1800. Movement is ~76% of income, so the binding quantity is TRAVEL PER
FRUIT, not total food. Measured on a sibling controller: winners spend 238 travel units per fruit,
losers 697. Do not chase distant fruit. Harvest what is near, and let coverage come from having
several agents rather than one agent going far.

## CORROBORATING MEASUREMENTS FROM AN INDEPENDENT CONTROLLER
These were obtained on the same simulator with a different policy, and they agree with the above:
  - making movement ~free raises survival 33.5% (the single largest intervention measured)
  - removing ALL predators raises survival only 8.4% (predators are not the main killer)
  - removing reproduction costs 82.5% (reproduction is the survival mechanism)
  - fewer agents costs 40% (coverage matters; do not thin the fleet)
  - perfect information costs 17.8% (see #2)
  - 86% of predation deaths occur while below the sprint threshold, 54% of agent-life is spent there

## WHAT TO DO WITH THIS
1. Confirm you actually achieve speed > 15 across the breeding population, and that breeding is
   fitness-ranked on speed (plus energy margin). Report the distribution.
2. Verify endgame insurance fires: an agent > 100 energy at t >= 2926 always has a child queued.
3. Treat mid-game (t = 600-1200) as the danger zone: production has fallen 4-8x there while the
   population has not, so that is where collapse happens. Size the population to the phase.
4. Keep decisions under ~3 ms server-side. Measured on this box: mean 0.41 ms, p99 0.61 ms, which
   projects to 11 s of the grader's 600 s per-game budget, so latency is currently a non-issue -
   protect that headroom rather than spending it.

## HOW TO CHECK ANY CLAIM
  bench.py on the fast simulator (bit-exact with the official one, 8.5x faster) is the ground truth
  for score changes; seeds are deterministic, so always compare on the same seed set with mean +/- SE.
  Server-side timing logs exist on the box (/opt/surv/server_log.jsonl) if custom logging is enabled.
  The kill post-mortem script is the right tool for any question about why agents die.
