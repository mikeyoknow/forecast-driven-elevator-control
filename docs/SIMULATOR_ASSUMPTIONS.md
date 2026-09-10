# Digital-Twin Assumptions and Validity Boundaries

## Purpose

The simulator is a controlled experimental environment for comparing elevator
policies on identical passenger demand. It is not a certified representation
of any particular commercial elevator system.

## Building and motion assumptions

- Floors use zero-based indices; floor 0 is the lobby.
- Every elevator begins at the lobby at the start of a simulated day.
- All elevators have the same capacity, travel speed, and door time.
- Travel time is constant per floor.
- Acceleration, deceleration, leveling delay, and door obstruction are folded
  into the simplified travel and door constants.
- A car moves at most one floor after each configured travel interval.
- Doors must finish their configured cycle before the car moves again.
- Passenger boarding and exiting occur at the instant a stop is served; their
  shared time cost is represented by the door cycle.
- Elevators do not fail, enter maintenance mode, or experience emergency stops.

## Passenger assumptions

- Every passenger has one arrival time, origin, and different destination.
- Passenger events are anonymous; no identity or demographic information is
  generated or used.
- A passenger enters the system only at the recorded arrival second.
- Passengers do not abandon the queue or choose stairs.
- A passenger boards only the elevator to which the active assignment policy
  assigned the call.
- Capacity overflow is returned to the global hall-call queue for reassignment.
- All passengers must eventually be served; otherwise the simulation fails.

## Dispatch assumptions

- The reactive baseline scores distance, existing stops, queued calls, current
  occupancy, travel direction, and same-floor availability.
- Full elevators cannot receive a new passenger assignment.
- Passenger service always overrides speculative idle-car positioning.
- Stop selection continues in the current direction when possible, then
  reverses after exhausting stops in that direction.
- Elevator-ID tie-breaking is deterministic.
- The current baseline uses passenger origins when assigning calls. Destination
  information affects the route after boarding, as in a conventional system.

## Traffic assumptions

- Passenger traffic is generated from morning, lunch, evening, mixed, and
  event-surge regimes.
- Per-day floor popularity, overall intensity, and short-term intensity vary
  stochastically under fixed and reproducible seeds.
- Synthetic origin-destination events let every policy receive exactly the same
  counterfactual demand.
- Synthetic performance demonstrates behavior within the model; it does not by
  itself establish real-building effectiveness.

## Evaluation assumptions

- One simulated day/session is the independent comparison unit.
- Policies are paired on identical passenger days.
- Passenger waiting time is boarding time minus arrival time.
- Ride time is drop-off time minus boarding time.
- Journey time is drop-off time minus arrival time.
- Long waits are defined by a configurable threshold, currently 60 seconds.
- Fairness is initially summarized as the gap between the highest and lowest
  observed origin-floor mean wait.
- Floors travelled is an energy and wear proxy, not a calibrated energy model.
- Paired confidence intervals resample days, not individual passengers.

## Automated validity checks

The test suite verifies deterministic generation, time and floor bounds,
passenger conservation, mutually exclusive waiting/onboard/unassigned states,
capacity, direction reversal, overflow reassignment, drain behavior, source-day
immutability, and hand-calculated timing.

## External-validity limitations to report

- Real elevator controllers have proprietary routing, safety, and motion logic.
- Real passenger arrivals may be clustered, correlated, strategic, or affected
  by visible elevator state.
- Accessibility needs, freight service, priority calls, and fire-service modes
  are not currently modeled.
- A future version should calibrate arrival rates, OD matrices, travel times,
  and door times against published or measured building data.
