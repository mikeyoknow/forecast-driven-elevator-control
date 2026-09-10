# Data Dictionary and Feature Availability

## Passenger-event table

| Field | Type | Meaning | Available when |
|---|---|---|---|
| `passenger_id` | integer | Day-local anonymous passenger identifier | Simulation only |
| `arrival_time` | integer seconds | Time the passenger enters the hall-call system | At arrival |
| `origin` | integer floor | Floor where the passenger requests service | At arrival |
| `destination` | integer floor | Requested destination | After boarding for conventional dispatch |
| `assigned_elevator` | optional integer | Current assigned car | After assignment |
| `boarding_time` | optional integer seconds | Time the passenger enters a car | After boarding |
| `dropoff_time` | optional integer seconds | Time the passenger exits | After completion |

Passenger identity is synthetic and anonymous. Identity is never used as a
forecast feature.

## Building-day table

| Field | Meaning |
|---|---|
| `day_id` | Unique split-scenario-repetition identifier |
| `split` | `train`, `validation`, or a separately controlled evaluation split |
| `scenario` | Morning, lunch, evening, mixed, or surge traffic regime |
| `seed` | Deterministic day-generation seed |
| `duration_minutes` | Passenger-generation interval |
| `passengers` | Passenger-event collection |
| `origin_counts` | Minute-by-floor origin count matrix |
| `destination_counts` | Minute-by-floor destination count matrix |

## Forecasting sample

One row represents one prediction minute within one building day. Windows never
cross day boundaries.

### Audit metadata

| Field | Meaning |
|---|---|
| `sample_id` | Unique day-minute-horizon identifier |
| `day_id`, `split`, `scenario`, `seed` | Source-day lineage |
| `input_start_minute` | Earliest minute included in the longest history |
| `input_end_minute_exclusive` | First minute not included in any feature |
| `target_start_minute` | First predicted minute |
| `target_end_minute_exclusive` | End of future target window |
| `horizon_minutes` | Prediction horizon |

The leakage rule is explicit:

```text
input_end_minute_exclusive <= target_start_minute
```

## Forecast features

The 15-floor development configuration produces 101 columns:

- `time_sin`, `time_cos`: cyclical clock representation
- `elapsed_fraction`: normalized position within the traffic session
- `total_origin_last_{1,3,5,10}m`: recent total traffic intensity
- `scenario_{morning,lunch,evening,mixed}`: training-regime indicators
- `origin_floor_{f}_last_{1,3,5,10}m`: per-floor origin histories
- `destination_floor_{f}_last_3m`: recent completed destination demand
- `origin_floor_{f}_recent_trend`: last-minute demand minus the preceding
  two-minute average

Every count feature ends strictly before the prediction minute. The unseen
`surge` scenario receives zeros for all training-regime indicators instead of a
new category learned from evaluation data.

## Forecast targets

For each floor `f`, the target is:

```text
origin_floor_f_next_{horizon}m
```

It counts passenger arrivals whose origin is floor `f` from the target start up
to, but not including, the target end. Horizons are 1, 2, and 5 minutes, with 2
minutes designated as the primary task.

## Operational outputs

- Passenger mean, median, and P95 wait
- Long-wait rate above 60 seconds
- Ride and total journey time
- Origin-floor service metrics and fairness gap
- Floors travelled, stops, and direction reversals
- Maximum load and peak capacity utilization
- Unserved passengers and post-generation drain time
