# discretization
GRID_N = 101

# hotspot thresholds
ALPHA_1 = 0.75
ALPHA_2 = 1.11

# pruning acceptance / safety
DELTA_C = 0.2
LAMBDA = 0.8
EPC_C = 1.0

# HBS selection
SIGMA_HBS = 2.5
N_CANDIDATES = int(4 * SIGMA_HBS)

# unfavored list
UNFAVORED_TTL = 200

# pumping pressure maintenance
NR_RECOMPUTE_PRESSURE = 20

# simplified constraints (contest-like)
T_MAX = 99.0
GRAD_T_MAX = 5.0

# stopping
MAX_REJECT_STREAK = 60
TAU_RADIUS = 4
RNG_SEED = 42

# initial pumping pressure search range
P_INIT_LOW = 1e3
P_INIT_HIGH = 1e6

# safe margin for pressure re-optimization
SAFE_MARGIN = 0.2

MAX_TEMP_RISE_STAGE3 = 1.0
MAX_GRAD_RISE_STAGE3 = 0.5