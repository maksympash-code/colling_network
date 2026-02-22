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
T_MAX = 80.0
GRAD_T_MAX = 25.0

# stopping
MAX_REJECT_STREAK = 60
TAU_RADIUS = 4 
RNG_SEED = 42