@dataclass(frozen=True)
class Stage2FullResult:
    net: CoolingNetwork
    thermal_before: ThermalResult
    thermal_after_straight1: ThermalResult
    thermal_after_barriers: Optional[ThermalResult]
    thermal_after_straight2: Optional[ThermalResult]
    used_barriers: bool
    barrier_params: Optional[List[Tuple[int, int]]]


def stage2_full_patterning(
    net: CoolingNetwork,
    power_map: np.ndarray,
    thermal_fn,
    alpha_1: float,
    alpha_2: float,
    io_q: IOQ,
    map_radius: int = 2,
    pad: int = 1,
    stripe: int = 2,
    # barrier candidates
    length_candidates: List[int] = None,
    theta_candidates: List[int] = None,
    length_scales: List[float] = None,
    barrier_thickness: int = 1,
    center_frac: float = 0.25,
    center_ratio_thr: float = 0.35,
) -> Stage2FullResult:
    """
    Stage 2 as in paper structure:
    - simulate -> hotspots(alpha1) -> straight-channels
    - if hotspots near center: optimize corner barriers -> apply
    - re-simulate -> hotspots(alpha2) -> straight-channels
    """
    if length_candidates is None:
        length_candidates = [6, 8, 10, 12, 14, 16]
    if theta_candidates is None:
        theta_candidates = [30, 45, 60]
    if length_scales is None:
        length_scales = [0.75, 1.0, 1.25]

    # ---- initial sim
    thermal_before: ThermalResult = thermal_fn(net, power_map)

    # ---- straight pass 1 (alpha_1)
    s2_1 = stage2_straight_channels(
        net=net,
        power_map=power_map,
        thermal_fn=thermal_fn,
        alpha_1=alpha_1,
        io_q=io_q,
        map_radius=map_radius,
        pad=pad,
        stripe=stripe,
    )
    net1 = s2_1.net
    thermal_after_straight1 = s2_1.thermal_after

    # determine center-hotspots from BEFORE straight (як у paper: після попередньої симуляції)
    hot_mask = hotspots_mask(thermal_before.T, alpha_1)
    used_barriers = hotspots_near_center_by_ratio(hot_mask, center_frac=center_frac, ratio_thr=center_ratio_thr)

    if not used_barriers:
        return Stage2FullResult(
            net=net1,
            thermal_before=thermal_before,
            thermal_after_straight1=thermal_after_straight1,
            thermal_after_barriers=None,
            thermal_after_straight2=None,
            used_barriers=False,
            barrier_params=None
        )

    # ---- barrier optimization (paper-like)
    # Base for barriers: after straight1 (бо в paper barriers + straight help together)
    per_corner = optimize_corner_barriers(
        base_net=net1,
        power_map=power_map,
        thermal_fn=thermal_fn,
        length_candidates=length_candidates,
        theta_candidates=theta_candidates,
        thickness=barrier_thickness,
    )
    best_params = line_search_lengths(
        base_net=net1,
        power_map=power_map,
        thermal_fn=thermal_fn,
        per_corner_params=per_corner,
        scales=length_scales,
        thickness=barrier_thickness,
    )

    net_bar = net1.clone()
    apply_corner_barriers(net_bar, best_params, thickness=barrier_thickness)
    net_bar.prune_irregular()
    if not net_bar.has_inlet_to_outlet_path():
        # fail-safe: if barriers broke connectivity, fall back to net1
        net_bar = net1.clone()

    thermal_after_barriers: ThermalResult = thermal_fn(net_bar, power_map)

    # ---- straight pass 2 (alpha_2) після бар’єрів + пересимуляції
    s2_2 = stage2_straight_channels(
        net=net_bar,
        power_map=power_map,
        thermal_fn=thermal_fn,
        alpha_1=alpha_2,          # reuse same function but with alpha_2 threshold
        io_q=io_q,
        map_radius=map_radius,
        pad=pad,
        stripe=stripe,
    )

    return Stage2FullResult(
        net=s2_2.net,
        thermal_before=thermal_before,
        thermal_after_straight1=thermal_after_straight1,
        thermal_after_barriers=thermal_after_barriers,
        thermal_after_straight2=s2_2.thermal_after,
        used_barriers=True,
        barrier_params=best_params
    )