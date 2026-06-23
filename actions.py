def make_action(
    ws="",
    ws_weight=0.0,
    ad="",
    ad_weight=0.0,
    qe="",
    qe_weight=0.0,
    rf="",
    rf_weight=0.0,
    fire=False,
):
    return {
        "moveWS": {"command": ws, "weight": ws_weight},
        "moveAD": {"command": ad, "weight": ad_weight},
        "turretQE": {"command": qe, "weight": qe_weight},
        "turretRF": {"command": rf, "weight": rf_weight},
        "fire": fire,
    }


def action_command(action, key):
    value = (action or {}).get(key, {})
    return value.get("command", "")


def action_weight(action, key):
    value = (action or {}).get(key, {})
    return value.get("weight", 0.0)