from ActionMapping import ACTION_TYPES

def action_to_vec(action_type):
    vec = [1 if action_type == act else 0 for act in ACTION_TYPES]
    return vec