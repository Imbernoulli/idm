from ActionMapping import action_types

def action_to_vec(action_type):
    vec = [1 if action_type == act else 0 for act in action_types()]
    return vec