"""Identity training-reward hook used by the pristine TMAX recipe."""


def shape_rewards(context):
    return list(context["base_rewards"])
