from available_actions import reach
from ltl_operators import ap
from ltl_operators import ltl_and, ltl_or, ltl_not, ltl_until, ltl_eventually, ltl_always, ltl_imply
def question():
    """
    Go to object_5 then the object_2.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_5 and then object_2.
    # object_5 to object_2: object_2 should not be reached before object_5 (`not until rules`)

    # create atomic propositions for objects
    reach_obj_5 = ap(reach, "object_5")  # reach(object_5)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_eventually(ltl_and(reach_obj_5, ltl_eventually(reach_obj_2)))  # Reach object_5 then object_2
    c2 = ltl_until(ltl_not(reach_obj_2), reach_obj_5)  # object_2 should not be reached before object_5
    formula = ltl_and(c1, c2)  # Combine the constraints
    return formula
