# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composite skills: whole steps an agent asks for, composed from the session's commands.

:class:`CompositeSkills` has the skills that use two clients, walking (``navigate_to``) and then
moving the arm: ``pick``, ``place``, ``place_upright``, ``open_drawer`` and ``close_drawer``. They
take explicit affordances: an object's box, a drawer's :class:`~phygs.session.ArticulatedPart`, a
point. Where an affordance comes from (the privileged hints, or an estimate from the cameras) is
the caller's. Stances are computed from the affordance's geometry and the robot's grip range, not
looked up per scene. Each skill tries a bounded, listed set of options in order (:class:`Tuning`),
returns an :class:`Outcome` naming the option that worked and what the robot measured, checks
itself with the robot's own readings (the gripper's hold reading, the hand's travel), and raises
:class:`~phygs.session.AgentFailed` when every option failed.

This is agent-side code: it imports ``phygs.session`` only (``tests/test_agent_imports.py``), reads
robot facts from ``session.robot`` and keeps no module state; its tuning is marked as tuning.
"""

from phygs.composite.skills import CompositeSkills, Outcome
from phygs.composite.tuning import TUNING, Tuning

__all__ = ["TUNING", "CompositeSkills", "Outcome", "Tuning"]
