"""Microduck aimed soccer kick: PPO policies and a contact-physics MuJoCo evaluation.

Closed-loop, observation-to-action controllers for the Open Duck Mini v2 in MuJoCo. The robot is commanded one of
three zones of a goal and must put the ball in that zone; a second policy celebrates after a goal, and an optional
approach policy walks the robot up to the ball first. Everything that moves the robot is a trained network.
"""
