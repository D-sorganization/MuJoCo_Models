"""Bench press model builder for MuJoCo MJCF.

The lifter lies supine on a bench. The barbell is gripped in both hands
at approximately shoulder width. The model starts in the lockout position
(arms extended) and the motion descends the bar to the chest then presses
back to lockout.

Biomechanical notes:
- Primary movers: pectoralis major, anterior deltoid, triceps brachii
- The bench constrains pelvis and torso to a supine orientation
- Scapular retraction and arch are simplified (torso stays rigid on bench)
- Grip width affects shoulder abduction angle and pec activation

The bench is modelled as a ground-welded platform that constrains the
pelvis to a supine position at bench height (0.43 m, standard IPF).
MuJoCo Z-up convention: gravity = (0, 0, -9.80665).
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.utils.mjcf_helpers import add_weld_constraint

logger = logging.getLogger(__name__)

BENCH_HEIGHT = 0.43  # IPF standard bench height (meters)

# Supine lockout position hints (radians)
_INITIAL_SHOULDER_FLEX = -1.5708  # arms overhead (~90 deg from standing)
_INITIAL_ELBOW_FLEX = 0.0  # fully extended


class BenchPressModelBuilder(ExerciseModelBuilder):
    """Builds a bench-press MuJoCo MJCF model.

    The pelvis is welded to ground in a supine orientation at bench height.
    The barbell shaft is welded to both hands at grip width.
    """

    def __init__(self, config: ExerciseConfig | None = None) -> None:
        super().__init__(config)

    @property
    def exercise_name(self) -> str:
        return "bench_press"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld barbell to both hands at grip width.

        The grip is approximately shoulder-width (~0.40 m from center
        on each side for a standard grip).
        """
        self._attach_barbell_to_hands(equality)

    def _add_bench(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Add a bench body and weld the pelvis to it."""
        bench = ET.SubElement(worldbody, "body")
        bench.set("name", "bench")
        bench.set("pos", f"0 0 {BENCH_HEIGHT - 0.02:.6f}")
        bench_geom = ET.SubElement(bench, "geom")
        bench_geom.set("type", "box")
        bench_geom.set("size", "0.30 0.65 0.02")
        bench_geom.set("rgba", "0.5 0.35 0.2 1")
        bench_geom.set("contype", "1")
        bench_geom.set("conaffinity", "1")

        add_weld_constraint(
            equality,
            name="pelvis_to_bench",
            body1="pelvis",
            body2="bench",
        )
        logger.debug("Added bench body at height %.3f m", BENCH_HEIGHT)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set supine lockout position: shoulder and elbow joint refs."""
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            joint_type = joint.get("type", "hinge")
            if joint_type != "hinge":
                continue
            if "shoulder" in name:
                joint.set("ref", str(math.degrees(_INITIAL_SHOULDER_FLEX)))
            elif "elbow" in name:
                joint.set("ref", str(math.degrees(_INITIAL_ELBOW_FLEX)))
        logger.debug(
            "Setting bench press initial pose: shoulder=%.1f°, elbow=%.1f°",
            math.degrees(_INITIAL_SHOULDER_FLEX),
            math.degrees(_INITIAL_ELBOW_FLEX),
        )

    def build(self) -> str:
        """Build bench press model, adding bench body and pelvis weld."""
        return self._build_with_bench()

    def _build_with_bench(self) -> str:
        """Full build with bench body injected before serialization."""
        from mujoco_models.shared.barbell import create_barbell_bodies
        from mujoco_models.shared.body import create_full_body
        from mujoco_models.shared.contracts.postconditions import ensure_mjcf_root
        from mujoco_models.shared.utils.mjcf_helpers import serialize_model

        logger.info("Building %s model (with bench)", self.exercise_name)

        root = ET.Element("mujoco", model=self.exercise_name)

        g = self.config.gravity
        ET.SubElement(
            root,
            "option",
            gravity=f"{g[0]:.6f} {g[1]:.6f} {g[2]:.6f}",
            timestep="0.001",
        )
        ET.SubElement(root, "compiler", angle="radian", coordinate="local")

        default = ET.SubElement(root, "default")
        ET.SubElement(default, "joint", damping="5.0", armature="0.1")
        ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(
            worldbody,
            "geom",
            name="ground",
            type="plane",
            size="5 5 0.1",
            rgba="0.9 0.9 0.9 1",
        )

        equality = ET.SubElement(root, "equality")

        body_bodies = create_full_body(worldbody, self.config.body_spec)
        barbell_bodies = create_barbell_bodies(
            worldbody, equality, self.config.barbell_spec
        )

        self.attach_barbell(equality, body_bodies, barbell_bodies)
        self._add_bench(worldbody, equality)
        self.set_initial_pose(worldbody)

        # Add position actuators for all hinge joints
        actuator = ET.SubElement(root, "actuator")
        for joint in worldbody.iter("joint"):
            if joint.get("type", "hinge") == "free":
                continue
            jname = joint.get("name", "")
            if jname:
                act = ET.SubElement(actuator, "position")
                act.set("name", f"act_{jname}")
                act.set("joint", jname)
                act.set("kp", "100")

        # Add joint position sensors
        sensor = ET.SubElement(root, "sensor")
        for joint in worldbody.iter("joint"):
            if joint.get("type", "hinge") == "free":
                continue
            jname = joint.get("name", "")
            if jname:
                s = ET.SubElement(sensor, "jointpos")
                s.set("name", f"pos_{jname}")
                s.set("joint", jname)

        xml_str = serialize_model(root)
        ensure_mjcf_root(xml_str)
        logger.debug("Successfully built %s model", self.exercise_name)
        return xml_str


def build_bench_press_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float = 50.0,
) -> str:
    """Convenience function to build a bench press model MJCF XML string."""
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=plate_mass_per_side),
    )
    return BenchPressModelBuilder(config).build()
