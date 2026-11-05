"""Run the real PerceptionControllerNode with a stand-in for VideoLLaMA3 (no GPU needed)."""
import sys
import types

from journalist_robot.models import videollama3_controller as real


class FakeController:
    def __init__(self, *args, **kwargs):
        pass

    def decide(self, frames, prompt, fps, temperature):
        text = ('{"linear_x": 0.3, "angular_z": 0.0, "confidence": 0.9, '
                '"newsworthiness_score": 0.95, "reasoning": "person at desk, approach"}')
        return real.ControllerOutput(text=text, latency_s=0.01)


fake = types.ModuleType("journalist_robot.models.videollama3_controller")
fake.VideoLLaMA3Controller = FakeController
sys.modules["journalist_robot.models.videollama3_controller"] = fake

from journalist_robot.nodes import perception_controller_node as pc  # noqa: E402

pc.main(args=sys.argv)
