import time
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def happy(mini):
    print("😊 HAPPY")

    mini.goto_target(
        head=create_head_pose(pitch=-8, degrees=True),
        antennas=np.deg2rad([25, 25]),
        duration=0.5,
        method="cartoon",
    )

    mini.goto_target(
        head=create_head_pose(pitch=5, degrees=True),
        antennas=np.deg2rad([-10, -10]),
        duration=0.35,
        method="cartoon",
    )

    mini.goto_target(
        head=create_head_pose(pitch=-3, degrees=True),
        antennas=np.deg2rad([15, 15]),
        duration=0.35,
        method="cartoon",
    )


def surprised(mini):
    print("😮 SURPRISED")

    mini.goto_target(
        head=create_head_pose(z=12, pitch=-8, degrees=True, mm=True),
        antennas=np.deg2rad([45, 45]),
        duration=0.4,
        method="cartoon",
    )

    time.sleep(0.6)

    mini.goto_target(
        head=create_head_pose(),
        antennas=np.deg2rad([10, 10]),
        duration=0.5,
        method="cartoon",
    )


def thinking(mini):
    print("🤔 THINKING")

    mini.goto_target(
        head=create_head_pose(yaw=-18, roll=8, degrees=True),
        antennas=np.deg2rad([10, -10]),
        duration=0.7,
        method="cartoon",
    )

    time.sleep(0.8)

    mini.goto_target(
        head=create_head_pose(yaw=18, roll=-8, degrees=True),
        antennas=np.deg2rad([-10, 10]),
        duration=0.7,
        method="cartoon",
    )

    time.sleep(0.8)

    mini.goto_target(
        head=create_head_pose(),
        antennas=np.deg2rad([0, 0]),
        duration=0.6,
        method="cartoon",
    )


def greeting(mini):
    print("👋 GREETING")

    mini.goto_target(
        head=create_head_pose(pitch=-5, degrees=True),
        antennas=np.deg2rad([20, 20]),
        duration=0.5,
        method="cartoon",
    )

    mini.goto_target(
        antennas=np.deg2rad([-25, 25]),
        duration=0.3,
        method="cartoon",
    )

    mini.goto_target(
        antennas=np.deg2rad([25, -25]),
        duration=0.3,
        method="cartoon",
    )

    mini.goto_target(
        antennas=np.deg2rad([0, 0]),
        duration=0.4,
        method="cartoon",
    )


def main():

    with ReachyMini(media_backend="default") as mini:

        print("🤖 Reachy expression test")
        print()

        greeting(mini)
        time.sleep(1)

        happy(mini)
        time.sleep(1)

        surprised(mini)
        time.sleep(1)

        thinking(mini)
        time.sleep(1)

        print("🏠 Returning home...")

        mini.goto_target(
            head=create_head_pose(),
            antennas=np.deg2rad([0, 0]),
            duration=1.0,
        )

        print("Done!")


if __name__ == "__main__":
    main()
