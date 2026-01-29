#!/usr/bin/env python

import argparse
import os
import sys
import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep


class TerminalKeyboard:
    """Minimal terminal keyboard reader that works over SSH (no DISPLAY required).

    It treats motion keys as "pressed" for a short window after each keypress.
    Holding a key relies on the terminal key-repeat to refresh that window.
    """

    def __init__(self, press_window_s: float = 0.25):
        self.press_window_s = press_window_s
        self._connected = False
        self._orig_term_settings = None
        self._active_until: dict[str, float] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if os.name != "posix":
            raise RuntimeError("TerminalKeyboard fallback is only supported on POSIX terminals.")
        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a TTY; cannot read keys in terminal mode.")

        import termios
        import tty

        self._orig_term_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._connected = True

    def _read_available_chars(self) -> list[str]:
        import select

        chars: list[str] = []
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            chars.append(ch)
        return chars

    def get_pressed_keys(self) -> list[str]:
        now = time.time()
        for ch in self._read_available_chars():
            # Ctrl+C should still interrupt, but in raw mode it might come as \x03
            if ch == "\x03":
                raise KeyboardInterrupt
            self._active_until[ch] = now + self.press_window_s

        # expire
        expired = [k for k, t in self._active_until.items() if t < now]
        for k in expired:
            del self._active_until[k]

        return list(self._active_until.keys())

    def disconnect(self) -> None:
        if not self._connected:
            return
        if os.name == "posix" and self._orig_term_settings is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_term_settings)
        self._connected = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate LeKiwi base (no leader arm required).")
    parser.add_argument("--remote_ip", required=True, help="IP address of the host machine running lekiwi_host")
    parser.add_argument("--id", default="my_lekiwi", help="Robot id (used only for logs/local calibration path)")
    parser.add_argument("--cmd_port", type=int, default=5555, help="ZMQ command port (host.port_zmq_cmd)")
    parser.add_argument(
        "--obs_port",
        type=int,
        default=5556,
        help="ZMQ observations port (host.port_zmq_observations)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    robot_config = LeKiwiClientConfig(
        remote_ip=args.remote_ip,
        id=args.id,
        port_zmq_cmd=args.cmd_port,
        port_zmq_observations=args.obs_port,
        cameras={},  # host may have cameras disabled
    )
    keyboard_config = KeyboardTeleopConfig(id=f"{args.id}_keyboard")

    robot = LeKiwiClient(robot_config)

    # Prefer global keyboard listener when available; fallback to terminal reader for SSH/no-DISPLAY.
    teleop: object
    keyboard = KeyboardTeleop(keyboard_config)

    robot.connect()
    keyboard.connect()

    if getattr(keyboard, "is_connected", False):
        teleop = keyboard
        print("Connected. Using pynput keyboard listener.")
    else:
        # Common on SSH/headless linux: DISPLAY not set => pynput intentionally disabled.
        terminal_kb = TerminalKeyboard()
        terminal_kb.connect()
        teleop = terminal_kb
        print("Connected. Using terminal keyboard (SSH-friendly, no DISPLAY).")

    print("Use WASD to translate, Z/X to rotate, R/F to change speed. Ctrl+C to exit.")

    try:
        while True:
            t0 = time.perf_counter()

            if isinstance(teleop, KeyboardTeleop):
                pressed_keys = teleop.get_action()
            else:
                pressed_keys = teleop.get_pressed_keys()

            base_action = robot._from_keyboard_to_base_action(pressed_keys)

            if base_action:
                robot.send_action(base_action)

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        try:
            if isinstance(teleop, TerminalKeyboard):
                teleop.disconnect()
            else:
                if getattr(keyboard, "is_connected", False):
                    keyboard.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
