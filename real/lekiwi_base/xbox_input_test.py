import math
import pygame

DEADZONE = 0.15
STICK = "left"          # "left" or "right"

# If your D-pad is buttons, set these to the indices you saw in diagnose_gamepad.py
DPAD_BUTTONS = dict(left=11, right=12, up=13, down=14)  # adjust if needed

def normalize_axis(v, dz):
    return 0.0 if abs(v) < dz else max(-1.0, min(1.0, v))

def stick_axes(js, which="left"):
    # Common macOS mapping: 0: LX, 1: LY, 2: LT, 3: RX, 4: RY, 5: RT
    ax_x, ax_y = (0, 1) if which == "left" else (3, 4)
    x, y = js.get_axis(ax_x), -js.get_axis(ax_y)  # invert y so +y is up
    return x, y

def to_polar(x, y):
    r = (x*x + y*y) ** 0.5
    ang = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return r, ang

def eight_way_label(angle_deg):
    dirs8 = ["E","NE","N","NW","W","SW","S","SE"]
    return dirs8[int((angle_deg + 22.5) // 45) % 8]

def read_dpad(js):
    """Return (hx, hy, angle_deg, label or None) for the D-pad."""
    # Prefer hat if available
    if js.get_numhats() > 0:
        hx, hy = js.get_hat(0)  # (-1..1, -1..1)
        r, ang = to_polar(hx, hy)
        return hx, hy, ang, (eight_way_label(ang) if r > 0 else None)

    # Fallback: D-pad as buttons
    # Build hx, hy from buttons (left/right/up/down)
    def pressed(btn_idx): 
        return js.get_button(btn_idx) == 1

    hx = (-1 if pressed(DPAD_BUTTONS["left"]) else 0) + (1 if pressed(DPAD_BUTTONS["right"]) else 0)
    hy = (-1 if pressed(DPAD_BUTTONS["down"]) else 0) + (1 if pressed(DPAD_BUTTONS["up"]) else 0)
    r, ang = to_polar(hx, hy)
    return hx, hy, ang, (eight_way_label(ang) if r > 0 else None)

def main():
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("No controller found.")
    js = pygame.joystick.Joystick(0)

    print(f"Using: {js.get_name()}")
    print(f"Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}, Hats: {js.get_numhats()}")

    last_label = None
    last_hat_state = None
    clock = pygame.time.Clock()

    while True:
        for _ in pygame.event.get():
            pass

        # stick
        x, y = stick_axes(js, STICK)
        x, y = normalize_axis(x, DEADZONE), normalize_axis(y, DEADZONE)
        r, ang = to_polar(x, y)
        label = eight_way_label(ang) if r > 0 else None
        print(f"{STICK} stick:", "neutral" if label is None else f"vec=({x:.2f},{y:.2f}) r={r:.2f} angle={ang:.1f}° dir={label}")

        # d-pad
        hx, hy, hang, hlabel = read_dpad(js)
        state = (hx, hy)
        print("D-pad:", "neutral" if state == (0,0) else f"({hx:+d},{hy:+d}) angle={hang:.1f}° dir={hlabel}")

        clock.tick(120)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
