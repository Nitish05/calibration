# Arduino Firmware

PlatformIO project for the **Arduino Uno** (despite the historical name
of the source folder, `platformio.ini` is `board = uno`) that sits on
the CNC Shield V3 and drives the six rig actuators in response to the
serial protocol documented in [`../SERIAL_COMMANDS.md`](../SERIAL_COMMANDS.md).

## Build & flash

```bash
cd firmware
pio run                 # compile
pio run -t upload       # compile + upload to /dev/cnc_aux (or auto-detect)
pio device monitor      # 115200 baud
```

PlatformIO will auto-fetch the two library deps (`Servo`,
`AccelStepper`) declared in `platformio.ini` on the first `pio run` if
they're not already cached under `.pio/libdeps/uno/` (a vendored copy
is committed for offline / deterministic builds).

## What's inside

- `src/main.cpp` — the whole firmware. Pin defs at lines 6-30; the
  protocol handler is `processCommand()` starting at line 82.
- `platformio.ini` — `[env:uno]`, `monitor_speed = 115200`, `lib_deps`.
- `include/`, `lib/`, `test/` — empty PlatformIO scaffold directories
  (the `README` files inside are the PlatformIO templates).
- `.pio/libdeps/uno/` — vendored `Servo` and `AccelStepper` source.

## Pinout

See [`../HANDOFF.md` §2.1](../HANDOFF.md#21-arduino--cnc-shield-v3-wiring)
for the full wiring table. Single source of truth for pin assignments
is `src/main.cpp:6-30`.
