# Arduino Motor Control - Serial Command Reference

## Overview

An Arduino Uno with a CNC Shield V3 controls 6 actuators via serial commands. The Arduino is connected to a Raspberry Pi 5 via USB. The RPi5 runs a Flask webserver that sends these commands over serial to control the motors.

## Serial Connection

- **Port**: `/dev/ttyACM0` (may vary — use `ls /dev/ttyACM* /dev/ttyUSB*` to find it)
- **Baud rate**: `115200`
- **Data format**: 8-N-1
- **Protocol**: Send a command string terminated by newline (`\n`). Format is a single letter followed by a number. The Arduino responds with a text confirmation on each command.

## Actuators

| # | Type | Description | Driver |
|---|---|---|---|
| 1 | NEMA stepper motor | X-axis linear movement | HR4988 (A4988 clone) on CNC Shield |
| 2 | NEMA stepper motor | Z-axis linear movement | HR4988 (A4988 clone) on CNC Shield |
| 3 | 28BYJ-48 stepper motor | Rotation (BYJ #1) | ULN2003 |
| 4 | 28BYJ-48 stepper motor | Rotation (BYJ #2) | ULN2003 |
| 5 | SG90 servo motor | Grip angle (0-180 degrees) | Direct PWM |
| 6 | DC motor (5V) | Continuous rotation fwd/rev | L298N (motor A side) |

## Commands

All commands are case-insensitive and must end with a newline (`\n`).

### Stepper Motors (relative steps)

| Command | Action | Arduino Response |
|---|---|---|
| `X<steps>` | Move X-axis stepper by N steps (negative = reverse) | `X MOVE <steps> -> <position>` |
| `Z<steps>` | Move Z-axis stepper by N steps (negative = reverse) | `Z MOVE <steps> -> <position>` |
| `B<steps>` | Move 28BYJ-48 #1 by N steps (negative = reverse) | `BYJ1 MOVE <steps> -> <position>` |
| `J<steps>` | Move 28BYJ-48 #2 by N steps (negative = reverse) | `BYJ2 MOVE <steps> -> <position>` |

### Servo (absolute angle)

| Command | Action | Arduino Response |
|---|---|---|
| `O<angle>` | Set servo angle (0-180) | `SERVO -> <angle>` |

### DC Motor (speed 0-100)

| Command | Action | Arduino Response |
|---|---|---|
| `F<speed>` | Run motor forward at speed (0-100) | `DC FWD speed: <speed>` |
| `G<speed>` | Run motor reverse at speed (0-100) | `DC REV speed: <speed>` |
| `S` | Stop motor | `DC STOP` |

### Reset

| Command | Action | Arduino Response |
|---|---|---|
| `R` | Reset all stepper positions to 0 and servo to 0 degrees | `RESET ALL` |
| `H` | Reset stepper positions only (servo unchanged) | `RESET STEPPERS` |

### Status

| Command | Action | Arduino Response |
|---|---|---|
| `P` | Print all stepper positions | `X: <pos> \| Z: <pos> \| BYJ1: <pos> \| BYJ2: <pos>` |

## Examples

```
X100      -> move X stepper forward 100 steps
X-100     -> move X stepper reverse 100 steps
Z50       -> move Z stepper up 50 steps
B256      -> move BYJ #1 right 256 steps
B-256     -> move BYJ #1 left 256 steps
J512      -> move BYJ #2 right 512 steps
O90       -> set servo to 90 degrees
O0        -> set servo to 0 degrees
O180      -> set servo to 180 degrees
F80       -> DC motor forward at speed 80
G50       -> DC motor reverse at speed 50
S         -> stop DC motor
R         -> reset all stepper positions to 0 and servo to 0
H         -> reset stepper positions only
P         -> print all positions
```

## Python Serial Example

```python
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino reset

# Move X stepper 200 steps forward
ser.write(b'X200\n')
response = ser.readline().decode().strip()
print(response)  # "X MOVE 200 -> 200"

# Move Z stepper 100 steps down
ser.write(b'Z-100\n')
response = ser.readline().decode().strip()
print(response)  # "Z MOVE -100 -> -100"

# Set servo to 90 degrees
ser.write(b'O90\n')
response = ser.readline().decode().strip()
print(response)  # "SERVO -> 90"

# DC motor forward at speed 80
ser.write(b'F80\n')
response = ser.readline().decode().strip()
print(response)  # "DC FWD speed: 80"

# Stop DC motor
ser.write(b'S\n')

# Get all positions
ser.write(b'P\n')
response = ser.readline().decode().strip()
print(response)  # "X: 200 | Z: -100 | BYJ1: 0 | BYJ2: 0"
```

## Notes

- Commands are case-insensitive (`x100` and `X100` both work)
- Stepper commands are incremental (relative movement from current position)
- Servo command is absolute (sets exact angle)
- DC motor speed is clamped to 0-100
- Servo angle is clamped to 0-180
- The Arduino sends a text response after every command (read with `readline()`)
- Stepper positions track cumulative steps from power-on (starting at 0)
- Wait for the Arduino's "Control Ready" message after opening the serial port before sending commands
- After opening the serial port, wait ~2 seconds for the Arduino to reset before sending commands
