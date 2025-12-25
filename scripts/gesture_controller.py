"""
Step 4: Gesture to Motor Command Mapping
File: gesture_controller.py

Maps gesture recognition output to robot motion commands.
- Gesture + confidence -> motor speeds (left, right)
- Handles confidence thresholding
- Outputs JSON format for robot control

This module produces the SAME command format as the joystick:
{
    "type": "motor_command",
    "left": <speed>,
    "right": <speed>,
    "source": "gesture"
}

Python 3.10.1 compatible
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MotorCommand:
    """Motor command structure"""
    left: int      # Left motor speed (-255 to 255)
    right: int     # Right motor speed (-255 to 255)
    source: str    # "gesture" or "joystick"
    gesture: str   # Name of recognized gesture
    confidence: float  # Confidence score (0-1)
    
    def to_json(self) -> str:
        """Convert to JSON for transmission"""
        return json.dumps({
            'type': 'motor_command',
            'left': self.left,
            'right': self.right,
            'source': self.source,
            'gesture': self.gesture,
            'confidence': f'{self.confidence:.2f}'
        })


class GestureController:
    """
    Maps gesture to motor commands
    """
    
    def __init__(self):
        """
        Initialize controller with gesture -> motion mapping
        """
        self.confidence_threshold = 0.6
        
        # Gesture to motor mapping
        # Format: gesture -> (left_speed, right_speed)
        # Speed range: -255 to 255
        #   Positive = forward
        #   Negative = backward
        self.gesture_to_motor = {
            'UP': (255, 255),        # Forward: both full forward
            'DOWN': (-255, -255),    # Backward: both full backward
            'LEFT': (-150, 255),     # Left turn: left slower, right faster
            'RIGHT': (255, -150),    # Right turn: right slower, left faster
            'STOP': (0, 0),          # Stop: both zero
            'NONE': (0, 0),          # No gesture: stop
        }
    
    def gesture_to_command(
        self,
        gesture: str,
        confidence: float,
        apply_threshold: bool = True
    ) -> MotorCommand:
        """
        Convert gesture to motor command
        
        Args:
            gesture: Recognized gesture name (UP, DOWN, LEFT, RIGHT, STOP, NONE)
            confidence: Confidence score (0-1)
            apply_threshold: If True, treat low confidence as STOP
        
        Returns:
            MotorCommand with left/right speeds
        """
        
        # Validate gesture
        if gesture not in self.gesture_to_motor:
            gesture = 'NONE'
            confidence = 0.0
        
        # Apply confidence threshold
        if apply_threshold and confidence < self.confidence_threshold:
            left, right = self.gesture_to_motor['STOP']
            effective_gesture = 'STOP'
        else:
            left, right = self.gesture_to_motor[gesture]
            effective_gesture = gesture
        
        return MotorCommand(
            left=int(left),
            right=int(right),
            source='gesture',
            gesture=effective_gesture,
            confidence=float(confidence)
        )
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set confidence threshold (0-1)
        Commands below this confidence are treated as STOP
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.confidence_threshold = threshold
    
    def set_gesture_mapping(self, gesture: str, left: int, right: int):
        """
        Customize gesture to motor mapping
        """
        if gesture not in self.gesture_to_motor:
            raise ValueError(f"Unknown gesture: {gesture}")
        
        # Clamp to valid range
        left = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        
        self.gesture_to_motor[gesture] = (left, right)
    
    @staticmethod
    def demo():
        """
        Demonstrate gesture to motor mapping
        """
        print("=" * 70)
        print("🎮 GESTURE TO MOTOR MAPPING DEMO")
        print("=" * 70)
        
        controller = GestureController()
        
        print(f"\nConfidence threshold: {controller.confidence_threshold}\n")
        
        # Test gestures
        test_cases = [
            ('UP', 0.95),
            ('DOWN', 0.88),
            ('LEFT', 0.92),
            ('RIGHT', 0.85),
            ('STOP', 0.90),
            ('NONE', 0.70),
            ('UP', 0.45),      # Below threshold
        ]
        
        print("Gesture      | Confidence | Left | Right | JSON")
        print("-" * 70)
        
        for gesture, conf in test_cases:
            cmd = controller.gesture_to_command(gesture, conf)
            gesture_display = gesture if conf >= controller.confidence_threshold else f"{gesture}*"
            print(f"{gesture_display:12s} | {conf:6.2f}     | " +
                  f"{cmd.left:4d}  | {cmd.right:5d}  | " +
                  f"L:{cmd.left:4d}, R:{cmd.right:4d}")
        
        print("\n* = Below confidence threshold (treated as STOP)")
        print("\nJSON Output Example:")
        cmd = controller.gesture_to_command('LEFT', 0.92)
        print(cmd.to_json())


class RobotMotionPlanner:
    """
    High-level motion planning
    Handles edge cases and safety
    """
    
    def __init__(self, controller: GestureController):
        self.controller = controller
        self.last_command = None
        self.command_history = []
    
    def plan_motion(
        self,
        gesture: str,
        confidence: float,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.0
    ) -> Dict:
        """
        Plan motion from gesture
        
        Returns dict with:
        - motor_command: MotorCommand object
        - velocities: (linear, angular) in m/s and rad/s
        - is_safe: bool
        """
        
        # Get motor command
        motor_cmd = self.controller.gesture_to_command(gesture, confidence)
        
        # Convert motor speeds to normalized velocities
        # Left and right motor speeds are -255 to 255
        # Robot motion:
        #   v = (left + right) / 2 / 255
        #   w = (right - left) / 2 / 255
        
        left_norm = motor_cmd.left / 255.0
        right_norm = motor_cmd.right / 255.0
        
        linear_vel = (left_norm + right_norm) / 2.0 * max_linear_speed
        angular_vel = (right_norm - left_norm) / 2.0 * max_angular_speed
        
        # Safety checks
        is_safe = abs(linear_vel) <= max_linear_speed and abs(angular_vel) <= max_angular_speed
        
        return {
            'motor_command': motor_cmd,
            'linear_velocity': float(linear_vel),
            'angular_velocity': float(angular_vel),
            'is_safe': is_safe,
            'left_motor': motor_cmd.left,
            'right_motor': motor_cmd.right
        }


if __name__ == '__main__':
    GestureController.demo()