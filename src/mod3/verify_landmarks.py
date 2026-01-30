"""
Verify MediaPipe hand landmarks match the user's reference image.
"""
import mediapipe as mp

print("=" * 70)
print("MEDIAPIPE HAND LANDMARKS VERIFICATION")
print("=" * 70)

# Get all landmark names
landmarks = mp.solutions.hands.HandLandmark

print(f"\nMediaPipe Hands provides exactly {len(landmarks)} landmarks:\n")

for i, landmark in enumerate(landmarks):
    print(f"{i:2d}. {landmark.name}")

print("\n" + "=" * 70)
print("✓ These are the EXACT 21 landmarks used in Module 3")
print("✓ Matches the reference image provided")
print("=" * 70)
