<img width="520" height="83" alt="image" src="https://github.com/user-attachments/assets/e895da18-fc10-480a-8dcb-cfcdc3418534" />

SIGNapse
Ananya Bontha, Rachel Krol, Shruti Sivaraman, Anoushka Gharat


SETUP
You need Python 3.9 to 3.12 and a webcam.
  pip3 install numpy opencv-python mediapipe
The first run downloads a small hand tracking model, so you need internet.


HOW TO RUN
1. Record at least 2 signs (200 samples each, hold up the sign and move
   your hand a little):
    python3 signapse_hands.py collect A
    python3 signapse_hands.py collect B
  
2. Train:
  python3 signapse_hands.py train

3. Run the live demo:
  python3 signapse_hands.py live

Press q to quit. Space pauses while recording.

The screen shows the letter, how confident it is, and how long it took.
The computer says each new letter out loud. Add --mute to turn that off.


RUNNING IT ON THE CHIP DESIGN
To run predictions through the chip's Verilog instead of Python, install
Icarus Verilog (Mac: brew install icarus-verilog), put the chip files in a
folder called sim, then add --backend verilog:

  python3 signapse_hands.py live --backend verilog
  python3 signapse_hands.py eval --backend verilog

eval checks that the chip gives exactly the same answers as Python.

IF SOMETHING GOES WRONG:
Camera won't open: on Mac go to System Settings > Privacy & Security >
Camera and turn on Terminal.
"Nothing saved": your hand wasn't seen. Try better light.
"Need at least 2 different signs": record another letter.
"No trained model yet": run collect and train first.
Letter keeps showing "?": record more samples of that sign.

Credits
Hand tracking: MediaPipe Hand Landmarker by Google (Apache 2.0)

Webcam: OpenCV (Apache 2.0)

Training: NumPy (BSD 3-Clause)

Chip simulation: Icarus Verilog (GPL 2.0)

Quantization-aware training: Jacob et al., "Quantization and Training of Neural Networks
for Efficient Integer-Arithmetic-Only Inference," CVPR 2018

Similar approach: Kazuhito00, hand-gesture-recognition-using-mediapipe (Apache 2.0)


<img width="306" height="86" alt="image" src="https://github.com/user-attachments/assets/0442d90a-34d3-4af8-97f7-886ec9db43ef" />

Proud sponsor of Basic's of ASICs
