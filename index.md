# Smart Glasses
I built a pair of smart glasses that help visually impaired users navigate their environment by recognizing objects in real time and providing spoken feedback. Using a Raspberry Pi with a camera, machine learning, and text-to-speech technology, the glasses detect and announce nearby objects. The biggest challenges were setting up the hardware-software integration and fine-tuning the object recognition to be both accurate and timely, but overcoming these hurdles has made the project highly impactful.




| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Saanvi G | Leland | Electrical/Computer Engineering | Incoming Senior


![Headstone Image](SaanviG (1).png)
<img src="SaanviG (1).png" alt="Headshot" height="400" width="500">
# Fourth Milestone
For my fourth milestone, I added a voice changer mode that would switch between all the modification: weather alerts, obstacle detection (with the ultraosnic sensor), object detection, and navigation. 
-->
# Third Milestone
<iframe width="560" height="315" src="https://www.youtube.com/embed/c_jwYHTCRQo?si=JgL7zhWAZ2tKjFEF" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my third milestone, I added obstacle detection using three ultrasonic sensors (positioned on the left, center, and right of the smart glasses), which allowed the glasses to calculate the distance of nearby objects and announce their direction and distance out loud using text-to-speech (TTS). The smart glasses speak messages like "Obstacle 45 centimeters to the right" depending on sensor readings. I also integrated this with object detection, so if an object is detected at a short distance, it will say, for example, "Person detected 35 centimeters to the center." This feature combines object recognition with physical world proximity.

**How I did this**
To start, I used three HC-SR04 ultrasonic sensors connected to the Raspberry Pi’s GPIO pins. In my Python script
```shell
(object_detect_tts.py)
```
, I created a SENSORS dictionary at line 26 to define each sensor's TRIG and ECHO pins:
```shell
SENSORS = {
    'left': {'TRIG': 20, 'ECHO': 21},
    'center': {'TRIG': 23, 'ECHO': 24},
    'right': {'TRIG': 16, 'ECHO': 2},
}
```

I then initialized these pins with
```shell
GPIO.setup()
```
and 
```shell
GPIO.output()
```
between lines 31–35, setting all TRIG pins low to start. The main function that measures distance is get_distance() (defined at line 57), which sends a pulse using the TRIG pin, waits for the echo signal, calculates the pulse duration, and converts that to distance using the speed of sound. Next, I initialized the TensorFlow Lite interpreter for object detection at lines 41–46, loading the .tflite model and setting up input/output details. In the main loop (starting at line 139), I opened the Pi camera and began processing frames. For each frame, I used the detect_object() function (defined at line 86) to resize the frame to the expected input shape (line 93), convert it to a NumPy array and match the dtype required by the model (line 95), and then run inference and get the top label with confidence (lines 99–102). After detecting an object, I collected distance measurements from all three sensors using a dictionary comprehension at line 149.
```shell
distances = {
    side: get_distance(pins['TRIG'], pins['ECHO'])
    for side, pins in SENSORS.items()
}
```


I then filtered these to check if any direction was under 100 cm. If yes, I chose the closest direction and constructed a message like "chair detected 55 centimeters to the left" or "Obstacle 38 centimeters to the center" depending on whether an object was recognized (starting at line 155).

Text-to-speech is handled using pyttsx3 which I initialized at line 11, and I created a speak() function (line 14) to speak the messages out loud.

I also added a debounce timer using last_spoken and debounce_time (line 143) to make sure the glasses don’t repeatedly speak the same thing too often.

**Challenges**
One major challenge I faced was debugging the ultrasonic sensor wiring and logic. Because I was using three sensors at once, there was a lot of interference between signals, especially when multiple TRIG pins were triggered simultaneously or ECHO pins shared wrong GPIO assignments. I solved this by isolating each sensor’s function call and spacing out the measurements using time.sleep(0.2) at the end of the main loop.

Another issue was using the wrong ECHO pin for the right sensor. I had mistakenly used pin 26 but realized (after troubleshooting incorrect distance readings) that it needed to be GPIO pin 2 since pin 26 was fried. Once I corrected that, the readings became accurate.

**Next Steps**
Now that obstacle detection is integrated and directional audio feedback is working, my next steps involve expanding the smart glasses’ functionality through voice-controlled mode switching using a wake word like “Hey Glasses” with Porcupine. I plan to incorporate weather alerts, battery level monitoring, and ChatGPT voice interaction to allow users to ask questions or receive assistance through natural speech. To ensure seamless transitions between these features, I will implement threaded mode switching so each function runs independently and can be interrupted or switched on demand. I also aim to optimize all capabilities into a single, efficient, and thread-safe Python script with low latency and fast response times on the Raspberry Pi. Looking ahead, I’m exploring the addition of bone conduction speaker, which would make the glasses more adaptable in quiet or noisy environments.

# Second Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/msAcJ9Dd6P4?si=iyv4Mc0ZYasIaY0i" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my second milestone, I implemented object recognition using a pre-trained machine learning model and added text-to-speech (TTS) so the classification can be read aloud into an earpiece. I also set up a virtual environment to manage Python dependencies and isolate packages. I followed the tutorial "Running TensorFlow Lite Object Recognition on the Raspberry Pi 4 or Pi 5" by M. LeBlanc-Williams.
**How I did This:**
I started by setting up a virtual environment:
```shell
sudo apt install python3.11-venv
python3 -m venv env --system-site-packages
source env/bin/activate
```
Then, I connected and tested the Pi camera:
```shell
libcamera-hello
```

After verifying the camera worked,  I began by enabling the Pi Camera and installing dependencies:
```shell
sudo apt update
sudo apt install python3-pip
sudo pip3 install --upgrade setuptools
```
I then downloaded the TensorFlow Lite example repository:
```shell
git clone https://github.com/tensorflow/examples.git --depth 1
cd examples/lite/examples/object_detection/raspberry_pi
```
From there, I ran the setup script provided:
```shell
sh setup.sh
```
This script installed required packages and libraries such as OpenCV, TensorFlow Lite runtime, and various system dependencies needed for image processing and model inference.
Then I installed all the required packages:
```shell
pip3 install --upgrade adafruit-python-shell
wget https://raw.githubusercontent.com/adafruit/Raspberry-Pi-Installer-Scripts/master/raspi-blinka.py
python3 raspi-blinka.py
apt install -y python3-numpy python3-pillow python3-pygame
apt install -y festival
#tensor flow
RELEASE=https://github.com/PINTO0309/Tensorflow-bin/releases/download/v2.15.0.post1/tensorflow-2.15.0.post1-cp311-none-linux_aarch64.whl
CPVER=$(python --version | grep -Eo '3\.[0-9]{1,2}' | tr -d '.')
pip install $(echo "$RELEASE" | sed -e "s/cp[0-9]\{3\}/CP$CPVER/g")
```

Once everything was installed, I tested the camera again using this code:
```shell
# SPDX-FileCopyrightText: 2021 Limor Fried/ladyada for Adafruit Industries
# SPDX-FileCopyrightText: 2021 Melissa LeBlanc-Williams for Adafruit Industries
#
# SPDX-License-Identifier: MIT

import time
import logging
import argparse
import pygame
import os
import subprocess
import sys
import numpy as np
import signal

CONFIDENCE_THRESHOLD = 0.5   # at what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # what percentage of the time we have to have seen a thing

def dont_quit(signal, frame):
   print('Caught signal: {}'.format(signal))
signal.signal(signal.SIGHUP, dont_quit)

# App
from rpi_vision.agent.capturev2 import PiCameraStream
from rpi_vision.models.mobilenet_v2 import MobileNetV2Base

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# initialize the display
pygame.init()
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
capture_manager = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')

    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270],
                        dest='rotation', action='store', default=0,
                        help='Rotate everything on the display by this amount')
    args = parser.parse_args()
    return args

last_seen = [None] * 10
last_spoken = None

def main(args):
    global last_spoken, capture_manager

    capture_manager = PiCameraStream(preview=False)

    if args.rotation in (0, 180):
        buffer = pygame.Surface((screen.get_width(), screen.get_height()))
    else:
        buffer = pygame.Surface((screen.get_height(), screen.get_width()))

    pygame.mouse.set_visible(False)
    screen.fill((0,0,0))
    try:
        splash = pygame.image.load(os.path.dirname(sys.argv[0])+'/bchatsplash.bmp')
        splash = pygame.transform.rotate(splash, args.rotation)
        # Scale the square image up to the smaller of the width or height
        splash = pygame.transform.scale(splash, (min(screen.get_width(), screen.get_height()), min(screen.get_width(), screen.get_height())))
        # Center the image
        screen.blit(splash, ((screen.get_width() - splash.get_width()) // 2, (screen.get_height() - splash.get_height()) // 2))

    except pygame.error:
        pass
    pygame.display.update()

    # Let's figure out the scale size first for non-square images
    scale = max(buffer.get_height() // capture_manager.resolution[1], 1)
    scaled_resolution = tuple([x * scale for x in capture_manager.resolution])

    # use the default font, but scale it
    smallfont = pygame.font.Font(None, 24 * scale)
    medfont = pygame.font.Font(None, 36 * scale)
    bigfont = pygame.font.Font(None, 48 * scale)

    model = MobileNetV2Base(include_top=args.include_top)

    capture_manager.start()
    while not capture_manager.stopped:
        if capture_manager.frame is None:
            continue
        buffer.fill((0,0,0))
        frame = capture_manager.read()
        # get the raw data frame & swap red & blue channels
        previewframe = np.ascontiguousarray(capture_manager.frame)
        # make it an image
        img = pygame.image.frombuffer(previewframe, capture_manager.resolution, 'RGB')
        img = pygame.transform.scale(img, scaled_resolution)

        cropped_region = (
            (img.get_width() - buffer.get_width()) // 2,
            (img.get_height() - buffer.get_height()) // 2,
            buffer.get_width(),
            buffer.get_height()
        )

        # draw it!
        buffer.blit(img, (0, 0), cropped_region)

        timestamp = time.monotonic()
        if args.tflite:
            prediction = model.tflite_predict(frame)[0]
        else:
            prediction = model.predict(frame)[0]
        logging.info(prediction)
        delta = time.monotonic() - timestamp
        logging.info("%s inference took %d ms, %0.1f FPS" % ("TFLite" if args.tflite else "TF", delta * 1000, 1 / delta))
        print(last_seen)

        # add FPS & temp on top corner of image
        fpstext = "%0.1f FPS" % (1/delta,)
        fpstext_surface = smallfont.render(fpstext, True, (255, 0, 0))
        fpstext_position = (buffer.get_width()-10, 10) # near the top right corner
        buffer.blit(fpstext_surface, fpstext_surface.get_rect(topright=fpstext_position))
        try:
            temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
            temptext = "%d\N{DEGREE SIGN}C" % temp
            temptext_surface = smallfont.render(temptext, True, (255, 0, 0))
            temptext_position = (buffer.get_width()-10, 30) # near the top right corner
            buffer.blit(temptext_surface, temptext_surface.get_rect(topright=temptext_position))
        except OSError:
            pass

        for p in prediction:
            label, name, conf = p
            if conf > CONFIDENCE_THRESHOLD:
                print("Detected", name)

                persistant_obj = False  # assume the object is not persistant
                last_seen.append(name)
                last_seen.pop(0)

                inferred_times = last_seen.count(name)
                if inferred_times / len(last_seen) > PERSISTANCE_THRESHOLD:  # over quarter time
                    persistant_obj = True

                detecttext = name.replace("_", " ")
                detecttextfont = None
                for f in (bigfont, medfont, smallfont):
                    detectsize = f.size(detecttext)
                    if detectsize[0] < screen.get_width(): # it'll fit!
                        detecttextfont = f
                        break
                else:
                    detecttextfont = smallfont # well, we'll do our best
                detecttext_color = (0, 255, 0) if persistant_obj else (255, 255, 255)
                detecttext_surface = detecttextfont.render(detecttext, True, detecttext_color)
                detecttext_position = (buffer.get_width()//2,
                                       buffer.get_height() - detecttextfont.size(detecttext)[1])
                buffer.blit(detecttext_surface, detecttext_surface.get_rect(center=detecttext_position))

                if persistant_obj and last_spoken != detecttext:
                    subprocess.call(f"echo {detecttext} | festival --tts &", shell=True)
                    last_spoken = detecttext
                break
        else:
            last_seen.append(None)
            last_seen.pop(0)
            if last_seen.count(None) == len(last_seen):
                last_spoken = None

        screen.blit(pygame.transform.rotate(buffer, args.rotation), (0,0))
        pygame.display.update()

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        capture_manager.stop()
```
Then I ran these commands in order to test out the object regonication and TTS:
```shell
cd rpi-vision
python3 tests/pitft_labeled_output.py --tflite
```

**Challenge:**
I kept on using sudo at first to download my packages, but every time I did that, my computer would crash. That’s when I learned that since I was downloading these in my virtual environment, using sudo bypasses the environment entirely and installs or modifies packages at the system level, which can cause conflicts, permission issues, and even crash the Raspberry Pi. From then on, I made sure to activate my environment and use regular pip install commands inside it, keeping everything isolated, safer, and more stable.

**Next:**
Now that object detection and audio output are working, I plan on working on my modifications which include obstacle detection with vibration feedback using ultrasonic sensors and motors, voice command mode switching for hands-free control using offline voice recognition, and a battery monitoring system that gives spoken alerts like “Battery at 15%” to prevent sudden shutdowns.
# First Milestone
<iframe width="560" height="315" src="https://www.youtube.com/embed/xsjFLJVsTAk?si=yyqdQAVWjonmXYLz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
For my first milestone, I set up my Raspberry Pi and configured the tools needed to control, code, and monitor it from my main computer. These tools include PuTTY, TigerVNC, Visual Studio Code, and OBS Studio. This setup laid the foundation for development and debugging throughout the project.
**How I did This:**
First, I installed OBS Studio on my Windows computer. I used this to view and configure the Raspberry Pi’s display by connecting a monitor, keyboard, and mouse directly to the Pi. OBS helped me visualize the Pi’s screen output to ensure it was functioning correctly during setup.

Next, I installed PuTTY. This tool allowed me to SSH (secure shell) into the Raspberry Pi from my laptop, meaning I could access and control the Pi’s terminal remotely over Wi-Fi. PuTTY was essential for running command-line operations without needing a physical display.

After that, I set up TigerVNC to create a virtual desktop interface. This gave me remote GUI (graphical user interface) access to the Pi’s desktop from my laptop. It was incredibly helpful for navigating the Raspberry Pi's interface and running visual applications without using a second screen.

Lastly, I installed VS Code with the Remote-SSH extension. This allowed me to open and edit files on the Raspberry Pi directly from VS Code on my laptop. It made writing, debugging, and testing Python scripts much easier and faster.

**Challenge:**
One major challenge I faced was network syncing between my Raspberry Pi and my laptop. Initially, both devices were connected to the school's Wi-Fi, but due to multiple routers, they were often on different sub-networks, which prevented them from communicating properly. I resolved this issue by switching to the Wi-Fi in my specific classroom, which only had one router, ensuring both devices were on the same local network and could sync.

**Next:**
I plan to move on to object recognition using machine learning and added text-to-speech (TTS) capabilities to communicate the detected object through audio.

# Starter Milestone
<iframe width="560" height="315" src="https://www.youtube.com/embed/61sh7fCtAME?si=76WeBUf0ZRzxAlKb" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

**Summary**
I chose the retro arcade console solder kit for my project because I enjoy classic gaming and wanted a hands-on experience building a device I can actually use for fun and entertainment. The kit works by assembling various electronic components on a printed circuit board (PCB) to create a fully functional arcade console. The core of the console is a microcontroller that runs the game software and processes inputs from the buttons and joystick. When a button is pressed, it closes a switch, completing the circuit and sending a signal to the microcontroller. The microcontroller then updates the display and game logic accordingly. The joystick operates similarly by providing directional inputs through switches that close when moved. The console outputs video and audio signals to a screen and speakers, allowing the player to see and hear the game in action. The assembly process primarily involved soldering the components such as buttons, joystick, resistors, capacitors, microcontroller, and connectors—onto the PCB. Careful soldering and component placement were essential to ensure the console worked correctly and was durable.
**Figure:**
The figure below illustrates how pressing a button closes a switch on the circuit board, completing the connection needed to send an input signal to the microcontroller. When the switch is open (button unpressed), no signal is sent.
<img src="retro.png" alt="finished product" height="400" width="500">

**Components Used:**
1. Printed Circuit Board (PCB)
2. Microcontroller (e.g., ATmega or similar)
3. Buttons (multiple push buttons)
4. Joystick Module
5. Resistors and Capacitors
6. Audio Amplifier and Speaker
7. HDMI or AV Output Connector
8. Power Supply Connector
9. Solder
10. Connecting Wires
11. Enclosure for the console housing
    
<img src="(OIP.webp" alt="materials" height="400" width="500">   

**Challenges:**
When soldering together my starter project I came across two main challenges. FIrst I had a hard time recongize which screws where which whne finishing off my project and screwing the case together. THe other rpoblem I had was soldering the wires. I ended up cutting my wire too much meaning that the ends of my wires were rubber which you can't solder cocmpletly. I had to slowley cut out the rubber around the wire to reveal the metal part in order to solder properly. this took a long time since I had to make sure that i wouldnt accidently cut the whole wire off and have to restart. 

**Next:**
I'm excited to work on my main project which is the object detection smart glasses!

# Code
** ultrasonic sensor (obstacle detection) code**
```shell
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import sys
from gtts import gTTS
import pygame

# === TTS Setup (gTTS + pygame) ===
pygame.mixer.init()
def speak(text):
    print("🔊", text, flush=True)
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("/tmp/speak.mp3")
        pygame.mixer.music.load("/tmp/speak.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"❌ Failed to speak: {e}")

# === GPIO Setup ===
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

SENSORS = {
    'left': {'TRIG': 20, 'ECHO': 21},
    'center': {'TRIG': 23, 'ECHO': 24},
    'right': {'TRIG': 16, 'ECHO': 2},
}

for pos in SENSORS.values():
    GPIO.setup(pos['TRIG'], GPIO.OUT)
    GPIO.setup(pos['ECHO'], GPIO.IN)
    GPIO.output(pos['TRIG'], GPIO.LOW)

time.sleep(1)

# === Load TFLite Model ===
MODEL_PATH = "/home/strawberry/Documents/object-glasses/object-glasses/env/model.tflite"

LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Distance Function ===
def get_distance(trig, echo):
    GPIO.output(trig, GPIO.LOW)
    time.sleep(0.01)
    GPIO.output(trig, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig, GPIO.LOW)

    start_time = time.time()
    timeout = start_time + 0.1

    while GPIO.input(echo) == 0:
        if time.time() > timeout:
            return 999
    pulse_start = time.time()

    timeout = pulse_start + 0.1
    while GPIO.input(echo) == 1:
        if time.time() > timeout:
            return 999
    pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = (pulse_duration * 34300) / 2
    return round(distance, 2)

# === Object Detection ===
def detect_object(frame):
    expected_shape = input_details[0]['shape']
    height, width = expected_shape[1], expected_shape[2]
    dtype = input_details[0]['dtype']

    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(img, axis=0).astype(dtype)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num = int(interpreter.get_tensor(output_details[3]['index'])[0])

    best_score = 0.0
    best_class = -1
    for i in range(num):
        if scores[i] > best_score and scores[i] > 0.3:
            best_score = scores[i]
            best_class = int(classes[i])

    if best_class >= 0 and best_class < len(LABELS):
        return LABELS[best_class], best_score
    return None, None

# === Main Distance + Detection Mode ===
def run_distance_mode():
    print("🚀 Distance mode started", flush=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera failed to open", flush=True)
        return

    last_spoken = time.time()
    debounce_time = 3
    last_spoken_msg = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            obj_label, confidence = detect_object(frame)
            distances = {
                side: get_distance(pins['TRIG'], pins['ECHO'])
                for side, pins in SENSORS.items()
            }

            if time.time() - last_spoken > debounce_time:
                valid = {d: dist for d, dist in distances.items() if dist < 100 and dist != 999}
                if valid:
                    direction = min(valid, key=valid.get)
                    dist = valid[direction]

                    if obj_label:
                        msg = f"{obj_label} detected {int(dist)} centimeters to the {direction}"
                    else:
                        msg = f"Obstacle {int(dist)} centimeters to the {direction}"

                    if msg != last_spoken_msg:
                        speak(msg)
                        last_spoken = time.time()
                        last_spoken_msg = msg

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("🛑 Distance mode stopped by user.", flush=True)

    finally:
        cap.release()
        GPIO.cleanup()

# === Allow standalone run ===
if __name__ == "__main__":
    run_distance_mode()

```
** nagivation code**
```shell
import os
import time
import re
import requests
import speech_recognition as sr
from gtts import gTTS
import pygame

# === Google API Key ===
GOOGLE_API_KEY = 'AIzaSyClBTJqcQHmNeuDqPvUAyyrp9TmJmXx0Mw'

# === Initialize pygame mixer for TTS ===
pygame.mixer.init()

# === TTS function using gTTS ===
def speak(text):
    print("Speaking:", text)
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("/tmp/speak.mp3")
        pygame.mixer.music.load("/tmp/speak.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"❌ Failed to speak: {e}")

# === Speech recognition function ===
def listen(prompt=None):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    if prompt:
        speak(prompt)

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Heard: {text}")
        return text.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
    except sr.RequestError:
        speak("Speech recognition service is not available.")
    return None

# === Get coordinates from address ===
def get_coordinates(address):
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": GOOGLE_API_KEY
    }
    res = requests.get(geocode_url, params=params)
    if res.status_code != 200 or not res.json()["results"]:
        speak("Could not find the location: " + address)
        return None
    location = res.json()["results"][0]["geometry"]["location"]
    return location["lat"], location["lng"]

# === Get walking directions ===
def get_walking_directions(origin_lat, origin_lng, destination_address):
    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": destination_address,
        "mode": "walking",
        "key": GOOGLE_API_KEY
    }
    res = requests.get(directions_url, params=params)
    if res.status_code != 200 or not res.json()["routes"]:
        speak("Failed to get directions.")
        return []

    steps = res.json()["routes"][0]["legs"][0]["steps"]
    directions = []
    for step in steps:
        instruction = re.sub('<[^<]+?>', '', step["html_instructions"])
        directions.append(instruction)
    return directions

# === Wait for navigation command ===
def wait_for_command():
    while True:
        command = listen("Say 'next', 'repeat', or 'back'.")
        if command:
            if "next" in command:
                return "next"
            elif "repeat" in command:
                return "repeat"
            elif "back" in command:
                return "back"
            else:
                speak("Command not recognized. Please say 'next', 'repeat', or 'back'.")
        else:
            speak("Please say a command.")

# === Main logic ===
def main():
    origin_address = listen("What is your current location?")
    if not origin_address:
        return
    origin_coords = get_coordinates(origin_address)
    if not origin_coords:
        return
    origin_lat, origin_lng = origin_coords

    destination_address = listen("Where do you want to go?")
    if not destination_address:
        return

    steps = get_walking_directions(origin_lat, origin_lng, destination_address)
    if steps:
        speak("Here are your walking directions. Say 'next' to go forward, 'repeat' to hear the step again, or 'back' to go back.")
        index = 0
        while 0 <= index < len(steps):
            speak(f"Step {index + 1}: {steps[index]}")
            command = wait_for_command()

            if command == "next":
                index += 1
            elif command == "repeat":
                continue  # Say the same step again
            elif command == "back":
                if index > 0:
                    index -= 1
                else:
                    speak("You're at the first step.")

        speak("You have arrived at your destination.")
    else:
        speak("Sorry, I couldn't find a walking route.")

if __name__ == "__main__":
    main()

```
** weather alerts code**
```shell
import requests
from datetime import datetime
import pytz
import time
import threading
import speech_recognition as sr
from gtts import gTTS
import pygame

class WeatherAssistant:
    def __init__(self):
        # === CONFIGURATION ===
        self.API_KEY = '90cf24e49aba700e6c94c8d4677f20a4'
        self.CITY = 'San Jose'
        self.LAT = 37.3394
        self.LON = -121.8950
        self.UNITS = 'imperial'
        self.TIMEZONE = 'America/Los_Angeles'

        # === Speech Recognizer Setup ===
        self.recognizer = sr.Recognizer()

        # === TTS Setup using gTTS ===
        pygame.mixer.init()

    def speak(self, text):
        print("🔊", text)
        try:
            tts = gTTS(text=text, lang='en')
            tts.save("/tmp/speak.mp3")
            pygame.mixer.music.load("/tmp/speak.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"❌ TTS error: {e}")

    def listen_mode(self):
        self.speak("Which mode would you like? You can say: full report, UV and air only, or weather and clothing only.")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("🎤 Listening using SpeechRecognition...")
            try:
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio).lower()
                print("🎧 Heard:", text)
                if "uv" in text and "air" in text:
                    return "uv_air"
                elif "weather" in text or "clothing" in text:
                    return "weather_clothing"
                elif "full" in text:
                    return "full"
                else:
                    self.speak("Sorry, I didn't understand that. I'll give you the full report.")
                    return "full"
            except sr.WaitTimeoutError:
                self.speak("Timeout, please try again.")
                return self.manual_mode_input()
            except sr.UnknownValueError:
                self.speak("Sorry, I couldn't understand. Please try again.")
                return self.manual_mode_input()
            except sr.RequestError as e:
                print("API error:", e)
                self.speak("Speech service is unavailable. Please type your mode.")
                return self.manual_mode_input()

    def manual_mode_input(self):
        print("Please type mode (full, uv_air, weather_clothing):")
        mode = input().strip().lower()
        return mode if mode in ["full", "uv_air", "weather_clothing"] else "full"

    def get_weather(self):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={self.CITY}&appid={self.API_KEY}&units={self.UNITS}"
        try:
            return requests.get(url).json()
        except Exception as e:
            print("Weather error:", e)
            return None

    def get_air_quality(self):
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={self.LAT}&lon={self.LON}&appid={self.API_KEY}"
        try:
            return requests.get(url).json()
        except Exception as e:
            print("AQI error:", e)
            return None

    def get_onecall(self):
        url = f"https://api.openweathermap.org/data/2.5/onecall?lat={self.LAT}&lon={self.LON}&exclude=minutely,alerts&appid={self.API_KEY}&units={self.UNITS}"
        try:
            return requests.get(url).json()
        except Exception as e:
            print("OneCall error:", e)
            return None

    def interpret_aqi(self, aqi):
        return {
            1: "Good",
            2: "Fair",
            3: "Moderate",
            4: "Poor",
            5: "Very Poor"
        }.get(aqi, "Unknown")

    def interpret_uv(self, uv):
        if uv <= 2:
            return "Low – minimal risk."
        elif uv <= 5:
            return "Moderate – sunglasses and sunscreen recommended."
        elif uv <= 7:
            return "High – wear SPF and avoid direct sunlight."
        elif uv <= 10:
            return "Very High – use full sun protection and seek shade."
        else:
            return "Extreme – stay indoors if possible."

    def clothing_advice(self, feels_like):
        if feels_like < 50:
            return "It feels cold. Wear a warm jacket or coat."
        elif feels_like < 70:
            return "It feels mild. A light jacket or sweater should be fine."
        elif feels_like < 85:
            return "It feels warm. Dress comfortably and stay hydrated."
        else:
            return "It feels very hot. Wear light clothing and avoid long sun exposure."

    def allergy_advice(self, aqi):
        return "Air quality is moderate or worse. If you have allergies or asthma, consider staying indoors or wearing a mask." if aqi >= 3 else "Air quality is okay for most people."

    def humidity_advice(self, humidity):
        return f"Humidity is high at {humidity}%. It may feel sticky and uncomfortable." if humidity > 70 else None

    def forecast_change_alert(self, hourly, current_desc):
        for hour in hourly[:6]:
            weather = hour['weather'][0]['main'].lower()
            if any(x in weather for x in ["rain", "storm", "snow"]) and not any(x in current_desc.lower() for x in ["rain", "storm", "snow"]):
                return "Although it's clear now, rain or storms are expected later today. You might want to take an umbrella."
        return None

    def format_time(self, unix_time):
        tz = pytz.timezone(self.TIMEZONE)
        return datetime.fromtimestamp(unix_time, tz).strftime('%I:%M %p')

    def generate_report(self, mode):
        weather = self.get_weather()
        air = self.get_air_quality()
        onecall = self.get_onecall()

        desc = ""
        if mode in ["weather_clothing", "full"] and weather:
            temp = int(weather['main']['temp'])
            feels_like = int(weather['main']['feels_like'])
            desc = weather['weather'][0]['description']
            humidity = int(weather['main']['humidity'])
            sunrise = self.format_time(weather['sys']['sunrise'])
            sunset = self.format_time(weather['sys']['sunset'])

            self.speak(f"The current temperature is {temp} degrees, but it feels like {feels_like} degrees with {desc}.")
            self.speak(self.clothing_advice(feels_like))

            humidity_msg = self.humidity_advice(humidity)
            if humidity_msg:
                self.speak(humidity_msg)

            self.speak(f"The sun rises at {sunrise} and sets at {sunset}.")

        if onecall and 'hourly' in onecall:
            forecast_msg = self.forecast_change_alert(onecall['hourly'], desc)
            if forecast_msg:
                self.speak(forecast_msg)

        if mode in ["uv_air", "full"] and air:
            aqi = air['list'][0]['main']['aqi']
            self.speak(f"The air quality index is {aqi}, which is considered {self.interpret_aqi(aqi)}.")
            self.speak(self.allergy_advice(aqi))

        if mode in ["uv_air", "full"] and onecall:
            current = onecall.get('current')
            if current and 'uvi' in current:
                uv = current['uvi']
                self.speak(f"The UV index is {uv:.1f}. {self.interpret_uv(uv)}")
            else:
                self.speak("UV information is not currently available.")

    def run(self):
        mode = self.listen_mode()
        self.generate_report(mode)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting...")

def run_weather_mode():
    assistant = WeatherAssistant()
    assistant.run()

if __name__ == "__main__":
    run_weather_mode()

```

**gemini code (need to be fixed)**
```shell
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import tempfile

class GeminiAssistant:
    def __init__(self, api_key):
        # === CONFIG ===
        self.api_key = api_key

        # === INIT Gemini ===
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # === INIT pygame safely ===
        pygame.mixer.init(frequency=22050)

    def speak(self, text):
        print("🤖 Gemini says:", text)
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
            tts.save(temp_path)

        try:
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            os.remove(temp_path)

    def listen(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        print("🎤 Listening... Speak now!")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("⏰ Listening timed out.")
                return None

        try:
            query = recognizer.recognize_google(audio)
            print("You asked:", query)
            return query
        except sr.UnknownValueError:
            self.speak("Sorry, I couldn't understand you.")
        except sr.RequestError as e:
            self.speak("Speech recognition is unavailable.")
            print("Error:", e)

        return None

    def ask_once(self):
        user_input = self.listen()
        if not user_input:
            return None

        if user_input.lower() in ["stop", "exit", "quit"]:
            self.speak("Goodbye!")
            return "exit"

        try:
            response = self.model.generate_content(user_input)
            reply = response.candidates[0].content.parts[0].text
            self.speak(reply)
            return reply
        except Exception as e:
            print("❌ Error from Gemini:", e)
            self.speak("Something went wrong talking to Gemini.")
            return None
if __name__ == "__main__":
    API_KEY = "AIzaSyC6O8dLHqXwgndSmbQVnZuZULSg6XZRJXc"  # 👈 Replace with your real key
    assistant = GeminiAssistant(API_KEY)
    while True:
        result = assistant.ask_once()
        if result == "exit":
            break

 
```

**object detcetion code**
```shell
# object_detection_runner.py

import time
import logging
import pygame
import os
import subprocess
import sys
import numpy as np
import signal
import argparse
from gtts import gTTS  # ✅ gTTS import added

from rpi_vision.agent.capturev2 import PiCameraStream
from rpi_vision.models.mobilenet_v2 import MobileNetV2Base

CONFIDENCE_THRESHOLD = 0.5
PERSISTANCE_THRESHOLD = 0.25

# Prevent quitting on SIGHUP
def dont_quit(signal, frame):
    print('Caught signal:', signal)
signal.signal(signal.SIGHUP, dont_quit)

class ObjectDetectionRunner:
    def __init__(self, include_top=True, tflite=False, rotation=0):
        self.include_top = include_top
        self.tflite = tflite
        self.rotation = rotation
        self.last_seen = [None] * 10
        self.last_spoken = None
        self.capture_manager = None
        self.screen = None
        self.buffer = None

    def _init_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        self.screen.fill((0, 0, 0))
        try:
            splash_path = os.path.join(os.path.dirname(sys.argv[0]), 'bchatsplash.bmp')
            splash = pygame.image.load(splash_path)
            splash = pygame.transform.rotate(splash, self.rotation)
            splash = pygame.transform.scale(
                splash,
                (min(self.screen.get_width(), self.screen.get_height()),) * 2
            )
            self.screen.blit(
                splash,
                ((self.screen.get_width() - splash.get_width()) // 2,
                 (self.screen.get_height() - splash.get_height()) // 2)
            )
        except pygame.error:
            pass
        pygame.display.update()

    def run(self):
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

        self._init_display()
        self.capture_manager = PiCameraStream(preview=False)
        self.capture_manager.start()

        if self.rotation in (0, 180):
            self.buffer = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        else:
            self.buffer = pygame.Surface((self.screen.get_height(), self.screen.get_width()))

        scale = max(self.buffer.get_height() // self.capture_manager.resolution[1], 1)
        scaled_resolution = tuple([x * scale for x in self.capture_manager.resolution])

        smallfont = pygame.font.Font(None, 24 * scale)
        medfont = pygame.font.Font(None, 36 * scale)
        bigfont = pygame.font.Font(None, 48 * scale)

        model = MobileNetV2Base(include_top=self.include_top)

        try:
            while not self.capture_manager.stopped:
                if self.capture_manager.frame is None:
                    continue

                self.buffer.fill((0, 0, 0))
                frame = self.capture_manager.read()
                previewframe = np.ascontiguousarray(self.capture_manager.frame)
                img = pygame.image.frombuffer(previewframe, self.capture_manager.resolution, 'RGB')
                img = pygame.transform.scale(img, scaled_resolution)

                cropped_region = (
                    (img.get_width() - self.buffer.get_width()) // 2,
                    (img.get_height() - self.buffer.get_height()) // 2,
                    self.buffer.get_width(),
                    self.buffer.get_height()
                )
                self.buffer.blit(img, (0, 0), cropped_region)

                timestamp = time.monotonic()
                prediction = model.tflite_predict(frame)[0] if self.tflite else model.predict(frame)[0]
                delta = time.monotonic() - timestamp
                fps_text = f"{1/delta:.1f} FPS"
                fpstext_surface = smallfont.render(fps_text, True, (255, 0, 0))
                self.buffer.blit(fpstext_surface, fpstext_surface.get_rect(topright=(self.buffer.get_width() - 10, 10)))

                try:
                    temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
                    temptext = f"{int(temp)}°C"
                    temptext_surface = smallfont.render(temptext, True, (255, 0, 0))
                    self.buffer.blit(temptext_surface, temptext_surface.get_rect(topright=(self.buffer.get_width() - 10, 30)))
                except OSError:
                    pass

                for p in prediction:
                    label, name, conf = p
                    if conf > CONFIDENCE_THRESHOLD:
                        print("Detected:", name)
                        self.last_seen.append(name)
                        self.last_seen.pop(0)
                        if self.last_seen.count(name) / len(self.last_seen) > PERSISTANCE_THRESHOLD:
                            detecttext = name.replace("_", " ")
                            for f in (bigfont, medfont, smallfont):
                                if f.size(detecttext)[0] < self.screen.get_width():
                                    detecttextfont = f
                                    break
                            else:
                                detecttextfont = smallfont

                            detect_color = (0, 255, 0)
                            surface = detecttextfont.render(detecttext, True, detect_color)
                            self.buffer.blit(surface, surface.get_rect(center=(self.buffer.get_width() // 2, self.buffer.get_height() - detecttextfont.size(detecttext)[1])))

                            if self.last_spoken != detecttext:
                                tts = gTTS(text=detecttext)
                                tts.save("/tmp/detect.mp3")
                                subprocess.call("mpg123 /tmp/detect.mp3", shell=True)
                                self.last_spoken = detecttext
                        break
                else:
                    self.last_seen.append(None)
                    self.last_seen.pop(0)
                    if all(x is None for x in self.last_seen):
                        self.last_spoken = None

                self.screen.blit(pygame.transform.rotate(self.buffer, self.rotation), (0, 0))
                pygame.display.update()

        except KeyboardInterrupt:
            print("🛑 Interrupted by user.")
        finally:
            self.capture_manager.stop()
            pygame.quit()

# Optional CLI for standalone run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool, default=True)
    parser.add_argument('--tflite', action='store_true', default=False)
    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270], default=0)
    args = parser.parse_args()

    runner = ObjectDetectionRunner(include_top=args.include_top, tflite=args.tflite, rotation=args.rotation)
    runner.run()

def run_object_mode():
    runner = ObjectDetectionRunner(include_top=True, tflite=False, rotation=0)
    runner.run()

```
**voice mode changer code**
```shell
import time
import object_detect_tts
import pitft_labeled_output
import weather_alerts
import gemini
from gemini import GeminiAssistant
import button_camera
import navigation
from navigation import NavigationMode

import speech_recognition as sr
from gtts import gTTS
import pygame
import tempfile
import os

# === TTS setup with gTTS ===
pygame.mixer.init()

def speak(text):
    print("🤖", text)
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
        tts.save(fp.name)
        pygame.mixer.music.load(fp.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

# === Speech Recognition setup ===
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_with_timeout(timeout=10):
    speak("I'm listening...")
    print("🎤 Listening for command...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Returning to idle.")
            return None

    try:
        text = recognizer.recognize_google(audio).lower()
        print("Heard:", text)
        return text
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
    except sr.RequestError as e:
        speak("Speech recognition is unavailable.")
        print("Error:", e)
    return None

def wait_for_wake_word(wake_word="hey glasses"):
    print("👂 Waiting for wake word...")
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                continue

        try:
            text = recognizer.recognize_google(audio).lower()
            print("Heard:", text)
            if wake_word in text:
                speak("Yes?")
                return
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print("Error with speech recognition:", e)
            continue

def run_mode(mode):
    if mode == "distance":
        speak("Entering distance mode.")
        object_detect_tts.run_distance_mode()

    elif mode == "object":
        speak("Entering object detection mode.")
        pitft_labeled_output.run_object_mode()

    elif mode == "weather":
        speak("Fetching weather alerts.")
        weather_alerts.run_weather_mode()

    elif mode == "gemini":
        speak("Opening Gemini Assistant.")
        from gemini import GeminiAssistant
        assistant = GeminiAssistant(api_key="YOUR_API_KEY")  # Replace with actual key

        while True:
            reply = assistant.ask_once()
            if reply == "exit":
                break
            if not reply:
                assistant.speak("Repeat that.")
            else:
                break  # Exit after one successful answer

    elif mode == "navigation":
        speak("Opening Navigation system.")
        navigation_mode = NavigationMode()
        navigation_mode.run()

    else:
        speak(f"I don't recognize the mode: {mode}")

if __name__ == "__main__":
    speak("Voice Mode Switcher is active.")
    try:
        while True:
            wait_for_wake_word()
            mode = listen_with_timeout()
            if mode is None:
                continue
            if any(word in mode for word in ["quit", "exit", "stop"]):
                speak("Goodbye.")
                break
            run_mode(mode)
    except KeyboardInterrupt:
        speak("Exiting.")

```
**old voice mode changer**
```shell
import time
import object_detect_tts
import pitft_labeled_output
import weather_alerts
import gemini
from gemini import GeminiAssistant
import button_camera
import navigation
from navigation import NavigationMode

import speech_recognition as sr
import pyttsx3

# === TTS setup ===
tts = pyttsx3.init()
tts.setProperty("rate", 150)

def speak(text):
    print("🤖", text)
    tts.say(text)
    tts.runAndWait()

# === Speech Recognition setup ===
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_with_timeout(timeout=10):
    speak("I'm listening...")
    print("🎤 Listening for command...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Returning to idle.")
            return None

    try:
        text = recognizer.recognize_google(audio).lower()
        print("Heard:", text)
        return text
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
    except sr.RequestError as e:
        speak("Speech recognition is unavailable.")
        print("Error:", e)
    return None

def wait_for_wake_word(wake_word="hey glasses"):
    print("👂 Waiting for wake word...")
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                continue

        try:
            text = recognizer.recognize_google(audio).lower()
            print("Heard:", text)
            if wake_word in text:
                speak("Yes?")
                return
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print("Error with speech recognition:", e)
            continue

def run_mode(mode):
    if mode == "distance":
        speak("Entering distance mode.")
        object_detect_tts.run_distance_mode()

    elif mode == "object":
        speak("Entering object detection mode.")
        pitft_labeled_output.run_object_mode()

    elif mode == "weather":
        speak("Fetching weather alerts.")
        weather_alerts.run_weather_mode()

    elif mode == "gemini":
        speak("Opening Gemini Assistant.")
        from gemini import GeminiAssistant
        assistant = GeminiAssistant(api_key="YOUR_API_KEY")  # Replace with actual key

        while True:
            reply = assistant.ask_once()
            if reply == "exit":
                break
            if not reply:
                assistant.speak("Repeat that.")
            else:
                break  # Exit after one successful answer

    elif mode == "navigation":
        speak("Opening Navigation system.")
        navigation_mode = NavigationMode()
        navigation_mode.run()

    else:
        speak(f"I don't recognize the mode: {mode}")

if __name__ == "__main__":
    speak("Voice Mode Switcher is active.")
    try:
        while True:
            wait_for_wake_word()
            mode = listen_with_timeout()
            if mode is None:
                continue
            if any(word in mode for word in ["quit", "exit", "stop"]):
                speak("Goodbye.")
                break
            run_mode(mode)
    except KeyboardInterrupt:
        speak("Exiting.")

```
# Bill of Materials
<!--- Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. -->



| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Raspberry Pi 4 | Allows the interaction between the software and hardware for the camera and audio system | $58.99 | <a href="https://www.amazon.com/dp/B0CMZST24Y?ref=cm_sw_r_cso_cp_apin_dp_Y63HAE5CHH0DGMEAV253_1&ref_=cm_sw_r_cso_cp_apin_dp_Y63HAE5CHH0DGMEAV253_1&social_share=cm_sw_r_cso_cp_apin_dp_Y63HAE5CHH0DGMEAV253_1&starsLeft=1"> Link </a> |
| Camera Module | Taking videos for the object recognition | $14.49 | <a href="https://www.amazon.com/dp/B01ER2SKFS?ref=cm_sw_r_cso_cp_apin_dp_1CTZAM6W55YVQMMQNVET&ref_=cm_sw_r_cso_cp_apin_dp_1CTZAM6W55YVQMMQNVET&social_share=cm_sw_r_cso_cp_apin_dp_1CTZAM6W55YVQMMQNVET&starsLeft=1"> Link </a> |
| Raspberry Pi audio expansion board | Enables the connection between the Pi and the audio | $25.99 | <a href="https://www.amazon.com/dp/B0D12M8D2T?ref=cm_sw_r_cso_cp_apin_dp_8G3N848P2VK3FW7TT4BB_1&ref_=cm_sw_r_cso_cp_apin_dp_8G3N848P2VK3FW7TT4BB_1&social_share=cm_sw_r_cso_cp_apin_dp_8G3N848P2VK3FW7TT4BB_1&starsLeft=1"> Link </a> |
| Ear Piece | Allows user to hear the audio in ear | $10.95 | <a href="https://www.amazon.com/dp/B07YZ6LLDH?ref=cm_sw_r_cso_cp_apin_dp_04A8H7YD6WEPZYG52YZK&ref_=cm_sw_r_cso_cp_apin_dp_04A8H7YD6WEPZYG52YZK&social_share=cm_sw_r_cso_cp_apin_dp_04A8H7YD6WEPZYG52YZK&starsLeft=1"> Link </a> |
| Glasses | Allows user to wear the device | $5.99 | <a href="https://www.amazon.com/dp/B0BSF4PL2Q?ref=cm_sw_r_cso_cp_apin_dp_9G1KS2KBDABJZ2GR6485&ref_=cm_sw_r_cso_cp_apin_dp_9G1KS2KBDABJZ2GR6485&social_share=cm_sw_r_cso_cp_apin_dp_9G1KS2KBDABJZ2GR6485&starsLeft=1"> Link </a> |
| Wires | Enables everything to be connected to allow information to flow | $6.98 | <a href="https://www.amazon.com/dp/B01EV70C78?ref=cm_sw_r_cso_cp_apin_dp_CCVPR26GTMPF9EXHBMFR&ref_=cm_sw_r_cso_cp_apin_dp_CCVPR26GTMPF9EXHBMFR&social_share=cm_sw_r_cso_cp_apin_dp_CCVPR26GTMPF9EXHBMFR&starsLeft=1"> Link </a> |

# Other Resources/Examples
- [Resource 1] https://learn.adafruit.com/running-tensorflow-lite-on-the-raspberry-pi-4/initial-setup
- [Resource 2] https://www.youtube.com/watch?v=NPXBRX7N3ec
<!---


