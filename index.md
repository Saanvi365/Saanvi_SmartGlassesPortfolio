# Smart Glasses
I built a pair of smart glasses that help visually impaired users navigate their environment by recognizing objects in real time and providing spoken feedback. Using a Raspberry Pi with a camera, machine learning, and text-to-speech technology, the glasses detect and announce nearby objects. The biggest challenges were setting up the hardware-software integration and fine-tuning the object recognition to be both accurate and timely, but overcoming these hurdles has made the project highly impactful.
<!--
You should comment out all portions of your portfolio that you have not completed yet, as well as any instructions:

<!--- This is an HTML comment in Markdown -->
<!--- Anything between these symbols will not render on the published site -->



| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Saanvi G | Leland | Electrical/Computer Engineering | Incoming Senior

**Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help.**

![Headstone Image](SaanviG.svg)
<img src="SaanviG.hsvg" alt="Headshot" height="400" width="500">

-->
# Third Milestone

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
<img src="retro.svg" alt="finished product" height="400" width="500">

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




