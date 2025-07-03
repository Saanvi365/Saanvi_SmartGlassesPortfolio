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

![Headstone Image](SaanviG.heic)

<!--
# Final Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/F7M7imOVGug" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For your final milestone, explain the outcome of your project. Key details to include are:
- What you've accomplished since your previous milestone
- What your biggest challenges and triumphs were at BSE
- A summary of key topics you learned about
- What you hope to learn in the future after everything you've learned at BSE



# Second Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/y3VAmNlER5Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For your second milestone, explain what you've worked on since your previous milestone. You can highlight:
- Technical details of what you've accomplished and how they contribute to the final goal
- What has been surprising about the project so far
- Previous challenges you faced that you overcame
- What needs to be completed before your final milestone 

# First Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/CaCazFBhYKs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For your first milestone, describe what your project is and how you plan to build it. You can include:
- An explanation about the different components of your project and how they will all integrate together
- Technical progress you've made so far
- Challenges you're facing and solving in your future milestones
- What your plan is to complete your project

# Schematics 
Here's where you'll put images of your schematics. [Tinkercad](https://www.tinkercad.com/blog/official-guide-to-tinkercad-circuits) and [Fritzing](https://fritzing.org/learning/) are both great resoruces to create professional schematic diagrams, though BSE recommends Tinkercad becuase it can be done easily and for free in the browser. 

# Code
Here's where you'll put your code. The syntax below places it into a block of code. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize it to your project needs. 

```c++
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println("Hello World!");
}

void loop() {
  // put your main code here, to run repeatedly:

}
```

# Bill of Materials
Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. 

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Raspberry Pi 4 | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Wires | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Glasses | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Ear Piece | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Camera Module | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |

# Other Resources/Examples
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Example 1](https://trashytuber.github.io/YimingJiaBlueStamp/)
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)

To watch the BSE tutorial on how to create a portfolio, click here.
-->
# Second Milestone
For my second milestone, I implemented object recognition using a pre-trained machine learning model and added text-to-speech (TTS) so the classification can be read aloud into an earpiece. I also set up a virtual environment to manage Python dependencies and isolate packages. I followed the tutorial "Running TensorFlow Lite Object Recognition on the Raspberry Pi 4 or Pi 5" by M. LeBlanc-Williams.
**How I did This:**
I started by setting up a virtual environment:
sudo apt install python3.11-venv
python3 -m venv env --system-site-packages
source env/bin/activate
Then, I connected and tested the Pi camera:
libcamera-hello
After verifying the camera worked,  I began by enabling the Pi Camera and installing dependencies:
sudo apt update
sudo apt install python3-pip
sudo pip3 install --upgrade setuptools
I then downloaded the TensorFlow Lite example repository:
git clone https://github.com/tensorflow/examples.git --depth 1
cd examples/lite/examples/object_detection/raspberry_pi
From there, I ran the setup script provided:
sh setup.sh
This script installed required packages and libraries such as OpenCV, TensorFlow Lite runtime, and various system dependencies needed for image processing and model inference.

Once everything was installed, I tested the camera again using:
libcamera-hello

**Challenge:**


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


**Challenges:**
When soldering together my starter project I came across two main challenges. FIrst I had a hard time recongize which screws where which whne finishing off my project and screwing the case together. THe other rpoblem I had was soldering the wires. I ended up cutting my wire too much meaning that the ends of my wires were rubber which you can't solder cocmpletly. I had to slowley cut out the rubber around the wire to reveal the metal part in order to solder properly. this took a long time since I had to make sure that i wouldnt accidently cut the whole wire off and have to restart. 

**Next:**
I'm excited to work on my main project which is smart glasses!




