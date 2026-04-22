# Waste Classifier for Raspberry Pi 4

This project is a Raspberry Pi 4 based waste classification system. A PIR motion sensor detects movement, the Raspberry Pi camera captures an image, a TensorFlow Lite model classifies the waste item, and the system responds through LED indicators and a simple dashboard.

The system is built as a closed-loop IoT solution where AI output triggers meaningful actions:
- image capture
- classification with a `.tflite` model
- LED feedback on the breadboard
- live information shown in a Flask dashboard

## Main features

- Motion detection using a PIR sensor
- Image capture using **Pi Camera Module 2**
- Waste classification using **TensorFlow Lite**
- LED indication for class result
- Flask dashboard for system status and results
- Raspberry Pi based hardware setup

## Hardware required

The project uses the following hardware:

- **Raspberry Pi 4 Model B**
- **Pi Camera Module 2**
- **PIR motion sensor**
- **Breadboard**
- **4 LEDs**
- **4 resistors** (typically 220Ω to 330Ω)
- **Jumper wires**
- **MicroSD card with Raspberry Pi OS**
- **Power supply for Raspberry Pi**
- Optional: monitor, keyboard and mouse for setup/debugging

## Hardware setup

### 1. LED wiring

The breadboard setup in the Fritzing illustration shows the LED part of the circuit.

- Each LED is connected to a separate GPIO pin
- Each LED uses a resistor in series
- All LEDs share a common ground rail
- The Raspberry Pi ground is connected to the breadboard ground rail

Example GPIO setup used in the project:

| Component | Raspberry Pi connection |
|---|---|
| LED 1 | GPIO22 |
| LED 2 | GPIO23 |
| LED 3 | GPIO24 |
| LED 4 | GPIO25 |
| Common LED ground | GND |

> Important: If your Python code uses different GPIO pins, update this table so the README matches the code.

### 2. PIR sensor wiring

The PIR sensor is **not shown in the Fritzing image** because the exact component was not available in the library.

Typical PIR connection used in this project:

| PIR pin | Raspberry Pi connection |
|---|---|
| VCC | 5V |
| OUT | GPIO17 |
| GND | GND |

The PIR sensor is responsible for detecting movement and triggering the capture/classification pipeline.

### 3. Camera setup

The **Pi Camera Module 2** is also **not included in the Fritzing illustration**.

It is connected directly to the Raspberry Pi using the **CSI camera port**, not through the breadboard.

- Connect the ribbon cable to the Raspberry Pi camera connector
- Make sure the cable is inserted in the correct direction
- Test the camera before running the full project

## About the hardware illustration

The Fritzing diagram is useful for showing the breadboard and LED connections, but it does **not** include all physical components used in the final system.

For the report and GitHub documentation, a good approach is:

1. Use the Fritzing image to document the **LED and GPIO wiring**
2. Add one or more **real photos** of the full setup
3. Mention that the **Pi Camera Module 2** and **PIR sensor** are not shown in the Fritzing file because matching parts were unavailable there

Suggested wording for the report:

> The Fritzing illustration documents the LED indication circuit and GPIO wiring on the breadboard. The Pi Camera Module 2 and the PIR motion sensor were not available in the Fritzing component library used for the drawing, and are therefore documented using physical photos and written hardware descriptions instead. The camera is connected through the Raspberry Pi CSI port, while the PIR sensor is connected separately to power, ground and a GPIO input pin.

## Software required

This project is intended to run on **Raspberry Pi OS** with Python installed.

You need:
- Python 3
- `pip`
- `venv` (recommended)
- Flask
- NumPy
- Pillow
- OpenCV
- `gpiozero` or the GPIO library used in your code
- TensorFlow Lite support (`tflite-runtime` or equivalent)
- Camera support on Raspberry Pi (`rpicam-*`, `libcamera-*`, `picamera2`, or whatever your code uses)

## Recommended project structure

A typical structure for this project can look like this:

```text
waste-classifier/
├── app.py
├── main.py
├── waste_classifier.tflite
├── labels.txt
├── static/
├── templates/
├── captures/
├── logs/
└── README.md
```

### File description

- **app.py**: Flask dashboard for showing system status, results and logs
- **main.py**: main program that runs the hardware + AI pipeline
- **waste_classifier.tflite**: trained TensorFlow Lite model
- **labels.txt**: class labels used by the model
- **captures/**: saved images from the camera
- **logs/**: optional log files for debugging and traceability

> If your main file has another name, such as `cocoversjon.py`, replace `main.py` in the commands below.

## Installation

### 1. Update the Raspberry Pi

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

Example:

```bash
pip install flask numpy pillow opencv-python gpiozero
```

For TensorFlow Lite support, install the option that matches your environment:

```bash
pip install tflite-runtime
```

If your code uses TensorFlow instead of `tflite-runtime`, install the version that matches your Raspberry Pi OS and Python setup.

### 4. Install Raspberry Pi camera support

Depending on the OS and how your code captures images, you may also need Raspberry Pi camera tools.

Example:

```bash
sudo apt install -y python3-picamera2
```

Some projects instead rely on `rpicam-still` or older `libcamera` commands.

## Testing hardware before running the project

### Test the camera

Depending on your Raspberry Pi OS version, test with one of these:

```bash
rpicam-hello
```

or

```bash
libcamera-hello
```

### Test the PIR sensor and LEDs

Before running the full project, it is a good idea to test:
- whether the PIR sensor output changes on motion
- whether each LED can turn on and off from Python
- whether the GPIO pin mapping in code matches the physical wiring

## How to run the project

The project uses two Python files:

### 1. Start the dashboard

```bash
python app.py
```

This starts the Flask dashboard.

By default, Flask often uses:

```text
http://127.0.0.1:5000
```

If port 5000 is already in use, change the port in `app.py` and run it on another port, for example 5001.

### 2. Start the main waste-classifier program

Open a second terminal and run:

```bash
python main.py
```

This starts the full pipeline:
- wait for motion from PIR sensor
- capture image from camera
- preprocess image if needed
- run TensorFlow Lite classification
- light the correct LED
- send result/status to dashboard

## Typical run flow

1. PIR sensor detects motion
2. Camera captures an image
3. Model predicts the waste class
4. Matching LED lights up
5. Dashboard shows the current status/result
6. Logs can be stored for debugging and traceability

## Notes about TensorFlow Lite

The project depends on a trained `.tflite` model.

Make sure these files are available in the correct location:
- `waste_classifier.tflite`
- `labels.txt`

If the model is missing, the classifier will not work.

## Troubleshooting

### Camera not detected
- Check CSI ribbon cable connection
- Reseat the cable carefully
- Test with `rpicam-hello` or `libcamera-hello`

### Dashboard does not start
- Make sure Flask is installed
- Check whether the chosen port is already in use

### GPIO does not work
- Verify the pin numbers in code
- Verify ground wiring
- Check LED direction and resistor placement

### PIR sensor reacts too slowly
- Many PIR modules have adjustable potentiometers for sensitivity and delay
- Reduce the delay if the sensor stays active too long after motion stops

### TFLite install issues
- `tflite-runtime` can be version-sensitive on Raspberry Pi
- Use the version that matches your OS and Python version
- If needed, switch to the TensorFlow-based inference path supported by your project

## Suggested images for GitHub and report

A good documentation package is:

- **Fritzing diagram** for LED/breadboard/GPIO wiring
- **Real photo of the Raspberry Pi + breadboard**
- **Real photo of the PIR sensor**
- **Real photo of the Pi Camera Module 2 and CSI cable**
- **Screenshot of the dashboard**

This gives both a clean technical overview and proof of the real physical setup.

## Summary

To run this project, you need:

- Raspberry Pi 4 hardware connected correctly
- Pi Camera Module 2 connected through CSI
- PIR sensor connected to GPIO input
- 4 LEDs with resistors connected to GPIO output pins
- Python environment with Flask, GPIO libraries and TensorFlow Lite support
- `app.py` for the dashboard
- `main.py` for the classification pipeline
- a valid `.tflite` model and labels file

