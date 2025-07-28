ASR Training Dashboard

A comprehensive, user-friendly desktop application for training and testing a Transformer-based Automatic Speech Recognition (ASR) model. Built with PyQt6 and TensorFlow, this tool provides an all-in-one interface for managing the end-to-end ASR workflow, from data preparation to real-time training, monitoring, and inference.

(Replace the link above with a screenshot of your application)

✨ Features

Intuitive Graphical User Interface: A clean and modern UI built with PyQt6, featuring multiple themes (Charcoal, Nord, Solarized, and more).

End-to-End Workflow: Manage everything from selecting data paths to launching and monitoring training, all within one application.

Real-Time Monitoring:

Live Log Viewer: Watch training progress, validation loss, and sample predictions as they happen.

Integrated TensorBoard: A built-in tab launches and embeds TensorBoard for in-depth visualization of model metrics and performance.

Responsive and Non-Blocking: The entire training process runs in a background thread, ensuring the GUI remains fully responsive.

Powerful ASR Model: Implements a state-of-the-art Transformer architecture for robust speech recognition.

Hyperparameter Control: Easily tweak model parameters like hidden dimensions, attention heads, dropout rate, and batch size directly from the GUI.

Training Management:

Start/Stop Control: Start, stop, and restart the training process at any time.

Early Stopping: Automatically halts training when the model stops improving to prevent overfitting.

Checkpointing: Automatically saves the best-performing model weights during training.

Inference Tab: Load a previously trained model and transcribe any single audio file to quickly test its performance.

Configuration Management: Save and load your hyperparameter configurations to a JSON file for reproducible experiments.

Persistent Settings: The application remembers your data paths, window size, and selected theme between sessions.

⚙️ Requirements

To run this application, you will need Python 3.8+ and the following libraries:

PyQt6

PyQt6-WebEngine

tensorflow

keras (usually included with TensorFlow)

librosa (for audio processing, although this script uses tf.audio)

You can install all dependencies using the provided requirements.txt file.

🛠️ Setup and Installation

Clone the Repository

Generated bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO


Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

Generated bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install Dependencies
Create a requirements.txt file with the following content:

Generated code
pyqt6
pyqt6-webengine
tensorflow
numpy
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Then, install them using pip:

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Note: Depending on your system and whether you have a compatible GPU, you might want to install a specific version of TensorFlow (e.g., tensorflow[and-cuda] for NVIDIA GPUs).

Prepare Your Data
The model expects a specific data format. See the Data Preparation section below for details.

🎧 Data Preparation

For the application to work correctly, your data must be structured as follows:

Audio Files:

All audio files must be in WAV format.

They must have a sample rate of 8000 Hz.

Place all your .wav files into a single folder (e.g., C:\data\lj_speech\wavs).

Transcript File:

You need a single text file, typically named transcript.txt.

This file maps the filename of each audio clip (without the extension) to its transcription.

The format should be a tuple-like string on each line: ("filename", "this is the transcription.")

Example transcript.txt:

Generated code
("lj_speech_0001", "this is the first audio file")
("lj_speech_0002", "this is another sentence for the second file")
("lj_speech_0003", "and a final example transcription")
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
🚀 How to Use

Launch the Application
Run the Python script from your terminal:

Generated bash
python your_script_name.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Configure Data Paths

In the Data Paths section, click "Browse..." to select your WAV file directory and your transcript.txt file.

(Optional) Select a location to save the final model using the "Model Save Path" field.

Adjust Hyperparameters

Modify any of the settings in the Hyperparameters section on the left panel.

Alternatively, go to File > Load Configuration to load settings from a .json file.

Start Training

Click the "Start Training" button at the bottom right.

The training process will begin in the background.

Monitor Progress

Live Logs Tab: Watch the command-line output, including epoch progress, validation loss, and learning rate.

TensorBoard Tab: After a minute, TensorBoard will load. Use it to track metrics like loss curves and graph visualizations.

Progress Bar: The progress bar at the bottom shows the completion of the current epoch.

Stop Training

Click the "Stop Training" button at any time. The process will finish the current batch and then gracefully shut down.

Test the Model (Inference)

Once training is complete and a model has been saved, navigate to the Inference tab.

Click "Load Model" and select the saved .keras model directory.

Click "Select Audio" to choose a single .wav file (must be 8kHz).

Click "Transcribe" to see the model's prediction.

📂 Code Structure

The script is organized into several logical parts for clarity and maintainability:

Part 1: PyQt6 Threading and Communication: Defines the QObject signals and custom logging handlers for safe communication between the training thread and the main GUI thread.

Part 2: Full ASR Model and Data Processing: Contains the complete TensorFlow/Keras implementation of the Transformer model, data loading functions (get_data), audio preprocessing (path_to_audio), and the learning rate schedule.

Part 3: The Background Worker: The TrainingWorker class encapsulates the entire training lifecycle. It runs on a separate QThread to avoid freezing the GUI.

Part 4: Main PyQt6 GUI Application: The ASRTrainingGUI class defines the main window, all its widgets, menus, themes, and connects user actions (button clicks) to the corresponding functions.

🎨 Customization

Themes: Change the application's look and feel by navigating to View > Themes and selecting from Charcoal, Arctic, Solarized, Nord, and more.

Save/Load Configurations: Use the File menu to save your current hyperparameter setup to a .json file or load a previous one. This is perfect for experiment tracking.

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
