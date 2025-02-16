# KlawTreeHacks2025
Seamless laptop control through hand gestures, facial expressions, and voice recognition, allowing users to navigate without a keyboard or mouse

## Inspiration
Klaw was inspired by the need for more intuitive, hands-free ways to interact with technology, especially for individuals with mobility or speech impairments.  As technology becomes increasingly integrated into daily life, traditional input methods like keyboards and mice are sometimes insufficient for the nuanced interactions that modern applications demand. We wanted to create an accessibility tool that bridges the gap between users and their devices, allowing the former to take a more relaxed, agile approach towards making the most of their laptops.

## What it does
Klaw enables seamless laptop control through hand gestures, facial expressions, and voice recognition, allowing users to navigate without a keyboard or mouse. It specifically supports features like gesture-based clicking, head-tilt volume control, and real-time speech-to-text captions.

Hand Gesture Control: Move your cursor by pointing with your index finger, perform clicks by pinching your fingers together, and navigate with swipe gestures.

Facial Expression Detection: Tilt your head left or right to adjust volume, raise your eyebrows to trigger actions, or smile to confirm selections.

Real-Time Speech-to-Text: Converts spoken words into live subtitles, aiding users with speech impairments or those in environments where typing is difficult.

Accessible Interaction: Klaw eliminates the need for a physical mouse or keyboard, making laptops more accessible for users with limited mobility.

## How we built it
We used Google MediaPipe for hand and face tracking, OpenCV for image processing, SpeechRecognition for real-time subtitles, Numpy for the joint angle calculations, and Pygame to render the test interface. The project was entirely done in Python. 

## Challenges we ran into
Fine-tuning gesture detection to avoid false positives.

Handling speech recognition delays and improving accuracy in noisy environments.

Preventing over-sensitive cursor clicking when user pinches.

Integrating the real-time angle calculations into the PyGame simulation

## Accomplishments that we're proud of
Successfully integrating multi-modal input (gesture, face, and voice) into a single tool.

Improving gesture accuracy for seamless, hands-free control.

Making an accessible and already functional prototype that can genuinely help users right away.

## What we learned
How to apply CV libraries in order to track body movements
  
How to fine-tune thresholds for gesture recognition to balance sensitivity with accuracy
  
Challenges in live speech-to-text processing and how to improve clarity

## What's next for Klaw
Some type of customizable gestures for more personalized controls

Machine learning enhancements to improve speech-to-text accuracy

Less sensitive mouse controls
