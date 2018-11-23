# Sign-Language-Recognition

## Problem

Social robotics invokes the ideal of interacting with robots, such as through speech. However, not everyone can rely on speech to communicate. In particular, communities that rely on sign language.

## Challenges

There are unique challenges involved with interacting with robots via sign. For instance, the difficulty of achieving high precision in sign language recognition where it may affect the interaction between people and robots. Furthermore, cultural differences in communication through sign versus speech might also influence how people interact with robots.

## Technologies, how they will interact/why now?

Today, the intersection of machine learning and computer vision techniques has enabled technologies that can recognise objects, shapes, and even gestures. However, to date the goal of social interaction with robotics eludes us.

By integrating depth perception capable cameras (e.g., RealSense, Kinect) and computer vision technology, the project will explore how we can use these technologies to observe and recognise hand movement through real-time camera frames with the view towards recognising signed language. Specifically, this will be implemented through a combination of OpenCV for computer vision, and Machine Learning libraries to provide relevant learning models.

## Questions we want to answer

The project will aim to address:
How to recognise the differences in various signed words, and gestures, which can change the meaning.
What are the current techniques for sign language and gesture recognition?
Which models and algorithms are most effective for various aspects of sign language recognition?
How to implement sign language recognition to enable users to interact with a robot?

## Goals/Implementation

In the early stages of the project, we aim to first utilise computer vision and machine learning techniques to distinguish relevant features of signed language and hand gestures vs irrelevant background features.

From there, we can use datasets and examples to demonstrate the feasibility of recognising a series of sign language gestures.

Implementation-wise, we aim to progressively explore:

- Recognising different simple gestures in real time
- Recognising different words, letters, and numbers through hand orientation, shape, and position
- Effectively recognising different words as a user transitions through a sentence.
- Cultural differences with interacting with a social robot through signed language, as opposed to spoken.
