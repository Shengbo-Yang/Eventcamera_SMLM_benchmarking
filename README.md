# Eventcamera_SMLM_benchmarking repository


## Introduction 

Event cameras, sometimes also referred as neuromorphic vision sensors, are fast asynchronous light detectors. Their pixels independently monitor light intensity change with microsecond level precision and return signals whenever the intensity changes exceed preset threshold. For further details about event camera, there is a nice review from [Gallego et al.](https://ieeexplore.ieee.org/document/9138762). There have been several implementations of event-based microscopy([Cabriel et al.](https://www.nature.com/articles/s41566-023-01308-8) ,[Ruipeng et al.](https://www.nature.com/articles/s41377-024-01502-5). ). However, there are still a many potentials of event camera left unexplored. This repository aims to provide benchmarking procedures and results for using event cameras as detectors for microscopy. 

## Event camera speed characterization 

There is no clear definition of the speed of event cameras as they work asycronously. We tried a imaging fluorescent beads excitied with modulated laser, or blinking beads. By comparing the modulation frequency and the retrived frequency from event-based data, we can have a idea of what the is highest 'pseudo frame rate' we can achieve using event camera. Frequency calculation algorithms are based on fast fourier transform (FFT) and frequency calculation algorithm from the camera manufacturer. 

## Event camera sensitivity characterization

What would be an major concern for event camera is its sensitivity. We know we can use it to see single molecules, but we don't have a quantified characterization of the event camera's response of different fluorescent dyes. In addition, the effects of configurable threshold and bias settings of event cameras are not closely investigated as well. 

The first set of experiments were designed to exam the full chip response of event camera, simliar to the procedure of characterizing a sCMOS camera. A backlight diffuse LED was used as light source to illuminate the event camera and a characterized sCMOS camera. The incident photons on both cameras were assumed to be the same, which were extracted from the sCMOS data. The corresponding event number was calculated from the event data and used to calculate 'how many photons are required to trigger a single postive or negative event. 

The second set of experiments (partly on-going now) exam the response of event camera of various common fluorescent dyes. Fluorescent dyes were diluted in polymers and excited by different laser power. There should be a cut-off point when the photon difference during which the molecules turn off is so small that the event camera can not detect them, giving us a guideline for what dyes and what excitations should we use for event-based SMLM imaging. 



