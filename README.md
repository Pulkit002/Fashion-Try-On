# Virtual Fashion Try-on
## Introduction
Our project aims at the transfer of a clothing item from its image onto a reference person, to virtually "try-on" the clothing item onto the person. The challenge here is generating photo-realistic images for simple or complex pose of the reference image of the person.
We plan on achieving this using a Content Generating and Preserving Network. We first predict the semantic layout of the person's image and then determining if the content needs to be preserved or generated based on the predicted semantic layout.
- Base Reference Paper : [Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content](https://doi.org/10.48550/arXiv.2003.05863)

## Objective
The objective of our project is to develop a visual try-on system based on the Adaptive Content Generating and Preserving Network (ACGPN). This system aims to transfer clothing from a target image onto a reference person while preserving clothing characteristics. By leveraging and modifying ACGPN, we seek to overcome challenges related to occlusions and complex poses in the reference image, ultimately generating photo-realistic try-on images with improved perceptual quality and rich clothing details.
![image](https://github.com/Pulkit002/Fashion-Try-On/assets/113465232/eceb8da5-dd12-4767-9e9c-4b30f534c5f3)
[Source](https://doi.org/10.48550/arXiv.2003.05863)
## Modules (Diagrams)

## Modules to be Done
1. Semantic Generation Module: Utilises semantic generation module, responsible for predicting the desired semantic layout after try-on .
- G1
- G2
2. Clothes Warping Module: responsible for warping the cloth images as per the generated mask.
3. Cloth Mask Inpainting: responsible for accurate non target body part regeneration.
4. Content Fusion Model: fuses the information from the first and second modules to generate accurate try-ons. 
