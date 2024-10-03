# Neural Style Transfer

**Neural Style Transfer (NST)** is a deep learning technique that combines the **content** of one image with the **style** of another, generating a new image that blends the two. This project leverages a pre-trained convolutional neural network (CNN) to achieve this.

---

## Introduction

This project demonstrates the use of deep learning to perform Neural Style Transfer (NST), a technique introduced by *Leon A. Gatys* in 2015. NST combines the **content** of a content image with the **style** of a different style image to generate a new, aesthetically compelling image.

The model uses a pre-trained **VGG19** convolutional neural network to extract features from both the content and style images. The loss function is optimized to minimize content loss (difference in content between the generated image and the content image) and style loss (difference in texture and color patterns between the generated image and the style image).

---

## How It Works

Neural Style Transfer works by:
1. Extracting **content features** from the content image using the deeper layers of a CNN.
2. Extracting **style features** from the style image using the shallower layers of the same CNN.
3. Iteratively optimizing an initial random image to minimize the content loss and style loss, producing an output that combines the content of the content image with the style of the style image.

---

![img1](https://github.com/SwamySaxena/nst-streamlit/blob/main/content_img.jpg)
![img2](https://github.com/SwamySaxena/nst-streamlit/blob/main/Mountain.jpg)
![img3](https://github.com/SwamySaxena/nst-streamlit/blob/main/output.jg.jpg)
