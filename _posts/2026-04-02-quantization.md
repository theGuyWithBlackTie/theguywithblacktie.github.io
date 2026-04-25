---
layout: post
title: 'Quantization: A Deep Dive into Model Compression'
date: 2026-04-02 11:00 +0530
math: true
description: 'An in-depth exploration of quantization techniques for model compression and efficiency.'
toc: true
image:
    path: assets/headers/dspy-header.png
    alt: Quantization
published: true
categories: [LLMs, Quantization]
tags: [optimization, quantization techniques]
---

# Introduction
Imagine you want to run Llama 2 70B on your machine. There's just one problem: in its native FP32 precision, the model weights alone take up roughly 280GB of RAM memory and additional memory of around 20GB for context which grows with sequence length. That's more than most high-end GPUs can handle, let alone a laptop.

Now what if you could shrink the moel down to 35GB or even 17GB withougb losing much of its quality?

That's what quanitzation does. It is the process of reducing the numerical precision of a model's weights and activations (for e.g. converting 32-bit floating point numbers to 8-bit integers) so that model gets smaller and its need less memory and compute to run. 

# Number Representation & Data Types
Before we can shrink a model, we need to understadn what we are shrinking. AI models internally performs mathematical operations on weights and activations, and how those parameters are stored determines both the precision & accuracy of the results and the memory model consumes.

Weights & activations are stored in <i>floating point</i> format, which has three components:
- **Sign bit**: Indicates whether the number is positive or negative.
- **Exponent**: Determines the range of the number (how large or small it can be).
- **Mantissa (or significand)**: Represents the precision of the number (how many decimal places it can have).

More bits means more room for the <i>exponent</i> and <i>mantissa</i>, which allows for a wider range of values and greater precision. For example, FP32 (32-bit floating point) has 1 sign bit, 8 bits for the exponent, and 23 bits for the mantissa, while FP16 (16-bit floating point) has 1 sign bit, 5 bits for the exponent, and 10 bits for the mantissa.

> <b>Precision</b> refers to the number of significant figures or decimal places a number can represent. In floating point numbers, precision is determined by the mantissa—more bits in the mantissa mean finer granularity and better accuracy for decimal values.
{: .prompt-info}

## Bits Spectrum
Here's how the common data types compare:

| Data Type | Total Bits | Sign Bits | Exponent Bits | Mantissa Bits | Range of Values | Memory Usage |
|-----------|------------|-----------|---------------|---------------|-----------------|--------------|
| FP32      | 32         | 1         | 8             | 23            | ~-3.40e38 to ~3.4e38 | 4 bytes
| FP16      | 16         | 1         | 5             | 10            | -65504 to 65504  | 2 bytes |
| BF16      | 16         | 1         | 8             | 7             | ~−3.38e38 to ~3.38e38 | 2 bytes |
| INT8      | 8          | 0         | 0             | 8             |  -128 to 127          | 1 byte |
| INT4      | 4          | 0         | 0             | 4             | -8 to 7           | 0.5 byte |
| INT2      | 2          | 0         | 0             | 2             | -2 to 1            | 0.25 byte |
| INT1      | 1          | 0         | 0             | 1             | -1 to 0           | 0.125 byte |

A few things to notice:
- <b>FP16</b> halves the memory usage of <b>FP32</b> but sacrifices both range and precision. This can cause overflow issues during training when gradients get very large.
- <b>BF16</b> is an interesting compromise: it keeps the same exponent (and thereof range) as <b>FP32</b> but has less precision than <b>FP16</b>. This makes it more effective for training because gradient magnitudes are preserved even if the values are slightly less precise.
- <b>INT8</b> and lower integer formats (INT4, INT2, INT1) are integer formats where there is no exponent, no mantissa, just whole numbers within a fixed rrange. They're much cheaper to compute with but can't represent the same nuance as floating point formats.

As you can see, as we reduce the number of bits, we also reduce the range and precision of the values we can represent. This is the trade-off that quantization makes: by using fewer bits, we can save memory and computational resources, but we may lose some accuracy in the process.

## How Memory Is Calculated
Eachh parameter is a model is stored as a number in a given data type, and each data type uses a fixed number of bits. So, the memory usage of a model can be calculated using the formula:
$$\text{Memory Usage} = \frac{\text{Number of Parameters} \times \text{Bits per Parameter}}{8} \text{ bytes}$$
Where:
- <b>Number of Parameters</b> is the total number of weights and biases in the model.
- <b>Bits per Parameter</b> is the number of bits used to represent each parameter (e.g., 32 for FP32, 16 for FP16, 8 for INT8, etc.).
- We divide by 8 to convert bits to bytes.

Let's take a 7 billion paramter model as an example:

| Data Type | Bits per Parameter | Calculation | Memory Usage (GB) |
|-----------|--------------------|-------------|-------------------|
| FP32      | 32                 | (7e9 * 32) / 8 | 28 GB |
| FP16      | 16                 | (7e9 * 16) / 8 | 14 GB |
| BF16      | 16                 | (7e9 * 16) / 8 | 14 GB |
| INT8      | 8                  | (7e9 * 8) / 8  | 7 GB |
| INT4      | 4                  | (7e9 * 4) / 8  | 3.5 GB |
| INT2      | 2                  | (7e9 * 2) / 8  | 1.75 GB |
| INT1      | 1                  | (7e9 * 1) / 8  | 0.875 GB |

As you can see, by reducing the precision of the data type, we can significantly reduce the memory usage of the model, especially for large models with billions of parameters, where the memory requirements can quickly become unmanageable.

## Seeing it in Code-Action

```python
import torch

model = torch.randn(1000,1000) # A simple tensor

print(f"FP32: {model.element_size() * model.nelement() / 1e6:.2f} MB")
print(f"FP16: {model.half().element_size() * model.nelement() / 1e6:.2f} MB")
print(f"INT8: {model.char().element_size() * model.nelement() / 1e6:.2f} MB")

# Outputs
FP32: 4.00 MB
FP16: 2.00 MB
INT8: 1.00 MB
```
In this example, we create a random tensor of shape (1000, 1000) which has 1 million elements. We then calculate the memory usage for FP32, FP16, and INT8 formats. As you can see, the memory usage decreases as we reduce the precision of the data type. Same tesnor, same shape but 4x less memory just by changing the data type. Now if we scale this to a model with billions of parameters, the memory savings become significant, allowing us to run larger models on hardware with limited resources.

## Quantization Techniques
In practice, we do not need to map the entire FP32 range [-3.4e38, 3.4e38] to the smaller range of INT8 [-128, 127]. Instead, we need to find a way to map <i>the range of our data (the model's paramters and activations)</i> to the smaller range of the target data type.

Symmetric & Asymmetric Quantization are two common techniques for doing this mapping and are forms of <i>linear mapping</i>.

### Symmetric Quantization
In symmetric quantization, the range of the original values is mapped to a symmetric range around zero in the quantized space. This means that the quantized value for zero in the original data type space is exactly zero in the quantized space.

![A diagram showing symmetric quantization mapping the range of original values to a symmetric range around zero in the quantized space.](assets/img/quantization/symmetric_quant_concept.png)

An example of symmetric quantization is <b>absolute maximum <i>(absmax)</i> quantization</b>, where given a list of values, the <i>highest</i> absolute value (<b>$\alpha$</b>) is taken as the range to perform the linear mapping, as shown in the diagram below:

![An illustration of absolute maximum (absmax) quantization, where the scale factor is determined by the maximum absolute value in the data.](assets/img/quantization/symmetric_quantization_example.png)

Among the given list of values, $10.8$ is the highest absolute value, so $\alpha$ is set to $10.8$ and while quantizing the <i>FP-32</i> to <i>INT-8</i>, $10.8$ will be mapped to $127$ and $-10.8$ will be mapped to $-127$ while maintianing symmetry around $0$.

#### Symmetric Quantization Algorithm
Since it is a linear mapping centered around zero, the process of quantization is straightforward.

We first calculate the scale factor ($s$) which determines how much we need to scale down the original values to fit into the range of the target data type. The formula for calculating the scale factor in symmetric quantization is:
$$S = \frac{2^{b-1} - 1}{\alpha}$$
Where:
- $s$ is the scale factor.
- $b$ is the number of bits in the target data type (e.g., 8 for INT8).
- $\alpha$ is the maximum absolute value in the original data (e.g., the weights or activations).

Then, we can quantize the original value ($x$) to the quantized value ($q$) using the formula:
$$X_{quantized} = \text{round}({S}\times{X})$$

Filling in the values would give us:
$$S = \frac{2^{8-1} - 1}{10.8} = \frac{127}{10.8} \approx 11.76$$
$$X_{quantized} = \text{round}(11.76 \times X)$$