# Deep Learning With Keras (O'Reilly) - MNIST CNN + Data Augmentation

This network builds on Network 2 by adding data augmentation, upscaling to 56x56, and tuning the regularisation. The architecture is the same CNN (two conv blocks → two dense layers → softmax), but with key changes: images upscaled from 28x28 to 56x56 to give augmentation more room to work, random rotation/translation/zoom on training images, ReLU instead of tanh in the dense layers, and reduced dropout (0.15 instead of 0.3).

## What I learned

- How data augmentation forces the network to learn position/rotation/scale-invariant features rather than memorising the specific framing of MNIST digits
- That augmentation acts as a regulariser itself — so heavy dropout on top of it can over-regularise (val accuracy exceeding train accuracy was the clue)
- The importance of normalising test data the same way as training data — forgetting `/ 255.0` on the test set caused a test loss of 9.6 despite 98% accuracy
- That ReLU in the dense layers trains faster than tanh for this architecture
- That aggressive augmentation on 28x28 images destroys too much information — upscaling to 56x56 gives the model more pixels to work with after transforms
- Finding the right augmentation strength is a balancing act — too mild and the model overfits to centred digits, too aggressive and it can't learn at all

## Changes from Network 2

- Upscaled input images from 28x28 to 56x56
- Added `RandomRotation(0.06)`, `RandomTranslation(0.3, 0.3)`, `RandomZoom((0.0, 0.7))` augmentation layers with `fill_mode="constant"`
- Switched dense layer activations from `tanh` to `relu`
- Reduced dropout from 0.3 to 0.15
- Increased EarlyStopping patience from 5 to 10
- Result: 98.7% test accuracy with -1.0% train/test gap — the negative gap shows the model handles clean test images better than the heavily augmented training images
