# Deep Learning With Keras (O'Reilly) - MNIST CNN + Data Augmentation

This network builds on Network 2 by adding data augmentation and tuning the regularisation. The architecture is the same CNN (two conv blocks → two dense layers → softmax), but with three key changes: random rotation/translation/zoom on training images, ReLU instead of tanh in the dense layers, and reduced dropout (0.15 instead of 0.3).

## What I learned

- How data augmentation forces the network to learn position/rotation/scale-invariant features rather than memorising the specific framing of MNIST digits
- That augmentation acts as a regulariser itself — so heavy dropout on top of it can over-regularise (val accuracy exceeding train accuracy was the clue)
- The importance of normalising test data the same way as training data — forgetting `/ 255.0` on the test set caused a test loss of 9.6 despite 98% accuracy
- That ReLU in the dense layers trains faster than tanh for this architecture

## Changes from Network 2

- Added `RandomRotation(0.03)`, `RandomTranslation(0.05, 0.05)`, `RandomZoom(0.05)` augmentation layers
- Switched dense layer activations from `tanh` to `relu`
- Reduced dropout from 0.3 to 0.15
- Increased EarlyStopping patience from 5 to 10
- Result: 99.2% test accuracy (up from 99.1%) with 0.0% train/test gap (down from 0.5%)
