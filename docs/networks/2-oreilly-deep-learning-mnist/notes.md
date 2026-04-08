# Deep Learning With Keras (O'Reilly) - MNIST CNN

This network is the Keras counterpart to the from-scratch NumPy MNIST notebook (Network 1). Same dataset, same task, but built with TensorFlow/Keras instead of writing forward and backward propagation by hand — and this time using a small convolutional network rather than a plain MLP.

## What I learned

- How convolutional layers exploit local image structure instead of treating pixels as independent features, and why they beat a dense-only network on the same task
- How `MaxPooling2D` shrinks the spatial dimensions between conv blocks to keep parameter counts manageable
- Why dropout belongs *between* the dense layers (not just at the end) and the role it plays in closing the train/test gap
- How `validation_split` gives you an honest generalisation signal *during* training, without touching the test set
- How `EarlyStopping` uses that validation signal to halt training automatically once `val_loss` plateaus, preventing overfitting without having to hand-tune the epoch count
- The difference between "epochs" and "steps per epoch" in Keras (steps = batches, not images)
- How to export a trained Keras model to TensorFlow.js format (`model.save` → `tensorflowjs_converter` → `tf.loadLayersModel` in the browser), which is how this page can run the same trained network client-side

## Things to try next

- Add `BatchNormalization` layers after each `Conv2D` to see how much faster it converges
- Try simple data augmentation (random rotation, translation) and see if it closes the remaining ~0.5% train/test gap
- Add a third `Conv2D` + `MaxPooling2D` block to see how much deeper the network can usefully go on 28×28 inputs
- Swap the dense-head `tanh` activations for `relu` and compare
- Retrain on Fashion-MNIST using the exact same architecture as a generalisation test
