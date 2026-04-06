# Grokking-MNIST Notes

This network was built while working through the book *Grokking Deep Learning* by Andrew Trask. It is a fully connected network built from scratch using only NumPy — no PyTorch, TensorFlow, or high-level ML frameworks.

## What I learned

- How forward propagation works layer by layer
- How backpropagation computes deltas and updates weights using gradient descent
- Why dropout helps prevent overfitting by forcing the network to generalise
- How softmax converts raw scores into probabilities that sum to 1
- The difference between training accuracy and test accuracy, and why the gap between them matters

## Things to try next

- Add a convolutional layer to improve accuracy further
- Experiment with different learning rates and hidden layer sizes
- Try training on Fashion-MNIST to see how well the same architecture generalises
