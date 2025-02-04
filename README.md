# Secure Multi-Party Computation and CrypTen

---

## Agenda

1. **Introduction**
2. **What is Secure Multi-Party Computation (MPC)?**
3. **Overview of CrypTen**
4. **MPC Techniques in CrypTen**
5. **Project Overview: Secure Logistic Regression**
6. **Data Preparation in MPC**
7. **Model Training with CrypTen**
8. **Secure Inference**
9. **Conclusion**
10. **References**

---

## Introduction

- The rise of data privacy concerns necessitates secure ways to perform computations on private data.
- **Secure Multi-Party Computation (MPC)** allows multiple parties to jointly compute a function over their inputs while keeping those inputs private.
- **CrypTen** is a framework built on PyTorch that enables easy implementation of MPC techniques for machine learning.

---

## What is Secure Multi-Party Computation (MPC)?

- **Definition**: MPC allows parties to compute a function over their inputs without revealing the inputs themselves.
- **Goal**: Ensure that no more information is revealed than what can be inferred from the output.
- **Applications**:
  - Privacy-preserving data analytics.
  - Secure training of machine learning models.
  - Collaborative computations among untrusted parties.

---

## Key Concepts in MPC

- **Secret Sharing**:
  - Data is split into shares distributed among parties.
  - No single party holds enough information to reconstruct the original data.
- **Computation on Shares**:
  - Parties perform operations on their shares.
  - Results are combined to obtain the final output without revealing individual inputs.

---

## Overview of CrypTen

- **CrypTen** is a research framework for secure, privacy-preserving machine learning.
- **Built on PyTorch**: Leverages PyTorch's APIs and functionalities.
- **Features**:
  - Easy-to-use API similar to PyTorch.
  - Support for common neural network modules.
  - Secure computation protocols implemented under the hood.

---

## Advantages of CrypTen

- **Machine Learning First**: Designed with ML practitioners in mind.
- **Eager Execution**: Immediate computation results, similar to PyTorch.
- **Interoperability**: Compatibility with existing PyTorch models and datasets.
- **Flexible and Extensible**: Supports custom models and functions.

---

## MPC Techniques in CrypTen

- **Arithmetic Secret Sharing**:
  - Data is represented as shares such that the sum of shares reconstructs the original value.
  - Used for operations like addition and multiplication.
- **Beaver Triples**:
  - Pre-shared random values that facilitate secure multiplication.
  - Allow parties to compute products without revealing operands.
- **Secure Non-Linear Computations**:
  - Approximations for functions like sigmoid, softmax, and logarithms.
  - Use iterative methods suitable for MPC.

---

## Arithmetic Secret Sharing in CrypTen

- **Additive Sharing**:
  - Each party holds a share \( [x]_i \) such that \( x = \sum_i [x]_i \).
- **Operations**:
  - **Addition**: Parties add their shares locally.
  - **Multiplication**: Uses Beaver triples to securely compute products.

---

## Beaver Triples Explained

- **Purpose**: Enable secure multiplication of secret-shared values.
- **Process**:
  1. Parties have \( [a] \), \( [b] \), \( [c] \) where \( c = a \times b \).
  2. Compute \( [e] = [x] - [a] \) and \( [f] = [y] - [b] \).
  3. Reconstruct \( e \) and \( f \) securely.
  4. Compute \( [z] = [c] + e[b] + f[a] + ef \).

---

## CrypTen's Approach to Non-Linear Functions

- Uses secure approximations for functions that are not natively supported in MPC.

- **Examples**:
  - **Sigmoid**: Computed using the secure exponential function and division.
  - **Logarithm**: Approximated using iterative methods like Newton-Raphson.

---

## Project Overview: Secure Logistic Regression

- **Objective**: Train a logistic regression model over data held privately by multiple parties.
- **Dataset**: Image data (similar to Fashion-MNIST) distributed among 3 parties.
- **Tasks**:
  - Prepare and encrypt data from each party.
  - Combine data securely without revealing individual data.
  - Train the model using CrypTen's secure computation protocols.
  - Evaluate the model on a test set.

---

## Data Preparation in MPC

- **Data Distribution**:
  - Each party holds its own private dataset.
  - Data is not shared in plaintext with other parties.
- **Processing Steps**:
  1. Each party normalizes its data locally.
  2. Data is encrypted using secret sharing.
  3. Encrypted data is saved and ready for computation.

---

## Secure Data Loading with CrypTen

- **Encrypting Data**:
  - Use `crypten.save_from_party()` to save encrypted data.
  - Specify the source party to ensure data origins are known.
- **Combining Data**:
  - Parties collaboratively load encrypted data from others using `crypten.load_from_party()`.
  - Data is combined securely, maintaining privacy.

---

## Model Training with CrypTen

- **Model Definition**:
  - Logistic Regression model defined using `crypten.nn.Module`.
  - Consists of a single linear layer suitable for multi-class classification.
- **Training Process**:
  - Loss computed using `crypten.nn.CrossEntropyLoss()`.
  - Gradients computed securely; parameters updated with `model.update_parameters()`.
- **MPC in Training**:
  - All computations are performed on encrypted data and model parameters.
  - Uses arithmetic secret sharing and Beaver triples.

---

## Secure Inference

- **Testing the Model**:
  - Decrypted model can be used for standard inference.
  - For secure inference, encrypt both the model and the test data.
- **Process**:
  1. Encrypt the trained model.
  2. Encrypt the test data.
  3. Perform inference using CrypTen's secure computation.
  4. Decrypt the predictions for evaluation.

---

## Results and Evaluation

- **Accuracy**:
  - Model evaluated on the test set.
  - Comparable performance to non-MPC training.
- **Privacy Preservation**:
  - Data from each party remains private throughout the process.
  - No sensitive information is leaked during training or inference.

---

## Conclusion

- **CrypTen** enables secure machine learning using MPC techniques without requiring deep cryptographic expertise.
- **Practical Implementation**:
  - Similar API to PyTorch makes adoption easier.
  - Secure computations are abstracted away, allowing focus on model development.
- **Future Work**:
  - Extend to more complex models and datasets.
  - Explore performance optimizations and scalability.

---

## References

1. **CrypTen: Secure Multi-Party Computation Meets Machine Learning**
   - Knott et al., NeurIPS 2021.
   - [Paper Link](https://proceedings.neurips.cc/paper/2021/file/2754518221cfbc8d25c13a06a4cb8421-Paper.pdf)
2. **CrypTen GitHub Repository**
   - [https://github.com/facebookresearch/crypten](https://github.com/facebookresearch/crypten)
3. **Secure Multi-Party Computation (MPC)**
   - Yao, A. C. (1986). How to generate and exchange secrets.
   - [Link](https://dl.acm.org/doi/10.1145/138927.138930)
4. **Beaver Multiplication Protocol**
   - Beaver, D. (1991). Efficient Multiparty Protocols Using Circuit Randomization.
   - [Link](https://link.springer.com/chapter/10.1007/0-387-34805-0_21)

---
