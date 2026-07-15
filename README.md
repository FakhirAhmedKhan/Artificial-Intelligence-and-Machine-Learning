



# Lesson 1: Introduction to Artificial Intelligence and Machine Learning

**Progress:** Lesson 1 in progress  
**Prerequisites:** Basic programming knowledge  
**Tools:** Python only—no external libraries yet

---

## 1. Learning Objectives

By the end of this lesson, you should be able to:

1. Define Artificial Intelligence, Machine Learning, and Deep Learning.
2. Explain how traditional programming differs from machine learning.
3. Identify major categories of machine learning.
4. Understand training, models, features, labels, and predictions.
5. Build a basic program that learns a decision rule from data.
6. Recognize situations where machine learning is unnecessary or unsuitable.

---

## 2. Real-World Example First

Imagine an email application that must identify spam.

### Traditional programming

A programmer writes rules:

```text
IF email contains "win money"
OR email contains "free prize"
OR sender looks suspicious
THEN mark as spam
```

This can work, but spammers change their wording. The developer must continuously add new rules.

### Machine learning

Instead of manually writing every rule, we provide thousands of examples:

```text
Email                                Correct answer
---------------------------------------------------
"Congratulations, claim your prize"  Spam
"Your meeting starts at 10 AM"       Not spam
"Get rich immediately"               Spam
"Project report attached"            Not spam
```

The machine-learning algorithm examines these examples and learns patterns that distinguish spam from normal email.

The important distinction is:

> Traditional programming receives rules written by humans. Machine learning discovers useful rules from data.

---

## 3. Theory

## Artificial Intelligence

Artificial Intelligence, or AI, is the broad field of building computer systems that perform tasks normally associated with human intelligence.

Examples include:

- Understanding language
- Recognizing images
- Making decisions
- Planning routes
- Playing games
- Recommending products
- Generating text or images

Not every AI system uses machine learning. A chess program based entirely on manually written rules can still be considered AI.

---

## Machine Learning

Machine Learning is a subfield of AI in which systems learn patterns from data instead of having every decision rule explicitly programmed.

A machine-learning system generally follows this process:

```text
Data → Learning algorithm → Trained model → Predictions
```

Examples:

- Predicting house prices
- Detecting fraud
- Classifying emails
- Recommending movies
- Predicting customer churn
- Recognizing handwritten digits

---

## Deep Learning

Deep Learning is a subfield of machine learning based on artificial neural networks with multiple processing layers.

Deep learning is especially effective for:

- Images
- Audio
- Video
- Natural language
- Large and complex datasets

Examples include:

- Face recognition
- Speech recognition
- Object detection
- Large language models
- Autonomous-driving perception systems

---

## Relationship Between AI, ML, and Deep Learning

```text
+------------------------------------------------+
|            Artificial Intelligence             |
|                                                |
|    +--------------------------------------+    |
|    |          Machine Learning            |    |
|    |                                      |    |
|    |    +----------------------------+    |    |
|    |    |       Deep Learning        |    |    |
|    |    +----------------------------+    |    |
|    +--------------------------------------+    |
+------------------------------------------------+
```

Therefore:

```text
Deep Learning ⊂ Machine Learning ⊂ Artificial Intelligence
```

Every deep-learning system is a machine-learning system, but not every machine-learning system uses deep learning.

---

## 4. Traditional Programming vs Machine Learning

### Traditional programming

```text
Input data + Human-written rules → Output
```

Example:

```python
age = 20

if age >= 18:
    print("Adult")
else:
    print("Minor")
```

The programmer already knows the rule: age greater than or equal to 18 means adult.

### Machine learning training

```text
Input data + Correct outputs → Learning algorithm → Model
```

### Machine learning prediction

```text
New input + Trained model → Predicted output
```

A model is the learned representation of patterns found in data.

---

## 5. Core Machine-Learning Vocabulary

### Dataset

A dataset is a collection of examples used for analysis or learning.

Example:

| Study hours | Attendance | Passed |
|---:|---:|---|
| 1.0 | 55% | No |
| 2.0 | 65% | No |
| 3.5 | 80% | Yes |
| 5.0 | 90% | Yes |

### Features

Features are the input values used to make a prediction.

In the table:

```text
Study hours
Attendance
```

are features.

Features are often represented as \(X\).

### Label or Target

The label is the correct answer that the model is expected to predict.

In the table:

```text
Passed
```

is the label.

Labels are commonly represented as \(y\).

### Model

A model is a mathematical or computational function that transforms features into predictions.

```text
Features → Model → Prediction
```

### Training

Training is the process of adjusting a model so that it performs well on known examples.

### Inference

Inference means using a trained model to make a prediction for new data.

### Prediction

The output produced by the model.

---

## 6. Major Types of Machine Learning

## Supervised Learning

The training data contains both inputs and correct answers.

```text
Input + Correct answer
```

Example:

| House size | Price |
|---:|---:|
| 1,000 sq ft | $200,000 |
| 1,500 sq ft | $280,000 |
| 2,000 sq ft | $370,000 |

The model learns to predict price from house size.

Common supervised-learning tasks:

- Regression
- Classification

### Regression

Predicts a continuous numerical value.

Examples:

- House price
- Temperature
- Monthly sales
- Delivery time

### Classification

Predicts a category.

Examples:

- Spam or not spam
- Fraud or legitimate
- Sick or healthy
- Cat, dog, or bird

---

## Unsupervised Learning

The data contains inputs but no provided correct answers.

```text
Input data without labels
```

The system attempts to discover hidden structures or groups.

Example:

```text
Customer purchasing data → Groups of similar customers
```

Common tasks include:

- Clustering
- Dimensionality reduction
- Pattern discovery

---

## Reinforcement Learning

An agent interacts with an environment and learns through rewards and penalties.

```text
Agent → Action → Environment → Reward → Agent improves
```

Examples:

- Game-playing systems
- Robot navigation
- Resource allocation
- Some autonomous-control systems

---

## 7. Visual Explanation of Training

Suppose we want to predict whether a student will pass based on study hours.

```text
Training data
     |
     v
+----------------------+
| Learning algorithm   |
| Tries possible rules |
+----------------------+
     |
     v
Learned model:
"Predict pass when study hours >= 2.75"
     |
     v
New student studies 4 hours
     |
     v
Prediction: Pass
```

The model does not understand education like a human. It finds a mathematical pattern that performs well on the examples it receives.

---

## 8. Mathematical Intuition

Suppose each student has one feature:

\[
x_i = \text{number of study hours}
\]

The correct answer is:

\[
y_i =
\begin{cases}
1 & \text{if the student passed} \\
0 & \text{if the student failed}
\end{cases}
\]

We create a simple model using a threshold \(t\):

\[
\hat{y}_i =
\begin{cases}
1 & \text{if } x_i \geq t \\
0 & \text{if } x_i < t
\end{cases}
\]

Where:

- \(y_i\) is the correct answer.
- \(\hat{y}_i\) is the model’s prediction.
- \(t\) is the threshold learned from data.

The model tries different threshold values and selects the one that produces the highest accuracy.

Accuracy is:

\[
\text{Accuracy}
=
\frac{\text{Number of correct predictions}}
{\text{Total number of predictions}}
\]

For example, if eight out of ten predictions are correct:

\[
\text{Accuracy} = \frac{8}{10} = 0.8 = 80\%
\]

This is a very simple model, but it demonstrates the basic machine-learning process:

1. Define a model.
2. Measure its errors.
3. adjust its parameters.
4. Keep the parameter that performs best.

---

## 9. Python Implementation

The following program learns the best study-hours threshold from data.

```python
def predict_pass(study_hours: float, threshold: float) -> int:
    """Predict whether a student passes."""
    return int(study_hours >= threshold)


def calculate_accuracy(
    study_hours: list[float],
    labels: list[int],
    threshold: float,
) -> float:
    """Calculate prediction accuracy for a threshold."""
    correct_predictions = 0

    for hours, actual_label in zip(study_hours, labels):
        predicted_label = predict_pass(hours, threshold)

        if predicted_label == actual_label:
            correct_predictions += 1

    return correct_predictions / len(labels)


def find_best_threshold(
    study_hours: list[float],
    labels: list[int],
) -> tuple[float, float]:
    """Find the threshold with the highest training accuracy."""
    candidate_thresholds = sorted(set(study_hours))

    best_threshold = candidate_thresholds[0]
    best_accuracy = 0.0

    for threshold in candidate_thresholds:
        accuracy = calculate_accuracy(
            study_hours,
            labels,
            threshold,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def main() -> None:
    """Train and test the student pass predictor."""
    study_hours = [0.5, 1.0, 1.5, 2.5, 3.0, 4.0, 5.0, 6.0]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    threshold, accuracy = find_best_threshold(
        study_hours,
        labels,
    )

    new_student_hours = 3.5
    prediction = predict_pass(
        new_student_hours,
        threshold,
    )

    prediction_text = "Pass" if prediction == 1 else "Fail"

    print(f"Learned threshold: {threshold:.1f} hours")
    print(f"Training accuracy: {accuracy:.2%}")
    print(
        f"Prediction for {new_student_hours} hours: "
        f"{prediction_text}"
    )


if __name__ == "__main__":
    main()
```

Expected output:

```text
Learned threshold: 3.0 hours
Training accuracy: 100.00%
Prediction for 3.5 hours: Pass
```

---

## 10. Code Explanation

### Prediction function

```python
def predict_pass(study_hours: float, threshold: float) -> int:
```

Defines a function with two decimal-number inputs. It returns an integer.

```python
"""Predict whether a student passes."""
```

A docstring explaining the function’s responsibility.

```python
return int(study_hours >= threshold)
```

The comparison produces `True` or `False`. Converting it to an integer produces:

```text
True  → 1
False → 0
```

---

### Accuracy function

```python
correct_predictions = 0
```

Creates a counter for correct predictions.

```python
for hours, actual_label in zip(study_hours, labels):
```

Pairs every study-hours value with its corresponding correct label.

```python
predicted_label = predict_pass(hours, threshold)
```

Uses the model to predict the result.

```python
if predicted_label == actual_label:
```

Checks whether the prediction matches the correct answer.

```python
correct_predictions += 1
```

Increases the counter when the prediction is correct.

```python
return correct_predictions / len(labels)
```

Divides the number of correct predictions by the total number of examples.

---

### Training function

```python
candidate_thresholds = sorted(set(study_hours))
```

`set()` removes duplicate values. `sorted()` arranges the values from smallest to largest.

```python
best_threshold = candidate_thresholds[0]
best_accuracy = 0.0
```

Creates variables for the best result found so far.

```python
for threshold in candidate_thresholds:
```

Tests every possible threshold.

```python
accuracy = calculate_accuracy(...)
```

Measures how well the current threshold performs.

```python
if accuracy > best_accuracy:
```

Checks whether the current result is better than the previous best result.

```python
best_accuracy = accuracy
best_threshold = threshold
```

Stores the better model.

```python
return best_threshold, best_accuracy
```

Returns the learned model parameter and its accuracy.

---

## 11. Time Complexity

Let \(n\) be the number of training examples.

There can be as many as \(n\) candidate thresholds.

For every threshold, the program checks all \(n\) examples:

\[
O(n \times n) = O(n^2)
\]

Sorting also requires:

\[
O(n \log n)
\]

Therefore, the overall time complexity is:

\[
O(n^2)
\]

Space complexity is approximately:

\[
O(n)
\]

because the program stores candidate thresholds and the dataset.

This implementation is educational, not optimized for large datasets.

---

## 12. What This Program Actually Learned

The model learned one parameter:

```text
threshold = 3.0
```

Its complete learned rule is:

```text
Study hours >= 3.0 → Pass
Study hours < 3.0  → Fail
```

This is machine learning because the threshold was selected from data rather than manually fixed by the programmer.

However, it is highly simplified. Real student performance may also depend on:

- Attendance
- Previous grades
- Sleep
- Course difficulty
- Teaching quality
- Exam difficulty

A model is only as useful as its data and assumptions.

---

## 13. Exercises

### Exercise 1

Use this dataset:

```python
study_hours = [1.0, 2.0, 3.0, 4.0, 5.0]
labels = [0, 0, 1, 1, 1]
```

Predict the threshold the program will learn.

### Exercise 2

Add a student who studied `6.0` hours but failed.

```python
study_hours = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
labels = [0, 0, 1, 1, 1, 0]
```

Observe whether the model can achieve 100% accuracy.

### Exercise 3

Change:

```python
new_student_hours = 3.5
```

to:

```python
new_student_hours = 2.0
```

Explain the prediction.

### Exercise 4

Add input validation so that the program rejects:

- Empty datasets
- Datasets of different lengths
- Negative study hours

---

## 14. Mini Project: Exam Eligibility Predictor

Build a small program that learns an attendance threshold.

### Dataset

```python
attendance_percentages = [
    40.0,
    50.0,
    60.0,
    70.0,
    75.0,
    80.0,
    90.0,
]

eligibility_labels = [
    0,
    0,
    0,
    0,
    1,
    1,
    1,
]
```

### Requirements

The program must:

1. Learn the best attendance threshold.
2. Display its training accuracy.
3. Accept a new attendance percentage.
4. Predict `Eligible` or `Not eligible`.
5. Reject attendance below `0` or above `100`.
6. Use functions and type hints.
7. Follow PEP 8.
8. Include at least three tests using `assert`.

Example test:

```python
assert predict_eligibility(80.0, 75.0) == 1
```

### Architecture

```text
Attendance data
      |
      v
Threshold-training function
      |
      v
Learned threshold
      |
      v
Prediction function
      |
      v
Eligible / Not eligible
```

---

## 15. Common Mistakes

### Assuming high training accuracy means the model is good

A model can memorize training data and still fail on new examples. This problem is called overfitting.

### Treating correlation as causation

Study hours may be associated with passing, but that does not prove study hours alone caused the result.

### Using poor-quality data

Incorrect, incomplete, biased, or unrepresentative data produces unreliable models.

### Training and evaluating on the same data

Our introductory program does this for simplicity. In real projects, separate training and testing data are required.

### Using machine learning when a fixed rule is enough

A simple age check does not require machine learning:

```python
is_adult = age >= 18
```

Using ML for an exact legal or business rule would add unnecessary complexity and uncertainty.

### Believing the model understands the subject

The threshold model does not understand students, education, or exams. It only applies a numerical pattern.

### Ignoring edge cases

The code should handle:

- Empty datasets
- Invalid labels
- Negative values
- Mismatched list lengths
- Tied model scores

---

## 16. Debugging Techniques

Print candidate thresholds:

```python
print(f"Testing threshold: {threshold}")
```

Print each prediction:

```python
print(
    f"Hours={hours}, "
    f"Actual={actual_label}, "
    f"Predicted={predicted_label}"
)
```

Check dataset lengths:

```python
assert len(study_hours) == len(labels)
```

Check labels:

```python
assert all(label in (0, 1) for label in labels)
```

Check that data exists:

```python
if not study_hours:
    raise ValueError("The training dataset cannot be empty.")
```

---

## 17. Interview Questions

1. What is Artificial Intelligence?
2. How is machine learning different from traditional programming?
3. What is the relationship between AI, ML, and deep learning?
4. What is a feature?
5. What is a label?
6. What is a model?
7. What is the difference between training and inference?
8. What is supervised learning?
9. What is unsupervised learning?
10. What is reinforcement learning?
11. What is the difference between regression and classification?
12. Why can high training accuracy be misleading?
13. When should machine learning not be used?
14. What is overfitting?
15. What is the time complexity of the threshold-training implementation?

---

## 18. Summary

- AI is the broad field of creating systems that perform intelligent tasks.
- Machine learning is a branch of AI that learns patterns from data.
- Deep learning is a branch of ML based on multilayer neural networks.
- Features are model inputs.
- Labels are the correct outputs used during supervised learning.
- Training creates or adjusts a model.
- Inference uses the trained model on new data.
- Supervised learning uses labeled data.
- Unsupervised learning discovers patterns without labels.
- Reinforcement learning uses rewards and penalties.
- A model does not necessarily understand its task; it learns mathematical relationships.
- Machine learning is inappropriate when an exact, stable rule already solves the problem.

---

# Lesson 1 Quiz

Do not use external resources.

### 1. Which statement is correct?

A. Every AI system uses deep learning.  
B. Deep learning contains machine learning.  
C. Deep learning is a subfield of machine learning.  
D. Machine learning and AI are unrelated.

### 2. What is the main difference between traditional programming and machine learning?

A. Machine learning does not use code.  
B. Traditional programming learns from data automatically.  
C. Machine learning discovers useful patterns from examples.  
D. Traditional programming cannot process data.

### 3. In a house-price prediction system, identify the feature and label:

```text
House size: 1,500 square feet
House price: $280,000
```

### 4. Is house-price prediction normally regression or classification? Explain why.

### 5. Is spam detection normally regression or classification?

### 6. What is the difference between training and inference?

### 7. A company groups customers based on purchasing behavior, but it has no predefined customer categories. Which learning type is being used?

A. Supervised learning  
B. Unsupervised learning  
C. Reinforcement learning  
D. Traditional programming

### 8. A robot receives `+10` points for reaching its destination and `-5` points for hitting an obstacle. Which learning type does this represent?

### 9. Why does 100% training accuracy not guarantee good performance on new data?

### 10. Should machine learning be used to determine whether someone is legally an adult when the rule is simply `age >= 18`? Explain.

### 11. Given the model:

```text
Predict Pass when study hours >= 3
```

What are the predictions for:

```text
2.5 hours
3.0 hours
5.0 hours
```

### 12. Explain this formula in plain language:

\[
\text{Accuracy}
=
\frac{\text{Correct predictions}}
{\text{Total predictions}}
\]
