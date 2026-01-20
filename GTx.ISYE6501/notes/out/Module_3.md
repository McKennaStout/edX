# Module 3 — Key concepts

## Model validation

- Definition (plain): Validation checks if your model will work on unseen data.
- Definition (formal): Model validation estimates how well a trained model will generalize to new data.
- Support (professor phrasing):
  - >> In a previous lesson we talked about validation, which is measuring the effectiveness of a model.
  - That way if the first set, the training set had unique random effects that the classifier was designed for, we wouldn't be counting those benefits when we measure effectiveness on the second set called the validation set.
  - As you can see in this example, this line classifies 90% of the points correctly in the training set, but only 80% of the points correctly in the validation set.

## Overfitting

- Definition (plain): Overfitting is memorizing the training set instead of learning the real pattern.
- Definition (formal): Overfitting occurs when a model fits noise in the training data and performs poorly on new data.

## Underfitting

- Definition (plain): Underfitting is being too simple to learn the pattern.
- Definition (formal): Underfitting occurs when a model is too simple to capture the true structure in the data.

## Train/validation/test split

- Definition (plain): You learn on one part, tune on another, and score on a final untouched part.
- Definition (formal): A data split partitions data into training, validation, and test sets for fitting, tuning, and final evaluation.

## Cross-validation

- Definition (plain): Cross-validation tests the model multiple times on different splits to get a more stable score.
- Definition (formal): Cross-validation repeatedly splits data into train/validation folds to reduce evaluation variance and tune models more reliably.
- Support (professor phrasing):
  - It's called cross-validation and we'll see what it is in a future lesson.
  - Cross-validation is a way to avoid that problem.
  - We have k-means, k nearest neighbor, and now k-fold cross-validation.

## Bias-variance tradeoff

- Definition (plain): Simple models miss patterns; complex models chase noise. You balance the two.
- Definition (formal): Bias-variance tradeoff describes how model complexity affects systematic error (bias) and sensitivity to data noise (variance).

