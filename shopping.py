import csv
import sys
import calendar
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from SimpleKNeighborsClassifier import SimpleKNeighborsClassifier


TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model_SimpleKNeighbors(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity, f1_score_label_0, f1_score_label_1 = evaluate(
        y_test, predictions
    )

    def count_same_items(list1, list2):
        count = 0
        for item1, item2 in zip(list1, list2):
            if item1 == item2:
                count += 1
        return count

    correctPred = count_same_items(y_test, predictions)

    # Print results
    print(f"Correct: {count_same_items(y_test, predictions)}")
    print(f"Incorrect: {len(predictions) - correctPred}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"F1-Score label=0: {100 * f1_score_label_0:.2f}")
    print(f"F1-Score label=1: {100 * f1_score_label_1:.2f}")

    report = classification_report(y_test, predictions)
    print(report)

    print(f"Support for label=0: {y_test.count(0)}")
    print(f"Support for label=1: {y_test.count(1)}")
    print(f"Accuracy: {correctPred / len(y_test)}")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    months = {
        month: index - 1 for index, month in enumerate(calendar.month_abbr) if index
    }
    months["June"] = months.pop("Jun")

    evidence = []
    labels = []

    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append(
                [
                    int(row["Administrative"]),
                    float(row["Administrative_Duration"]),
                    int(row["Informational"]),
                    float(row["Informational_Duration"]),
                    int(row["ProductRelated"]),
                    float(row["ProductRelated_Duration"]),
                    float(row["BounceRates"]),
                    float(row["ExitRates"]),
                    float(row["PageValues"]),
                    float(row["SpecialDay"]),
                    months[row["Month"]],
                    int(row["OperatingSystems"]),
                    int(row["Browser"]),
                    int(row["Region"]),
                    int(row["TrafficType"]),
                    1 if row["VisitorType"] == "Returning_Visitor" else 0,
                    1 if row["Weekend"] == "TRUE" else 0,
                ]
            )
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def train_model_SimpleKNeighbors(evidence, labels):
    model = SimpleKNeighborsClassifier(k=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.
    This is the recall of the positive class!

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    This is the recall of the negative class!

    `f1-score` should be a floating-point value from 0 to 1.
    The formula is: (2 * precision * recall) / (precision + recall)
    Recall = Sensitivity
    Precision = tp / (tp + fp)
    Precision for the positive class = tp / (tp + fp)
    Precision for the negative class = tn / (tn + fn)
    """
    sensitivity = float(0)
    specificity = float(0)

    total_positive = float(0)
    total_negative = float(0)

    true_positive = float(0)
    false_negative = float(0)

    true_negative = float(0)
    false_positive = float(0)

    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 0:
            true_negative += 1
        elif label == 1 and prediction == 0:
            false_negative += 1
        elif label == 0 and prediction == 1:
            false_positive += 1

    recall_label_1 = saveDivide(true_positive, (true_positive + false_negative))
    recall_label_0 = saveDivide(true_negative, (true_negative + false_positive))

    precision_label_1 = saveDivide(true_positive, (true_positive + false_positive))
    precision_label_0 = saveDivide(true_negative, (true_negative + false_negative))

    sensitivity = recall_label_1
    specificity = recall_label_0

    f1_score_label_0 = saveDivide((2 * precision_label_0 * recall_label_0), (
        precision_label_0 + recall_label_0
    ))
    f1_score_label_1 = saveDivide((2 * precision_label_1 * recall_label_1), (
        precision_label_1 + recall_label_1
    ))

    return sensitivity, specificity, f1_score_label_0, f1_score_label_1

def saveDivide(numerator, divisor):
    # I use the convention here to set precision, recall, and f1-score to 0 if the divisor is 0
    if divisor == 0:
        return 0
    else: 
        return numerator / divisor
  
if __name__ == "__main__":
    main()
