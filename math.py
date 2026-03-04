import math



def validate_data(data):
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    if not all(isinstance(i, (int, float)) for i in data):
        raise TypeError("All elements must be numbers")



def calculate_mean(numbers):
    validate_data(numbers)
    return sum(numbers) / len(numbers)


def calculate_variance(numbers):
    validate_data(numbers)
    mean_value = calculate_mean(numbers)
    return sum((num - mean_value) ** 2 for num in numbers) / len(numbers)


def calculate_standard_deviation(numbers):
    return math.sqrt(calculate_variance(numbers))


def calculate_covariance(x, y):
    validate_data(x)
    validate_data(y)

    if len(x) != len(y):
        raise ValueError("Lists must have the same length")

    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)

    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)


def calculate_correlation(x, y):
    std_x = calculate_standard_deviation(x)
    std_y = calculate_standard_deviation(y)

    if std_x == 0 or std_y == 0:
        return 0

    return calculate_covariance(x, y) / (std_x * std_y)


def calculate_z_score(value, data):
    mean_value = calculate_mean(data)
    std_value = calculate_standard_deviation(data)

    if std_value == 0:
        return 0

    return (value - mean_value) / std_value


def min_max_scale(data):
    validate_data(data)

    minimum = min(data)
    maximum = max(data)

    if maximum == minimum:
        return [0 for _ in data]

    return [(value - minimum) / (maximum - minimum) for value in data]



def basic_probability(success, total):
    if total <= 0:
        raise ValueError("Total must be greater than zero")
    return success / total


def apply_bayes(p_b_given_a, p_a, p_b):
    if p_b == 0:
        raise ValueError("P(B) cannot be zero")
    return (p_b_given_a * p_a) / p_b




def multiply_matrices(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not allow multiplication")

    rows_A = len(A)
    cols_B = len(B[0])

    result = [[0] * cols_B for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]



def simple_linear_prediction(x, intercept, slope):
    return intercept + slope * x


def calculate_mse(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Lists must be the same length")

    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)



def logistic_sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    else:
        exp_val = math.exp(value)
        return exp_val / (1 + exp_val)



def compute_accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0


def compute_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def compute_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def compute_f1(tp, fp, fn):
    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def calculate_entropy(prob_list):
    validate_data(prob_list)
    return sum(-p * math.log2(p) for p in prob_list if p > 0)



if __name__ == "__main__":

    try:
        data = [5, 10, 15, 20, 25]
        x = [1, 2, 3, 4, 5]
        y = [3, 6, 9, 12, 15]

        print("Mean:", calculate_mean(data))
        print("Variance:", calculate_variance(data))
        print("Standard Deviation:", calculate_standard_deviation(data))
        print("Correlation:", calculate_correlation(x, y))
        print("Z Score (15):", calculate_z_score(15, data))
        print("Scaled Data:", min_max_scale(data))

        print("Probability:", basic_probability(4, 10))
        print("Bayes Result:", apply_bayes(0.7, 0.5, 0.6))

        A = [[2, 3], [4, 5]]
        B = [[1, 2], [3, 4]]

        print("Matrix Multiplication:", multiply_matrices(A, B))
        print("Transpose:", transpose(A))

        predictions = [simple_linear_prediction(i, 0, 3) for i in x]
        print("MSE:", calculate_mse(y, predictions))

        print("Sigmoid(2):", logistic_sigmoid(2))

        print("Accuracy:", compute_accuracy(50, 40, 5, 5))
        print("Precision:", compute_precision(50, 5))
        print("Recall:", compute_recall(50, 5))
        print("F1 Score:", compute_f1(50, 5, 5))

        probs = [0.6, 0.4]
        print("Entropy:", calculate_entropy(probs))

    except Exception as e:
        print("Error occurred:", e)