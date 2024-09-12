import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import read_vectors as rv

# Plot the first vector's values over time
def plot_vector(vector, title):
    plt.plot(vector)
    plt.title(title)
    plt.xlabel('Dimensions')
    plt.ylabel('Values')
    plt.show()

# Retrieve the first vector and its metadata
first_vector = rv.word_vectors[rv.metadata[0]]
plot_vector(first_vector, f"Vector Plot for {rv.metadata[0]}")

# Calculate the similarity between the first two vectors
def calculate_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)

# Retrieve the vectors and calculate similarity
vector1 = rv.word_vectors[rv.metadata[0]]
vector2 = rv.word_vectors[rv.metadata[1]]
similarity = calculate_similarity(vector1, vector2)
print(f"Similarity between {rv.metadata[0]} and {rv.metadata[1]}: {similarity:.4f}")

# Define trading strategy based on vector changes
def trading_strategy(vector1, vector2, threshold=0.1):
    if sum(vector2) - sum(vector1) > threshold:
        return "Buy"
    elif sum(vector2) - sum(vector1) < -threshold:
        return "Sell"
    return "Hold"

# Backtest the trading strategy
def backtest_strategy(word_vectors, metadata, threshold=0.1):
    actions = []
    for i in range(1, len(metadata)):
        action = trading_strategy(word_vectors[metadata[i - 1]], word_vectors[metadata[i]], threshold)
        actions.append((metadata[i], action))
    return actions

# Execute backtest and print first 10 actions
actions = backtest_strategy(rv.word_vectors, rv.metadata)
for action in actions[:10]:
    print(action)