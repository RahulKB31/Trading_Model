from gensim.models import KeyedVectors

# Load the pre-trained vectors
word_vectors = KeyedVectors.load("Reliance_embeddings.kv", mmap='r')

# Check Gensim version and access metadata
import gensim
if gensim.__version__ >= '4.0.0':
    metadata = word_vectors.index_to_key
else:
    metadata = word_vectors.key_to_index

# Print the first 5 keys and their vectors
for key in (metadata[:5] if gensim.__version__ >= '4.0.0' else list(metadata.keys())[:5]):
    print(f"Key: {key}")
    print(f"Vector: {word_vectors[key]}\n")

