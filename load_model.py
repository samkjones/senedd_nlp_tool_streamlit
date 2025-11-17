from sentence_transformers import SentenceTransformer

# This will download and cache the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Save it to a local directory
model.save("./local_model")
