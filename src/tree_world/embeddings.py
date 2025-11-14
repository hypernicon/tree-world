import sentence_transformers

hf_models = {}
def embed_text_sentence_transformers(text, method="BAAI/bge-large-en-v1.5"):
    global hf_models
    if method not in hf_models:
        hf_models[method] = sentence_transformers.SentenceTransformer(method)

    model = hf_models[method]

    if isinstance(text, list):
        if len(text) == 0:
            return None
        # note that we clone the output to avoid messages about inference tensors not requiring grad
        return model.encode(text, convert_to_tensor=True, normalize_embeddings=True).clone()
    else:
        return model.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0].clone()

