from . import models


def _post_init_ai_ollama(env):
    # Re-embed existing sources with the Ollama embedding model
    env['ai.embedding']._ai_ollama_switch_embedding_model()
