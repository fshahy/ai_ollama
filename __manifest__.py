{
    'name': 'AI: Ollama',
    'version': '1.1',
    'category': 'Productivity',
    'summary': "Run Odoo AI on an Ollama (or any OpenAI-compatible) server instead of IAP",
    'description': """
Routes the Odoo AI completions and embeddings to an Ollama server, or any other
server speaking the OpenAI API (llama.cpp server, vLLM, LM Studio, ...), instead of Odoo IAP.

Covered: agents and chat (tool calling included), one-shot AI features (AI fields,
server actions, composer, ...) with structured output, and RAG embeddings.
Not covered, still on IAP: audio transcription and realtime voice.
Not available: image generation and web search (hidden from the agents).
""",
    'depends': ['ai', 'base_setup'],
    'data': [
        'data/ir_config_parameter_data.xml',
        'views/res_config_settings_views.xml',
    ],
    'post_init_hook': '_post_init_ai_ollama',
    'license': 'LGPL-3',
}
