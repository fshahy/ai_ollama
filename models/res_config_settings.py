from odoo import fields, models


class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    ai_ollama_provider = fields.Selection(
        [('odoo', "Odoo AI"), ('ollama', "Ollama Server")],
        string="AI Provider", default='ollama', required=True,
        config_parameter='ai_ollama.provider',
    )
    ai_ollama_endpoint = fields.Char("Ollama Endpoint", config_parameter='ai_ollama.endpoint')
    ai_ollama_api_key = fields.Char("Ollama API Key", config_parameter='ai_ollama.api_key')
    ai_ollama_chat_model = fields.Char("Chat Model", config_parameter='ai_ollama.chat_model')
    ai_ollama_embedding_model = fields.Char("Embedding Model", config_parameter='ai_ollama.embedding_model')
    ai_ollama_timeout = fields.Integer("Timeout (seconds)", config_parameter='ai_ollama.timeout')
    ai_ollama_context_length = fields.Integer("Context Length (tokens)", config_parameter='ai_ollama.context_length')

    def set_values(self):
        ICP = self.env['ir.config_parameter'].sudo()
        keys = ('ai_ollama.provider', 'ai_ollama.embedding_model')
        before = [ICP.get_str(key) for key in keys]
        super().set_values()
        if [ICP.get_str(key) for key in keys] != before:
            self.env['ai.embedding']._ai_ollama_switch_embedding_model()
