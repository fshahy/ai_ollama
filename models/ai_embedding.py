import logging

import requests

from odoo import api, models, modules
from odoo.exceptions import UserError

from ..utils import ollama

_logger = logging.getLogger(__name__)


class AiEmbedding(models.Model):
    _inherit = 'ai.embedding'

    @api.ormcache()
    def _get_default_embedding_model(self):
        if modules.module.current_test or not ollama.is_enabled(self.env):
            return super()._get_default_embedding_model()
        return self.env['ir.config_parameter'].sudo().get_str('ai_ollama.embedding_model')

    @api.ormcache()
    def _get_supported_embedding_models(self):
        if modules.module.current_test or not ollama.is_enabled(self.env):
            return super()._get_supported_embedding_models()
        return [self._get_default_embedding_model()]

    @api.model
    def _get_embeddings(self, input, model, mode):
        if not ollama.is_enabled(self.env):
            return super()._get_embeddings(input, model, mode)
        # `model` is ignored: records of any other model are reset and re-embedded
        # by the "deprecated embedding models" cron, see `_ai_ollama_switch_embedding_model`
        connection = ollama.get_connection_data(self.env)
        try:
            return ollama.get_embeddings(connection, input, mode)
        except ollama.OllamaError as e:
            _logger.warning("Ollama embedding failed: %s", e)
            raise UserError(self.env._("The Ollama server could not compute the embeddings.")) from e
        except requests.exceptions.RequestException as e:
            _logger.warning("Ollama embedding failed: %s", e)
            raise UserError(self.env._("The Ollama server is unreachable. Please try again later.")) from e

    @api.model
    def _ai_ollama_switch_embedding_model(self):
        """Re-embed all the sources with the embedding model of the configured provider."""
        self.env.transaction.invalidate_ormcache()
        self.env.ref('ai.ir_cron_update_deprecated_embedding_models')._trigger()
