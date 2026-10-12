from odoo import models

from ..utils import ollama


class AiAgent(models.Model):
    _inherit = 'ai.agent'

    def _get_available_skills(self):
        # web search relies on the web grounding of the Odoo AI service, which an
        # Ollama server does not have: never advertise it, so that small models
        # do not pick it over the database skills
        skills = super()._get_available_skills()
        if ollama.is_enabled(self.env):
            skills -= self.env.ref('ai.ai_skill_web_search')
        return skills
