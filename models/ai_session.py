import copy
import json
import logging
import threading
import uuid

import requests

from odoo import api, models
from odoo.exceptions import UserError
from odoo.modules.registry import Registry
from odoo.tools import hmac
from odoo.tools.urls import urljoin as url_join

from odoo.addons.ai.utils.ai_utils import ODOO_AI_COMPLETION_CALLBACK_PATH

from ..utils import ollama

_logger = logging.getLogger(__name__)


class AiSession(models.Model):
    _inherit = 'ai.session'

    @api.model_create_multi
    def create(self, vals_list):
        # no web search with Ollama, see `ai.agent._get_available_skills`
        sessions = super().create(vals_list)
        if ollama.is_enabled(self.env):
            sessions.filtered('enable_web_search').enable_web_search = False
        return sessions

    def validate_session_update(self, updated_config):
        # the client sends its whole config with every message, possibly with a stale
        # or toggled web search: ignore it rather than failing the message, the config
        # published back at the end of `update_session_config` resets the toggle
        if updated_config.get('enable_web_search') and ollama.is_enabled(self.env):
            updated_config = {**updated_config, 'enable_web_search': False}
        return super().validate_session_update(updated_config)

    def _initialize_tool_state(self, state):
        if not ollama.is_enabled(self.env):
            return super()._initialize_tool_state(state)
        # chats created before the install still have web search enabled
        if self.enable_web_search:
            self.enable_web_search = False
        # small models hardly think of loading a skill before answering, they rather
        # ask the user or guess: give them the search tools up front (only if the agent
        # has them, as `super` keeps the available skills only)
        search_skill = self.env.ref('ai.ai_skill_information_retrieval_query', raise_if_not_found=False)
        if search_skill and search_skill.id not in state.get('loaded_skills', []):
            state['loaded_skills'] = [*state.get('loaded_skills', []), search_skill.id]
        return super()._initialize_tool_state(state)

    def _save_and_submit_request(
        self, payload, *, callback_type,
        request_round, request_round_limit,
        state,
    ) -> None:
        """Same as the original, except for the submission after commit: instead of
        queuing the request on IAP, the completion is computed by the Ollama server in a
        background thread, which then calls the completion webhook like IAP would.
        """
        if not ollama.is_enabled(self.env):
            return super()._save_and_submit_request(
                payload, callback_type=callback_type,
                request_round=request_round, request_round_limit=request_round_limit,
                state=state,
            )
        self.ensure_one()
        guest = self.env.context.get("guest")
        vals = {
            "loop_state": "waiting_model",
            "request_uuid": str(uuid.uuid4()),
            "request_webhook_secret": uuid.uuid4().hex,
            "request_round": request_round,
            "request_round_limit": request_round_limit,
            "request_payload": payload,
            "request_user_id": self.env.uid,
            "request_guest_id": guest.id if guest else False,
            "request_context": self._get_request_context_snapshot(),
            "resume_token": False,
            "pending_tool_call": False,
        }
        state["callback_type"] = callback_type
        vals["state"] = state
        self.write(vals)
        self._publish_response_state()

        job = {
            'connection': ollama.get_connection_data(self.env),
            'payload': copy.deepcopy(payload),
            'request_uuid': vals['request_uuid'],
            'webhook_secret': vals['request_webhook_secret'],
            'webhook_url': url_join(self._ai_ollama_get_callback_base_url(), ODOO_AI_COMPLETION_CALLBACK_PATH),
            'dbname': self.env.cr.dbname,
            'user_id': self.env.uid,
            'guest_id': vals['request_guest_id'],
            'context': vals['request_context'],
            'callback_type': callback_type,
            'error_message': self.env._("The Ollama server is unreachable. Please try again later."),
        }

        @self.env.cr.postcommit.add
        def submit():
            threading.Thread(
                target=_complete_and_notify, args=(job,),
                name=f"ai_ollama {job['request_uuid']}", daemon=True,
            ).start()

    def _get_completions(self, messages, instructions, tools=None, **options):
        if not ollama.is_enabled(self.env):
            return super()._get_completions(messages, instructions, tools, **options)
        return self._ai_ollama_complete({
            'messages': messages,
            'instructions': instructions,
            'tools': tools,
            **options,
        })

    @api.model
    def _ai_ollama_get_callback_base_url(self):
        return self.env['ir.config_parameter'].sudo().get_str('ai_ollama.callback_url') or self.get_base_url()

    @api.model
    def _ai_ollama_complete(self, payload):
        if payload.get('image_generation'):
            raise UserError(self.env._("Image generation is not available with the Ollama server."))
        if payload.get('web_grounding'):
            # web grounding is only a bonus for its callers (AI fields always ask for it):
            # answer without it, but tell the model so that it does not refuse because
            # the instructions ask for a web search
            payload = {
                **payload,
                'web_grounding': False,
                'instructions': (payload.get('instructions') or '') + (
                    "\n\nWeb search is not available: never mention it, rely on the provided "
                    "data and on your own knowledge instead."
                ),
            }
        connection = ollama.get_connection_data(self.env)
        try:
            return ollama.get_completion(connection, payload)
        except requests.exceptions.Timeout:
            raise UserError(self.env._("The Ollama server did not answer in time.")) from None
        except (requests.exceptions.RequestException, ollama.OllamaError) as e:
            _logger.warning("Ollama completion failed: %s", e)
            raise UserError(self.env._("The Ollama server is unreachable. Please try again later.")) from e


def _complete_and_notify(job):
    """Run the completion out of any cursor, then deliver the result to the webhook
    of `ai`, which resumes the agent loop in a regular HTTP request.
    """
    threading.current_thread().dbname = job['dbname']
    llm_result, llm_error = False, False
    try:
        llm_result = ollama.get_completion(job['connection'], job['payload'])
    except Exception as e:  # noqa: BLE001
        _logger.exception("Ollama completion failed for AI request %s", job['request_uuid'])
        llm_error = str(e)

    # sign exactly what the webhook will decode
    llm_result = json.loads(json.dumps(llm_result))
    signature = hmac(None, "odoo_ai-webhook", (
        job['request_uuid'], llm_result, llm_error,
    ), secret=job['webhook_secret'])
    try:
        response = requests.post(
            job['webhook_url'],
            json={
                'request_uuid': job['request_uuid'],
                'llm_result': llm_result,
                'llm_error': llm_error,
                'signature': signature,
            },
            headers={'X-Odoo-Database': job['dbname']},
            # the webhook runs the tools and the next round submission synchronously
            timeout=(10, job['connection']['timeout']),
        )
        response.raise_for_status()
    except requests.exceptions.ReadTimeout:
        # the webhook is still processing, the session is not stuck
        _logger.warning("AI completion webhook is slow to answer for request %s", job['request_uuid'])
    except Exception:  # noqa: BLE001
        _logger.exception(
            "Could not deliver the Ollama server result to %s, check the ai_ollama.callback_url parameter",
            job['webhook_url'],
        )
        _rescue_session(job)


def _rescue_session(job):
    """Fail the exchange so the session does not stay stuck on 'waiting_model'."""
    try:
        with Registry(job['dbname']).cursor() as cr:
            env = api.Environment(cr, job['user_id'], job['context'])
            if job['guest_id']:
                env = env(context={**env.context, 'guest': env['mail.guest'].browse(job['guest_id'])})
            session = env['ai.session'].sudo().search([
                ('request_uuid', '=', job['request_uuid']),
                ('loop_state', '=', 'waiting_model'),
            ])
            if not session:
                return
            if job['callback_type'] == 'channel_name':
                session.unlink()
                return
            error_message = job['error_message']
            session._finish_exchange('failed', content=error_message if session.parent_session_id else None)
    except Exception:  # noqa: BLE001
        _logger.exception("Failed to clean up failed AI session submission %s", job['request_uuid'])
