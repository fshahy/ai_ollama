import ast
import json
import math
from unittest.mock import Mock, patch

from odoo.tests import TransactionCase, tagged
from odoo.tools import hmac

from odoo.addons.ai_ollama.models import ai_session as ai_session_module
from odoo.addons.ai_ollama.utils import ollama

CONNECTION = {
    'endpoint': 'http://llm.test/v1',
    'api_key': '',
    'chat_model': 'chat-model',
    'embedding_model': 'embedding-model',
    'embedding_query_prefix': '',
    'timeout': 10,
    'context_length': 16384,
    'extra_params': {},
}


def _http_response(data):
    return Mock(ok=True, status_code=200, json=Mock(return_value=data), raise_for_status=Mock())


def _chat_response(content=None, tool_calls=None):
    message = {'role': 'assistant', 'content': content}
    if tool_calls:
        message['tool_calls'] = tool_calls
    return {'choices': [{'index': 0, 'message': message}]}


@tagged('post_install', '-at_install')
class TestOllamaFormat(TransactionCase):

    def test_to_openai_messages(self):
        messages = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Hello'},
                {'type': 'inline_data', 'mimetype': 'image/png', 'data': 'AAAA'},
            ]},
            {'role': 'assistant', 'provider_metadata': {}, 'content': [
                {'type': 'text', 'text': 'Let me check'},
                {'type': 'tool_call', 'name': 'search', 'args': {'q': 'x'}, 'call_id': 'call_1'},
                {'type': 'tool_call', 'name': 'count', 'args': {}, 'call_id': 'call_2'},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_name': 'search', 'tool_call_id': 'call_1', 'success': True,
                 'result': [{'type': 'text', 'text': '[1, 2]'}]},
                {'type': 'tool_result', 'tool_name': 'count', 'tool_call_id': 'call_2', 'success': True,
                 'result': [{'type': 'text', 'text': {'count': 2}}]},
            ]},
        ]
        self.assertEqual(ollama.to_openai_messages('Be nice', messages), [
            {'role': 'system', 'content': 'Be nice'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Hello'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            ]},
            {'role': 'assistant', 'content': 'Let me check', 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'search', 'arguments': '{"q": "x"}'}},
                {'id': 'call_2', 'type': 'function', 'function': {'name': 'count', 'arguments': '{}'}},
            ]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': '[1, 2]'},
            {'role': 'tool', 'tool_call_id': 'call_2', 'content': '{"count": 2}'},
        ])

    def test_build_chat_request(self):
        payload = {
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}],
            'instructions': 'Sys',
            'tools': [{'name': 'tool_a', 'instructions': 'Does A', 'schema': None}],
            'schema': {'type': 'object', 'properties': {'value': {'type': 'string'}}},
            'usage': 'ai_field',
        }
        request = ollama.build_chat_request({**CONNECTION, 'extra_params': {'temperature': 0.2}}, payload)
        self.assertEqual(request['model'], 'chat-model')
        self.assertEqual(request['temperature'], 0.2)
        self.assertEqual(request['messages'][1], {'role': 'user', 'content': 'Hi'})
        self.assertEqual(request['tools'][0]['function']['name'], 'tool_a')
        self.assertEqual(request['tools'][0]['function']['parameters']['type'], 'object')
        self.assertEqual(request['response_format']['json_schema']['schema'], payload['schema'])
        self.assertNotIn('usage', request)

    def test_parse_chat_response(self):
        message = ollama.parse_chat_response(_chat_response(
            content='<think>hmm</think>\nDone',
            tool_calls=[
                {'id': 'c1', 'type': 'function', 'function': {'name': 'a', 'arguments': '{"x": 1}'}},
                {'type': 'function', 'function': {'name': 'b', 'arguments': 'not json'}},
            ],
        ))
        text, call_a, call_b = message['content']
        self.assertEqual(text, {'type': 'text', 'text': 'Done'})
        self.assertEqual(call_a, {'type': 'tool_call', 'name': 'a', 'args': {'x': 1}, 'call_id': 'c1'})
        self.assertEqual(call_b['args'], {})
        self.assertTrue(call_b['call_id'])

    def test_fit_to_context(self):
        def conversation():
            return [
                {'role': 'system', 'content': 'Instructions'},
                {'role': 'user', 'content': 'Old question ' + 'x' * 3000},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c0'}]},
                {'role': 'tool', 'tool_call_id': 'c0', 'content': 'y' * 3000},
                {'role': 'assistant', 'content': 'Old answer'},
                {'role': 'user', 'content': 'New question'},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1'}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': 'z' * 40000},
            ]
        big_context = ollama.RESPONSE_TOKENS + 20000
        small_context = ollama.RESPONSE_TOKENS + 3000  # ~10k chars

        messages = conversation()
        ollama.fit_to_context(messages, big_context)
        self.assertEqual(messages, conversation())

        # the old exchange goes first, then the tool result of the current one is cut
        messages = conversation()
        ollama.fit_to_context(messages, small_context)
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'tool'])
        self.assertEqual(messages[1]['content'], 'New question')
        self.assertTrue(messages[3]['content'].startswith('z' * ollama.MIN_TOOL_RESULT_CHARS))
        self.assertIn('characters cut', messages[3]['content'])
        self.assertLessEqual(
            sum(len(json.dumps(m)) for m in messages), (small_context - ollama.RESPONSE_TOKENS) * 3.5,
        )

        # the size of the tools counts too
        messages = conversation()
        ollama.fit_to_context(messages, big_context, reserved_chars=65000)
        self.assertEqual(len(messages), 4)

    def test_parse_text_tool_calls(self):
        tools = {
            'start_session': {'type': 'object', 'properties': {
                'agent_id': {'type': 'integer'}, 'message': {'type': 'string'},
            }},
            'search': None,
        }
        call = '{"name": "start_session", "arguments": {"agent_id": "4", "message": "Count"}}'
        for content in (
            call,
            f'<think>hmm</think>\n```json\n{call}\n```',
            f'<tool_call>\n{call}\n</tool_call>\n<tool_call>{{"name": "search", "parameters": {{}}}}</tool_call>',
        ):
            parts = ollama.parse_chat_response(_chat_response(content=content), tools)['content']
            self.assertEqual(parts[0]['type'], 'tool_call', content)
            self.assertEqual(parts[0]['name'], 'start_session')
            self.assertEqual(parts[0]['args'], {'agent_id': 4, 'message': 'Count'})
            self.assertTrue(all(part['type'] == 'tool_call' for part in parts))

        # a domain sent as a list instead of a string
        search_schema = {'type': 'object', 'properties': {
            'domain': {'type': 'string'}, 'fields': {'type': 'array', 'items': {'type': 'string'}},
        }}
        parts = ollama.parse_chat_response(_chat_response(tool_calls=[{'function': {
            'name': 'search', 'arguments': '{"domain": [["share", "=", true]], "fields": ["id"]}',
        }}]), {'search': search_schema})['content']
        self.assertEqual(parts[0]['args'], {'domain': "[['share', '=', True]]", 'fields': ['id']})
        self.assertEqual(ast.literal_eval(parts[0]['args']['domain']), [['share', '=', True]])

        # missing required flags are set to false, other missing arguments are left out
        ask_schema = {'type': 'object', 'required': ['question', 'multi_select', 'domain'], 'properties': {
            'question': {'type': 'string'}, 'multi_select': {'type': 'boolean'},
            'allow_free_text': {'type': 'boolean'}, 'domain': {'type': 'string'},
        }}
        parts = ollama.parse_chat_response(_chat_response(tool_calls=[{'function': {
            'name': 'ask', 'arguments': '{"question": "Which one?"}',
        }}]), {'ask': ask_schema})['content']
        self.assertEqual(parts[0]['args'], {'question': 'Which one?', 'multi_select': False})

        # plain answers, unknown tools and tools not offered stay as text
        for content, tool_names in (
            ('Here it is: {"name": "search"}', tools),
            ('{"name": "delete_everything", "arguments": {}}', tools),
            (call, ()),
        ):
            parts = ollama.parse_chat_response(_chat_response(content=content), tool_names)['content']
            self.assertEqual(parts, [{'type': 'text', 'text': content}])

    def test_fit_vector(self):
        self.assertEqual(ollama.fit_vector([1.0, 2.0], size=4), [1.0, 2.0, 0.0, 0.0])
        vector = ollama.fit_vector([3.0, 4.0, 5.0], size=2)
        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(math.hypot(*vector), 1.0)

    def test_get_embeddings(self):
        data = {'data': [
            {'index': 1, 'embedding': [0.0] * 2000},
            {'index': 0, 'embedding': [1.0] * 768},
        ]}
        with patch.object(ollama.requests, 'post', return_value=_http_response(data)) as post:
            result = ollama.get_embeddings(
                {**CONNECTION, 'embedding_query_prefix': 'Query: '},
                [{'title': 'Doc', 'content': 'Body'}], 'query',
            )
        self.assertEqual(post.call_args.kwargs['json']['input'], ['Query: Doc\n\nBody'])
        first, second = result['embeddings']
        self.assertEqual(len(first), ollama.VECTOR_SIZE)
        self.assertEqual(first[767:769], [1.0, 0.0])
        self.assertEqual(len(second), ollama.VECTOR_SIZE)


@tagged('post_install', '-at_install')
class TestOllamaSession(TransactionCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env['ir.config_parameter'].sudo().set_str('ai_ollama.provider', 'ollama')
        cls.tool =cls.env['ir.actions.server'].create({
            'model_id': cls.env['ir.model']._get_id('ai.agent'),
            'state': 'code',
            'name': 'Test tool',
            'ai_tool_name': 'test_tool',
            'use_in_ai': True,
            'code': "ai['result'] = 'tool output'",
        })

    def test_direct_response_with_tool_call(self):
        responses = [
            _chat_response(tool_calls=[{
                'id': 'call_1', 'type': 'function',
                'function': {'name': 'test_tool', 'arguments': '{}'},
            }]),
            _chat_response(content='The tool said: tool output'),
        ]
        with patch.object(ollama.requests, 'post', side_effect=[_http_response(r) for r in responses]) as post:
            result = self.env['ai.session']._get_direct_response(
                instructions='Use the tool',
                message=[{'type': 'text', 'text': 'Go'}],
                tools=self.tool,
            )
        self.assertEqual(result, [{'type': 'text', 'text': 'The tool said: tool output'}])
        second_request = post.call_args_list[1].kwargs['json']
        self.assertEqual(second_request['messages'][-1], {
            'role': 'tool', 'tool_call_id': 'call_1', 'content': 'tool output',
        })

    def test_web_search_disabled(self):
        agent = self.env.ref('ai.ai_default_agent')
        web_search = self.env.ref('ai.ai_skill_web_search')
        self.assertIn(web_search, agent.skill_ids)
        self.assertNotIn(web_search, agent._get_available_skills())
        self.assertNotIn('name="Web Search"', agent._get_instructions().split('<other_skills>')[0])

        session = self.env['ai.session'].create({'agent_id': agent.id})
        self.assertFalse(session.enable_web_search)
        session.update_session_config({'enable_web_search': True, 'auto_confirm': True})
        self.assertFalse(session.enable_web_search)
        self.assertTrue(session.auto_confirm)

        # a chat created before the install
        session.enable_web_search = True
        state = {}
        session._initialize_tool_state(state)
        self.assertFalse(session.enable_web_search)
        self.assertFalse(set(web_search.tool_ids.ids) & set(state['available_tools']))
        # the search skill is preloaded
        search_skill = self.env.ref('ai.ai_skill_information_retrieval_query')
        self.assertIn(search_skill.id, state['loaded_skills'])
        self.assertLessEqual(set(search_skill.tool_ids.ids), set(state['available_tools']))

        # web grounding (always asked by AI fields) is dropped instead of failing
        chat = _http_response(_chat_response(content='A blue chair.'))
        with patch.object(ollama.requests, 'post', return_value=chat) as post:
            result = self.env['ai.session']._get_completions(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'describe'}]}], 'Describe', web_grounding=True,
            )
        self.assertEqual(result['result']['content'], [{'type': 'text', 'text': 'A blue chair.'}])
        system = post.call_args.kwargs['json']['messages'][0]
        self.assertEqual(system['role'], 'system')
        self.assertIn('Web search is not available', system['content'])

    def test_complete_and_notify_signature(self):
        """The webhook of `ai` must accept the signature computed by the thread."""
        job = {
            'connection': CONNECTION,
            'payload': {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}]},
            'request_uuid': 'uuid-1',
            'webhook_secret': 'secret',
            'webhook_url': 'http://odoo.test/ai/completion_result_ready',
            'dbname': self.env.cr.dbname,
        }
        chat = _http_response(_chat_response(content='Hello!'))
        webhook = _http_response(None)
        # both modules share the `requests` module: chat call, then webhook call
        with patch.object(ai_session_module.requests, 'post', side_effect=[chat, webhook]) as post:
            ai_session_module._complete_and_notify(job)
        post_webhook = post.call_args_list[1]
        self.assertEqual(post_webhook.args, (job['webhook_url'],))
        params = json.loads(json.dumps(post_webhook.kwargs['json']))
        self.assertEqual(params['llm_result']['result']['content'], [{'type': 'text', 'text': 'Hello!'}])
        expected = hmac(None, "odoo_ai-webhook", (
            params['request_uuid'], params['llm_result'], params['llm_error'],
        ), secret='secret')
        self.assertEqual(params['signature'], expected)
        self.assertEqual(post_webhook.kwargs['headers'], {'X-Odoo-Database': self.env.cr.dbname})

    def test_odoo_provider(self):
        """With the Odoo AI provider, the original IAP behavior is kept."""
        self.env['ir.config_parameter'].sudo().set_str('ai_ollama.provider', 'odoo')
        agent = self.env.ref('ai.ai_default_agent')
        self.assertIn(self.env.ref('ai.ai_skill_web_search'), agent._get_available_skills())

        session = self.env['ai.session'].create({'agent_id': agent.id, 'enable_web_search': True})
        self.assertTrue(session.enable_web_search)
        state = {}
        session._initialize_tool_state(state)
        self.assertTrue(session.enable_web_search)

        iap_result = {'result': {'content': [{'type': 'text', 'text': 'From IAP'}]}}
        with (
            patch('odoo.addons.ai.models.ai_session.call_odoo_ai', return_value=iap_result) as call_odoo_ai,
            patch.object(ollama.requests, 'post') as post,
        ):
            result = self.env['ai.session']._get_completions(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}], 'Answer',
            )
        self.assertEqual(result, iap_result)
        self.assertEqual(call_odoo_ai.call_args.args[1], '1/get_completions_sync')
        post.assert_not_called()

        with patch('odoo.addons.ai.models.ai_embedding.call_odoo_ai', return_value={'embeddings': []}) as call_odoo_ai:
            self.env['ai.embedding']._get_embeddings(['text'], 'some-model', 'document')
        self.assertEqual(call_odoo_ai.call_args.args[1], '1/get_embeddings')
