"""Client for an OpenAI-compatible LLM server (Ollama, llama.cpp, vLLM, LM Studio...).

Translates the normalized message format of the `ai` module (the one exchanged with
the Odoo AI IAP service, see `odoo.addons.ai.utils.types`) to the OpenAI chat
completions / embeddings API and back. It is ORM-free so that it can run outside of
a cursor, e.g. in the background thread serving the asynchronous agent loop.
"""
import base64
import io
import json
import logging
import math
import re
import uuid

import requests

from odoo.tools.json import json_default
from odoo.tools.pdf import OdooPdfFileReader

_logger = logging.getLogger(__name__)

# Size of `ai.embedding.embedding_vector`
VECTOR_SIZE = 1536
DEFAULT_ENDPOINT = 'http://localhost:11434/v1'
DEFAULT_TIMEOUT = 300
DEFAULT_CONTEXT_LENGTH = 16384
# Conservative estimate, so that the prompt fits for real in the context
CHARS_PER_TOKEN = 3.5
# Room kept in the context for the answer
RESPONSE_TOKENS = 2048
# Tool results are never shortened below this size
MIN_TOOL_RESULT_CHARS = 1500
# Reasoning models may inline their chain of thought in the answer
THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
# Small models sometimes write their tool calls in the answer instead of using the
# native `tool_calls`: as bare JSON, in a code fence or in Hermes `<tool_call>` tags
TEXT_TOOL_CALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*(?:</tool_call>|$)', re.DOTALL)
CODE_FENCE_RE = re.compile(r'^```(?:json)?\s*(.*?)\s*```$', re.DOTALL)


class OllamaError(Exception):
    pass


def is_enabled(env):
    """Whether the AI requests go to the Ollama server rather than to Odoo AI (IAP)."""
    return (env['ir.config_parameter'].sudo().get_str('ai_ollama.provider') or 'ollama') == 'ollama'


def get_connection_data(env):
    """Snapshot the configuration into plain scalars, usable without a cursor."""
    ICP = env['ir.config_parameter'].sudo()
    try:
        extra_params = json.loads(ICP.get_str('ai_ollama.extra_params') or '{}')
    except json.JSONDecodeError:
        _logger.warning("Ignoring invalid JSON in ai_ollama.extra_params")
        extra_params = {}
    return {
        'endpoint': (ICP.get_str('ai_ollama.endpoint') or DEFAULT_ENDPOINT).rstrip('/'),
        'api_key': ICP.get_str('ai_ollama.api_key'),
        'chat_model': ICP.get_str('ai_ollama.chat_model'),
        'embedding_model': ICP.get_str('ai_ollama.embedding_model'),
        'embedding_query_prefix': ICP.get_str('ai_ollama.embedding_query_prefix'),
        'timeout': ICP.get_int('ai_ollama.timeout') or DEFAULT_TIMEOUT,
        'context_length': ICP.get_int('ai_ollama.context_length') or DEFAULT_CONTEXT_LENGTH,
        'extra_params': extra_params if isinstance(extra_params, dict) else {},
    }


def _post(connection, path, payload):
    headers = {}
    if connection['api_key']:
        headers['Authorization'] = f"Bearer {connection['api_key']}"
    url = f"{connection['endpoint']}/{path}"
    response = requests.post(url, json=payload, headers=headers, timeout=connection['timeout'])
    if not response.ok:
        raise OllamaError(f"{url} returned {response.status_code}: {response.text[:1000]}")
    return response.json()


# ------------------------------------------------------------
# Odoo AI format -> OpenAI format
# ------------------------------------------------------------

def _part_text(part):
    # a text part may hold any JSON value (e.g. a structured output)
    text = part['text']
    return text if isinstance(text, str) else json.dumps(text, default=json_default, ensure_ascii=False)


def _pdf_to_text(b64_data):
    try:
        reader = OdooPdfFileReader(io.BytesIO(base64.b64decode(b64_data)), strict=False)
        return '\n'.join(page.extract_text() or '' for page in reader.pages).strip()
    except Exception:  # noqa: BLE001
        _logger.warning("Could not extract the text of a PDF sent to the Ollama server", exc_info=True)
        return ''


def _inline_data_to_openai(part):
    mimetype = part['mimetype']
    if mimetype.startswith('image/'):
        return {'type': 'image_url', 'image_url': {'url': f"data:{mimetype};base64,{part['data']}"}}
    if mimetype == 'application/pdf' and (text := _pdf_to_text(part['data'])):
        return {'type': 'text', 'text': f"<attached_pdf>\n{text}\n</attached_pdf>"}
    return {'type': 'text', 'text': f"[An attached file ({mimetype}) could not be read by the model]"}


def _user_content(parts):
    # plain strings are the most widely supported format, keep lists for images
    if all(part['type'] == 'text' for part in parts):
        return '\n\n'.join(part['text'] for part in parts)
    return parts


def to_openai_messages(instructions, messages):
    result = [{'role': 'system', 'content': instructions}] if instructions else []
    for message in messages:
        if message['role'] == 'assistant':
            text = '\n'.join(_part_text(part) for part in message['content'] if part['type'] == 'text')
            tool_calls = [{
                'id': str(part['call_id']),
                'type': 'function',
                'function': {
                    'name': part['name'],
                    'arguments': json.dumps(part['args'] or {}, default=json_default, ensure_ascii=False),
                },
            } for part in message['content'] if part['type'] == 'tool_call']
            assistant_message = {'role': 'assistant', 'content': text}
            if tool_calls:
                assistant_message['tool_calls'] = tool_calls
            result.append(assistant_message)
            continue

        # Odoo sends the tool results in a user message, OpenAI wants one `tool`
        # message per call right after the assistant message that made them.
        user_parts = []
        for part in message['content']:
            if part['type'] == 'tool_result':
                result.append({
                    'role': 'tool',
                    'tool_call_id': str(part['tool_call_id']),
                    'content': '\n'.join(
                        _part_text(p) for p in part['result'] if p['type'] == 'text'
                    ) or ('success' if part['success'] else 'error'),
                })
                # tool messages are text only, files go to the next user message
                user_parts.extend(
                    _inline_data_to_openai(p) for p in part['result'] if p['type'] == 'inline_data'
                )
            elif part['type'] == 'text':
                user_parts.append({'type': 'text', 'text': _part_text(part)})
            elif part['type'] == 'inline_data':
                user_parts.append(_inline_data_to_openai(part))
        if user_parts:
            result.append({'role': 'user', 'content': _user_content(user_parts)})
    return result


def to_openai_tools(tools):
    return [{
        'type': 'function',
        'function': {
            'name': tool['name'],
            'description': tool['instructions'],
            'parameters': tool['schema'] or {'type': 'object', 'properties': {}, 'required': []},
        },
    } for tool in tools]


def _size(value):
    return len(json.dumps(value, default=json_default, ensure_ascii=False))


def fit_to_context(messages, context_length, reserved_chars=0):
    """Shrink the conversation in place so that it fits in the context of the model.

    Servers like Ollama silently drop the *beginning* of a prompt that is too long,
    i.e. the instructions and the question, leaving the model with e.g. a bare tool
    result. Instead, drop the oldest exchanges first, then shorten the biggest tool
    results of the current exchange. The system message is always kept.

    :param messages: OpenAI messages, the first one being the system message
    :param reserved_chars: size of what is sent besides the messages (the tools)
    """
    budget = int((context_length - RESPONSE_TOKENS) * CHARS_PER_TOKEN) - reserved_chars
    total = sum(_size(message) for message in messages)
    if total <= budget:
        return

    # the current exchange starts after the last final answer
    first = 1 if messages and messages[0]['role'] == 'system' else 0
    current = next((
        index + 1 for index in range(len(messages) - 1, first - 1, -1)
        if messages[index]['role'] == 'assistant' and not messages[index].get('tool_calls')
    ), first)
    dropped = 0
    while total > budget and first < current:
        # drop whole exchanges: the history must start with a user message
        total -= _size(messages.pop(first))
        current -= 1
        dropped += 1
        while first < current and messages[first]['role'] != 'user':
            total -= _size(messages.pop(first))
            current -= 1
            dropped += 1

    shortened = 0
    tool_messages = [m for m in messages if m['role'] == 'tool' and isinstance(m['content'], str)]
    for message in sorted(tool_messages, key=lambda m: len(m['content']), reverse=True):
        if total <= budget:
            break
        content = message['content']
        keep = max(MIN_TOOL_RESULT_CHARS, len(content) - (total - budget) - 200)
        if keep >= len(content):
            continue
        message['content'] = content[:keep] + (
            f"\n[... {len(content) - keep} characters cut: the result is too long for the context."
            " If the answer is not above, retry asking for less data (fewer fields, narrower domain, lower limit).]"
        )
        total += _size(message) - _size({**message, 'content': content})
        shortened += 1

    _logger.info(
        "Ollama prompt fitted to a %s tokens context: %s old messages dropped, %s tool results shortened%s",
        context_length, dropped, shortened, "" if total <= budget else ", still too long",
    )


def build_chat_request(connection, payload):
    """:param payload: the params the `ai` module sends to IAP `get_completions`"""
    request = {
        **connection['extra_params'],
        'model': connection['chat_model'],
        'messages': to_openai_messages(payload.get('instructions'), payload['messages']),
        'stream': False,
    }
    if tools := payload.get('tools'):
        request['tools'] = to_openai_tools(tools)
    fit_to_context(request['messages'], connection['context_length'], _size(request.get('tools') or []))
    if schema := payload.get('schema'):
        request['response_format'] = {
            'type': 'json_schema',
            'json_schema': {'name': 'response', 'schema': schema, 'strict': True},
        }
    return request


# ------------------------------------------------------------
# OpenAI format -> Odoo AI format
# ------------------------------------------------------------

def _parse_tool_arguments(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            _logger.warning("Ollama returned invalid tool arguments: %s", arguments)
            return {}
    return arguments if isinstance(arguments, dict) else {}


def _coerce_value(value, schema):
    """Fix the types small models tend to get wrong (`"4"` for `4`, a list for a domain
    string), as the `ai` module validates the tool arguments strictly against their schema."""
    schema = schema or {}
    param_type = schema.get('type')
    if isinstance(value, list) and param_type == 'array':
        return [_coerce_value(item, schema.get('items')) for item in value]
    if isinstance(value, dict) and param_type == 'object':
        return _coerce_args(value, schema)
    if isinstance(value, (list, dict)) and param_type == 'string':
        # domains, `having`, x2m commands... are Python literals passed as strings,
        # which the `ai` tools parse with `ast.literal_eval`
        return repr(value)
    if not isinstance(value, str):
        return value
    try:
        if param_type == 'integer':
            return int(value.strip())
        if param_type == 'number':
            return float(value.strip())
    except ValueError:
        return value
    if param_type == 'boolean' and value.strip().lower() in ('true', 'false'):
        return value.strip().lower() == 'true'
    return value


def _coerce_args(args, schema):
    properties = (schema or {}).get('properties') or {}
    args = {key: _coerce_value(value, properties.get(key)) for key, value in args.items()}
    # small models omit the flags they do not care about, and then keep retrying the
    # call without them; false is the conservative value for the required ones
    for key in (schema or {}).get('required') or []:
        if key not in args and (properties.get(key) or {}).get('type') == 'boolean':
            args[key] = False
    return args


def _parse_text_tool_calls(content, tool_names):
    """Recover the tool calls a model wrote as text in its answer.

    :param tool_names: container of the names of the tools offered to the model

    Only a content made of nothing but calls to known tools is considered, so that
    an answer merely showing some JSON to the user is left untouched.

    :return: a list of OpenAI `tool_calls`, empty if the content is not a tool call
    """
    if not tool_names:
        return []
    if tagged := TEXT_TOOL_CALL_RE.findall(content):
        if TEXT_TOOL_CALL_RE.sub('', content).strip():
            return []
        candidates = tagged
    else:
        fenced = CODE_FENCE_RE.match(content)
        candidates = [fenced.group(1) if fenced else content]

    calls = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            return []
        for call in value if isinstance(value, list) else [value]:
            if not isinstance(call, dict) or call.get('name') not in tool_names:
                return []
            arguments = call.get('arguments', call.get('parameters'))
            calls.append({'function': {'name': call['name'], 'arguments': arguments or {}}})
    return calls


def parse_chat_response(data, tool_schemas=None):
    """:param tool_schemas: {name: parameters schema} of the tools offered to the model
    :return: an `AssistantMessage`
    """
    tool_schemas = tool_schemas or {}
    try:
        message = data['choices'][0]['message']
    except (KeyError, IndexError, TypeError):
        raise OllamaError(f"Unexpected response from the Ollama server: {str(data)[:1000]}") from None

    content = message.get('content') or ''
    if isinstance(content, list):
        content = ''.join(part.get('text', '') for part in content if isinstance(part, dict))
    content = THINK_RE.sub('', content).strip()

    tool_calls = message.get('tool_calls') or []
    if not tool_calls and (tool_calls := _parse_text_tool_calls(content, tool_schemas)):
        _logger.info("Ollama wrote its tool calls as text, running them anyway")
        content = ''

    parts = [{'type': 'text', 'text': content}] if content else []
    for tool_call in tool_calls:
        function = tool_call.get('function') or {}
        name = function.get('name')
        parts.append({
            'type': 'tool_call',
            'name': name,
            'args': _coerce_args(_parse_tool_arguments(function.get('arguments')), tool_schemas.get(name)),
            'call_id': tool_call.get('id') or f"call_{uuid.uuid4().hex[:24]}",
        })
    if not parts:
        parts.append({'type': 'text', 'text': ''})
    return {'role': 'assistant', 'content': parts, 'provider_metadata': {}}


def get_completion(connection, payload):
    """Ollama equivalent of the IAP `get_completions(_sync)` routes.

    :return: a `CompletionResponse`
    """
    data = _post(connection, 'chat/completions', build_chat_request(connection, payload))
    tool_schemas = {tool['name']: tool['schema'] for tool in payload.get('tools') or []}
    return {'result': parse_chat_response(data, tool_schemas)}


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

def fit_vector(vector, size=VECTOR_SIZE):
    """Adapt a vector to the fixed size of the database column.

    Zero-padding keeps the cosine distance unchanged. Truncating is only meaningful
    for Matryoshka models (nomic-embed, qwen3-embedding, mxbai, ...), whose leading
    dimensions carry the information; the result is re-normalized.
    """
    if len(vector) < size:
        return list(vector) + [0.0] * (size - len(vector))
    vector = vector[:size]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def get_embeddings(connection, inputs, mode):
    """Ollama equivalent of the IAP `get_embeddings` route.

    :param inputs: list of `EmbeddingInput`
    :param mode: 'document' or 'query'
    :return: an `EmbeddingResponse`
    """
    texts = [
        f"{item['title']}\n\n{item['content']}" if item.get('title') else item['content']
        for item in inputs
    ]
    if mode == 'query' and connection['embedding_query_prefix']:
        texts = [connection['embedding_query_prefix'] + text for text in texts]
    data = _post(connection, 'embeddings', {'model': connection['embedding_model'], 'input': texts})
    try:
        rows = sorted(data['data'], key=lambda row: row.get('index', 0))
        return {'embeddings': [fit_vector(row['embedding']) for row in rows]}
    except (KeyError, TypeError):
        raise OllamaError(f"Unexpected response from the Ollama server: {str(data)[:1000]}") from None
