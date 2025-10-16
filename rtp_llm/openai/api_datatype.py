import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
import json
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.utils.base_model_datatypes import AuxInfo


# Pre-import these to avoid repeated imports in the performance-critical function
from dataclasses import asdict, is_dataclass
from enum import Enum

def _asdict_with_enum_handling(obj, _cache=None):
    """处理包含枚举的 dataclass 字典转换"""
    # 使用弱引用缓存来避免循环引用和重复计算
    if _cache is None:
        _cache = set()

    obj_id = id(obj)
    if obj_id in _cache:
        return obj  # 避免循环引用
    _cache.add(obj_id)

    try:
        def convert_enum(value):
            # 处理各种枚举类型
            if isinstance(value, Enum):
                return value.value
            # 处理嵌套的 dataclass
            elif is_dataclass(value) and not isinstance(value, (type, type(open))):  # Optimized type check
                return _asdict_with_enum_handling(value, _cache)
            # 递归处理字典 - optimize with generator
            elif isinstance(value, dict):
                return {k: convert_enum(v) for k, v in value.items()}
            # 递归处理列表 - use list comprehension
            elif isinstance(value, list):
                # Check if list is empty to optimize
                if not value:
                    return value
                return [convert_enum(item) for item in value]
            elif isinstance(value, tuple):
                # 优化：保持tuple类型 - use generator
                if not value:
                    return value
                return tuple(convert_enum(item) for item in value)
            elif isinstance(value, set):
                # 处理set类型
                if not value:
                    return value
                return {convert_enum(item) for item in value}
            else:
                return value

        result = asdict(obj)
        # Process the result dictionary efficiently
        return {k: convert_enum(v) for k, v in result.items()}
    finally:
        _cache.discard(obj_id)


# 兼容性装饰器，用于添加 Pydantic 风格的方法
def add_pydantic_compatibility(cls):
    """为 dataclass 添加 Pydantic 风格的方法"""
    def model_dump(self, *args, **kwargs):
        # 处理 exclude_none 参数
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def model_dump_json(self, *args, **kwargs):
        import json
        # 处理 exclude_none 参数
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
            # 过滤掉 exclude_none 参数，避免传递给 json.dumps
            json_kwargs = {k: v for k, v in kwargs.items() if k != 'exclude_none'}
            return json.dumps(result, *args, **json_kwargs)
        return json.dumps(result, *args, **kwargs)

    def dict(self, *args, **kwargs):
        # 处理 exclude_none 参数
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def json(self, *args, **kwargs):
        import json
        # 处理 exclude_none 参数
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
            # 过滤掉 exclude_none 参数，避免传递给 json.dumps
            json_kwargs = {k: v for k, v in kwargs.items() if k != 'exclude_none'}
            return json.dumps(result, *args, **json_kwargs)
        return json.dumps(result, *args, **kwargs)

    cls.model_dump = model_dump
    cls.model_dump_json = model_dump_json
    cls.dict = dict
    cls.json = json
    return cls


# 自定义 Field 函数，用于处理默认值
def Field(default=None, default_factory=None):
    if default_factory is not None:
        return field(default_factory=default_factory)
    return field(default=default)


@add_pydantic_compatibility
@dataclass
class ModelCard:
    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


@add_pydantic_compatibility
@dataclass
class ModelList:
    object: str = "list"
    data: List[ModelCard] = field(default_factory=list)


@add_pydantic_compatibility
@dataclass
class FunctionCall:
    name: Optional[str] = None
    arguments: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = None
        if self.arguments is None:
            self.arguments = None


@add_pydantic_compatibility
@dataclass
class ToolCall:
    # 参照 openai 官方api definition
    index: Optional[int] = None
    id: Optional[str] = None
    type: str = ""
    function: FunctionCall = field(default_factory=FunctionCall)

    def __post_init__(self):
        if self.function is None:
            self.function = FunctionCall()


class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    function = "function"
    tool = "tool"
    observation = "observation"


class ContentPartTypeEnum(str, Enum):
    text = "text"
    igraph = "igraph"
    image_url = "image_url"
    video_url = "video_url"
    audio_url = "audio_url"


@add_pydantic_compatibility
@dataclass
class MMPreprocessConfigPart:
    resized_width: Optional[int] = None
    resized_height: Optional[int] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    fps: Optional[int] = None
    min_frames: Optional[int] = None
    max_frames: Optional[int] = None

    def __post_init__(self):
        # 处理 None 值
        pass


@add_pydantic_compatibility
@dataclass
class IgraphInfo:
    table_name: str = ""
    item_id: str = ""


@add_pydantic_compatibility
@dataclass
class ImageURL:
    url: str = ""
    detail: Optional[str] = "auto"

    def __post_init__(self):
        if self.detail is None:
            self.detail = "auto"


@add_pydantic_compatibility
@dataclass
class AudioURL:
    url: str = ""


@add_pydantic_compatibility
@dataclass
class ContentPart:
    type: ContentPartTypeEnum = ""
    text: Optional[str] = None
    igraph: Optional[IgraphInfo] = None
    image_url: Optional[ImageURL] = None
    video_url: Optional[ImageURL] = None
    audio_url: Optional[AudioURL] = None
    preprocess_config: Optional[MMPreprocessConfigPart] = None

    def __post_init__(self):
        if self.igraph is None:
            self.igraph = None
        if self.image_url is None:
            self.image_url = None
        if self.video_url is None:
            self.video_url = None
        if self.audio_url is None:
            self.audio_url = None
        if self.preprocess_config is None:
            self.preprocess_config = None


@add_pydantic_compatibility
@dataclass
class ChatMessage:
    role: RoleEnum = ""
    content: Union[str, None, List[ContentPart]] = ""
    reasoning_content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    partial: Optional[bool] = False
    tool_call_id: Optional[str] = None

    def __post_init__(self):
        if self.content is None:
            self.content = ""
        if self.function_call is None:
            self.function_call = None
        if self.tool_calls is None:
            self.tool_calls = None
        if self.partial is None:
            self.partial = False
        if self.tool_call_id is None:
            self.tool_call_id = None


# NOTE: according to openai api definition, `function_call` is deprecated, and replaced by `tool_calls`.
# see `openai/types/chat/chat_completion_chunk.py`

# TODO: maybe also implement Qwen Style function call.
# See https://github.com/QwenLM/Qwen/blob/35023b6f2a932bde6ed27f21ec03164ccf09a25f/examples/function_call_examples.py#L47


@add_pydantic_compatibility
@dataclass
class GPTFunctionDefinition:
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    # These parameters are for qwen style function.
    name_for_model: Optional[str] = None
    name_for_human: Optional[str] = None
    description_for_model: Optional[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.name_for_model is None:
            self.name_for_model = None
        if self.name_for_human is None:
            self.name_for_human = None
        if self.description_for_model is None:
            self.description_for_model = None


@add_pydantic_compatibility
@dataclass
class GPTToolDefinition:
    # 目前仅考虑type为function的tool
    type: str = "function"
    function: GPTFunctionDefinition = field(default_factory=GPTFunctionDefinition)

    def __post_init__(self):
        if self.function is None:
            self.function = GPTFunctionDefinition()


@add_pydantic_compatibility
@dataclass
class ChatCompletionRequest:
    model: Optional[str] = None
    messages: List[ChatMessage] = field(default_factory=list)
    functions: Optional[List[GPTFunctionDefinition]] = None
    tools: Optional[List[GPTToolDefinition]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = field(default_factory=list)
    stream: Optional[bool] = False
    user: Optional[str] = None
    seed: Optional[int] = None
    n: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    # ---- These functions are not implemented yet.
    # presence_penalty: Optional[float] = 0.0
    # frequency_penalty: Optional[float] = 0.0
    # logit_bias: Optional[Dict[str, float]] = None

    # ---- These params are hacked for our framework, not standard.
    extra_configs: Optional[GenerateConfig] = None
    private_request: bool = False
    trace_id: Optional[str] = None
    chat_id: Optional[str] = None
    template_key: Optional[str] = None
    user_template: Optional[str] = None
    debug_info: Optional[bool] = False
    aux_info: Optional[bool] = True
    extend_fields: Optional[Dict[str, Any]] = None  # This field is not effective, only for logging.
    master_info: Optional[Dict[str, Any]] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.functions is None:
            self.functions = None
        if self.tools is None:
            self.tools = None
        if self.stop is None:
            self.stop = []
        if self.stream is None:
            self.stream = False
        if self.temperature is None:
            self.temperature = 0.7
        if self.top_p is None:
            self.top_p = 1.0
        if self.logprobs is None:
            self.logprobs = None
        if self.top_logprobs is None:
            self.top_logprobs = None
        if self.extend_fields is None:
            self.extend_fields = None
        if self.master_info is None:
            self.master_info = None
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = None

    @staticmethod
    def is_openai_request(request: Dict[str, Any]):
        return "messages" in request

    def get_chat_template_kwargs(self):
        if (
            self.extra_configs is not None
            and self.extra_configs.chat_template_kwargs is not None
        ):
            return self.extra_configs.chat_template_kwargs
        else:
            return self.chat_template_kwargs

    def disable_thinking(self):
        if (
            self.get_chat_template_kwargs() is not None
            and self.get_chat_template_kwargs().get("enable_thinking", True) is False
        ):
            return True
        else:
            return False

    def to_dict(self, *args, **kwargs):
        return self.dict(*args, **kwargs)

    

@add_pydantic_compatibility
@dataclass
class CompletionTokensDetails:
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    def __post_init__(self):
        if self.audio_tokens is None:
            self.audio_tokens = None
        if self.reasoning_tokens is None:
            self.reasoning_tokens = None


@add_pydantic_compatibility
@dataclass
class PromptTokensDetails:
    audio_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    def __post_init__(self):
        if self.audio_tokens is None:
            self.audio_tokens = None
        if self.cached_tokens is None:
            self.cached_tokens = None


@add_pydantic_compatibility
@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    completion_tokens_details: Optional[CompletionTokensDetails] = None
    prompt_tokens_details: Optional[PromptTokensDetails] = None

    def __post_init__(self):
        if self.completion_tokens is None:
            self.completion_tokens = 0
        if self.completion_tokens_details is None:
            self.completion_tokens_details = None
        if self.prompt_tokens_details is None:
            self.prompt_tokens_details = None


@add_pydantic_compatibility
@dataclass
class TopLogprob:
    token: str = ""
    bytes: Optional[List[int]] = None
    logprob: float = 0.0

    def __post_init__(self):
        if self.bytes is None:
            self.bytes = None


@add_pydantic_compatibility
@dataclass
class ChatCompletionTokenLogprob:
    token: str = ""
    bytes: Optional[List[int]] = None
    logprob: float = 0.0
    top_logprobs: List[TopLogprob] = field(default_factory=list)

    def __post_init__(self):
        if self.bytes is None:
            self.bytes = None
        if self.top_logprobs is None:
            self.top_logprobs = []


@add_pydantic_compatibility
@dataclass
class ChoiceLogprobs:
    content: Optional[List[ChatCompletionTokenLogprob]] = None
    refusal: Optional[List[ChatCompletionTokenLogprob]] = None

    def __post_init__(self):
        if self.content is None:
            self.content = None
        if self.refusal is None:
            self.refusal = None


class FinisheReason(str, Enum):
    stop = "stop"
    length = "length"
    function_call = "function_call"
    tool_calls = "tool_calls"


@add_pydantic_compatibility
@dataclass
class RendererInfo:
    class_name: str = ""
    renderer_model_type: str = ""
    extra_stop_word_ids_list: List[List[int]] = field(default_factory=list)
    extra_stop_words_list: List[str] = field(default_factory=list)
    template: Optional[Union[str, Dict[str, str]]] = None

    def __post_init__(self):
        if self.extra_stop_word_ids_list is None:
            self.extra_stop_word_ids_list = []
        if self.extra_stop_words_list is None:
            self.extra_stop_words_list = []
        if self.template is None:
            self.template = None


@add_pydantic_compatibility
@dataclass
class DebugInfo:
    input_prompt: str = ""
    input_ids: List[int] = field(default_factory=list)
    input_urls: List[str] = field(default_factory=list)
    tokenizer_info: str = ""
    max_seq_len: int = 0
    eos_token_id: Optional[int] = None
    stop_word_ids_list: List[List[int]] = field(default_factory=list)
    stop_words_list: List[str] = field(default_factory=list)
    renderer_info: RendererInfo = field(default_factory=RendererInfo)
    generate_config: GenerateConfig = None

    def __post_init__(self):
        if self.input_ids is None:
            self.input_ids = []
        if self.input_urls is None:
            self.input_urls = []
        if self.stop_word_ids_list is None:
            self.stop_word_ids_list = []
        if self.stop_words_list is None:
            self.stop_words_list = []
        if self.renderer_info is None:
            self.renderer_info = RendererInfo()
        # Note: generate_config 可能需要从其他地方获取默认值


@add_pydantic_compatibility
@dataclass
class ChatCompletionResponseChoice:
    index: int = 0
    message: ChatMessage = field(default_factory=ChatMessage)
    finish_reason: Optional[FinisheReason] = None
    logprobs: Optional[ChoiceLogprobs] = None

    def __post_init__(self):
        if self.message is None:
            self.message = ChatMessage()
        if self.finish_reason is None:
            self.finish_reason = None
        if self.logprobs is None:
            self.logprobs = None


@add_pydantic_compatibility
@dataclass
class ChatCompletionExtraOutputs:
    hidden_states: Optional[Union[List[float], List[List[float]]]] = None
    all_hidden_states: Optional[Union[List[float], List[List[float]]]] = None
    loss: Optional[Union[float, List[float]]] = None
    logits: Optional[Union[List[float], List[List[float]]]] = None
    output_ids: Optional[List[List[int]]] = None
    input_ids: Optional[List[List[int]]] = None

    def __post_init__(self):
        if self.hidden_states is None:
            self.hidden_states = None
        if self.loss is None:
            self.loss = None
        if self.logits is None:
            self.logits = None
        if self.output_ids is None:
            self.output_ids = None
        if self.input_ids is None:
            self.input_ids = None


@add_pydantic_compatibility
@dataclass
class ChatCompletionResponse:
    id: str = field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionResponseChoice] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)
    debug_info: Optional[Union[DebugInfo, str]] = None
    aux_info: Optional[AuxInfo] = None
    extra_outputs: Optional[ChatCompletionExtraOutputs] = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = []
        if self.usage is None:
            self.usage = UsageInfo()
        if self.debug_info is None:
            self.debug_info = None
        if self.aux_info is None:
            self.aux_info = None
        if self.extra_outputs is None:
            self.extra_outputs = None


@add_pydantic_compatibility
@dataclass
class DeltaMessage:
    role: Optional[RoleEnum] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None

    def __post_init__(self):
        if self.role is None:
            self.role = None
        if self.content is None:
            self.content = None
        if self.reasoning_content is None:
            self.reasoning_content = None
        if self.function_call is None:
            self.function_call = None
        if self.tool_calls is None:
            self.tool_calls = None


@add_pydantic_compatibility
@dataclass
class ChatCompletionResponseStreamChoice:
    index: int = 0
    delta: DeltaMessage = field(default_factory=DeltaMessage)
    finish_reason: Optional[FinisheReason] = None
    logprobs: Optional[ChoiceLogprobs] = None

    def __post_init__(self):
        if self.delta is None:
            self.delta = DeltaMessage()
        if self.finish_reason is None:
            self.finish_reason = None
        if self.logprobs is None:
            self.logprobs = None


@add_pydantic_compatibility
@dataclass
class ChatCompletionStreamResponse:
    id: str = field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: Optional[str] = None
    choices: List[ChatCompletionResponseStreamChoice] = field(default_factory=list)
    usage: Optional[UsageInfo] = field(default=None)
    debug_info: Optional[Union[DebugInfo, str]] = None
    aux_info: Optional[AuxInfo] = None
    extra_outputs: Optional[ChatCompletionExtraOutputs] = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = []
        if self.usage is None:
            self.usage = None
        if self.debug_info is None:
            self.debug_info = None
        if self.aux_info is None:
            self.aux_info = None
        if self.extra_outputs is None:
            self.extra_outputs = None
