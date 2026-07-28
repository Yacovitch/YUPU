import hashlib
import os
import urllib
import warnings
from types import SimpleNamespace
from typing import Union, List, Optional

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from lidiff.models.model import build_model
from lidiff.models.simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


__all__ = ["available_models", "load", "tokenize"]

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

_LOCAL_WEIGHTS_ROOT = os.environ.get("CLIP_LOCAL_WEIGHTS_ROOT", "/nas2/jacob/LiDiff_HPC/clip_weights")

_HF_MODEL_ALIASES = {
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
}


class HuggingFaceCLIPWrapper(torch.nn.Module):
    _MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def __init__(self, model, processor, device: Union[str, torch.device]):
        super().__init__()
        self.model = model
        self.processor = processor
        target_device = torch.device(device)
        self.model.to(target_device)
        self.visual = SimpleNamespace(input_resolution=self._infer_input_resolution())

    def _infer_input_resolution(self) -> int:
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor and hasattr(image_processor, "crop_size"):
            crop_size = image_processor.crop_size
            if isinstance(crop_size, dict):
                return crop_size.get("height", crop_size.get("width", 224))
            if isinstance(crop_size, (list, tuple)):
                return crop_size[0]
            return int(crop_size)
        # fallback to config default
        return getattr(self.model.config.vision_config, "image_size", 224)

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def _prepare_pixel_values(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image tensor, got {type(image)}")

        pixel_values = image.to(self._device, dtype=self.dtype)
        mean = self._MEAN.to(device=pixel_values.device, dtype=pixel_values.dtype)
        std = self._STD.to(device=pixel_values.device, dtype=pixel_values.dtype)
        pixel_values = (pixel_values - mean) / std
        return pixel_values

    def encode_image(self, image: torch.Tensor):
        pixel_values = self._prepare_pixel_values(image)
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        hidden_states = vision_outputs.last_hidden_state  # [B, 1+N, hidden]

        proj_weight = self.model.visual_projection.weight
        projected = hidden_states @ proj_weight.t()
        if self.model.visual_projection.bias is not None:
            projected = projected + self.model.visual_projection.bias

        global_features = projected[:, 0, :]
        patch_features = projected[:, 1:, :]
        return global_features, patch_features

    def encode_text(self, text: torch.Tensor):
        if text.dim() != 2:
            raise ValueError(f"Expected 2D token tensor, got shape {tuple(text.shape)}")
        input_ids = text.to(self._device, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = text_outputs.last_hidden_state
        idx = input_ids.argmax(dim=-1)
        pooled = hidden_states[torch.arange(hidden_states.size(0)), idx]
        proj_weight = self.model.text_projection.weight
        text_features = pooled @ proj_weight.t()
        if self.model.text_projection.bias is not None:
            text_features = text_features + self.model.text_projection.bias
        return text_features

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features, _ = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def _resolve_hf_local_path(model_id: str) -> Optional[str]:
    if os.path.isdir(model_id):
        return model_id

    if _LOCAL_WEIGHTS_ROOT:
        candidate = os.path.join(_LOCAL_WEIGHTS_ROOT, model_id.replace("/", os.sep))
        if os.path.isdir(candidate):
            return candidate

    return None


def _hf_image_preprocess(processor):
    def preprocess(image):
        processed = processor(images=image, return_tensors="pt")
        return processed["pixel_values"][0]

    return preprocess


def _try_load_local_weight(filename: str, expected_sha256: str):
    if not _LOCAL_WEIGHTS_ROOT:
        return None

    local_path = os.path.join(_LOCAL_WEIGHTS_ROOT, filename)
    if not os.path.isfile(local_path):
        return None

    if expected_sha256:
        with open(local_path, "rb") as candidate:
            actual_sha256 = hashlib.sha256(candidate.read()).hexdigest()
        if actual_sha256 != expected_sha256:
            warnings.warn(
                f"Local CLIP weight {local_path} does not match expected checksum; using it anyway"
            )

    return local_path


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]

    local_weight = _try_load_local_weight(filename, expected_sha256)
    if local_weight:
        return local_weight

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP sem_models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip_sem.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip_sem"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    hf_error: Optional[BaseException] = None
    if name in _HF_MODEL_ALIASES:
        model_id = _HF_MODEL_ALIASES[name]
        resolved_path = _resolve_hf_local_path(model_id) or model_id
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers package is required to load Hugging Face CLIP checkpoints"
            ) from exc

        load_kwargs = {}
        if os.path.isdir(resolved_path):
            load_kwargs["local_files_only"] = True

        try:
            model = CLIPModel.from_pretrained(resolved_path, **load_kwargs)
            processor = CLIPProcessor.from_pretrained(resolved_path, **load_kwargs)
            wrapped_model = HuggingFaceCLIPWrapper(model, processor, device)
            if str(device) == "cpu":
                wrapped_model.float()
            return wrapped_model, _hf_image_preprocess(processor)
        except Exception as exc:
            hf_error = exc
            warnings.warn(
                f"Falling back to legacy OpenAI CLIP loader for {name}: {exc}"
            )

    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        if hf_error:
            raise RuntimeError(f"Failed to load Hugging Face CLIP model '{name}': {hf_error}") from hf_error
        raise RuntimeError(f"Model {name} not found; available sem_models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=str(device) if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            opened_file.seek(0)
            state_dict = torch.load(opened_file, map_location="cpu")
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    if not jit:
        model = build_model(state_dict or model.state_dict(), str(device))#.to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.script(lambda: torch.ones([]), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.script(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False, config_path: str = None) -> Union[
    torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP sem_models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    config_path: str
        Path to the config file containing the BPE vocabulary path

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    # Initialize tokenizer with config path
    tokenizer = _Tokenizer(config_path=config_path)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def load_clip_to_cpu(backbone_name):
    url = _MODELS[backbone_name]
    model_path = _download(url, os.path.expanduser("~/.cache/clip"))
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = build_model(state_dict or model.state_dict())
    return model
