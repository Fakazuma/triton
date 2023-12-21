from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import tritonclient.utils as trutils
from functools import lru_cache
import numpy as np
from transformers import AutoTokenizer
from export_model import TextHeadWithProj
import torch
import onnxruntime as ort

# seed = 0
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_tirton_ensemble_onnx(text: str) -> np.ndarray:
    triton_client = get_client()
    text = np.array([text.encode("utf-8")], dtype=object)

    input_text = InferInput(
        name="TEXTS", shape=text.shape, datatype=trutils.np_to_triton_dtype(text.dtype)
    )
    input_text.set_data_from_numpy(text, binary_data=True)

    infer_response = triton_client.infer(
        "ensemble-onnx",
        [input_text],
        outputs=[InferRequestedOutput("EMBEDDINGS", binary_data=True)],
    )
    embeddings = infer_response.as_numpy("EMBEDDINGS")[0]
    return embeddings


def call_triton_onnx(ids: np.ndarray[int], attention_mask: np.ndarray[int]) -> np.ndarray:
    triton_client = get_client()

    ids = ids.reshape(1, -1)
    attention_mask = attention_mask.reshape(1, -1)

    input_ids = InferInput(
        name="INPUT_IDS", shape=list(ids.shape), datatype=trutils.np_to_triton_dtype(ids.dtype)
    )
    input_ids.set_data_from_numpy(ids, binary_data=True)

    input_attention_mask = InferInput(
        name="ATTENTION_MASK",
        shape=list(attention_mask.shape),
        datatype=trutils.np_to_triton_dtype(attention_mask.dtype)
    )
    input_attention_mask.set_data_from_numpy(attention_mask, binary_data=True)

    infer_response = triton_client.infer(
        "onnx-rubert",
        [input_ids, input_attention_mask],
        outputs=[InferRequestedOutput("EMBEDDINGS", binary_data=True)],
    )
    embeddings = infer_response.as_numpy("EMBEDDINGS")[0]
    return embeddings


def call_triton_tokenizer(text: str) -> np.ndarray:
    triton_client = get_client()
    text = np.array([text.encode("utf-8")], dtype=object)

    input_text = InferInput(
        name="TEXTS", shape=text.shape, datatype=trutils.np_to_triton_dtype(text.dtype)
    )
    input_text.set_data_from_numpy(text, binary_data=True)

    infer_response = triton_client.infer(
        "python-tokenizer",
        [input_text],
        outputs=[
            InferRequestedOutput("INPUT_IDS", binary_data=True),
            InferRequestedOutput("ATTENTION_MASK", binary_data=True),
        ],
    )
    input_ids = infer_response.as_numpy("INPUT_IDS")[0]
    attention_mask = infer_response.as_numpy("ATTENTION_MASK")[0]
    return input_ids, attention_mask


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
        else:
            print(type(module))
        # if isinstance(module, old):
        #     ## simple module
        #     setattr(model, n, new)


def main():
    texts = [
        "Сегодня солнечный день",
        "Сегодня пасмурный день",
        "Датасаенс -- это не эмэль",
        "Стальная ложка",
    ]

    model = TextHeadWithProj('ai-forever/ruBert-base', 96).to('cpu')
    model.load_state_dict(torch.load('./model.pt'))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
    encoded = tokenizer(
        texts[0],
        padding="max_length",
        max_length=16,
        truncation=True,
    )

    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    input_ids = np.array(input_ids)
    attention_mask = np.array(attention_mask)

    _input_ids, _attention_mask = call_triton_tokenizer(texts[0])
    _input_ids = np.array(_input_ids)
    _attention_mask = np.array(_attention_mask)

    assert (input_ids == _input_ids).all() and (attention_mask == _attention_mask).all()

    embeddings = call_triton_onnx(_input_ids, _attention_mask)
    gt_out = model(
        torch.tensor(_input_ids.reshape(1, -1), dtype=torch.int64, device="cpu"),
        torch.tensor(_attention_mask.reshape(1, -1), dtype=torch.int64, device="cpu"),
    )

    assert np.allclose(embeddings, gt_out.detach().numpy(), atol=1e-5)


if __name__ == "__main__":
    main()
