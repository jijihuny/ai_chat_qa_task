from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerateBeamDecoderOnlyOutput
from torch.nn.functional import softmax
from typing import Union, Dict, Literal
from numpy import ndarray
from arguments import GenerationConfig


def get_beam_search_sequences(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, inputs: list[str], **kwargs
) -> list[Dict[Literal["generated_texts", "scores"], Union[list[str], ndarray]]]:
    if kwargs.get("return_full_text"):
        kwargs["return_full_text"] = None
    if kwargs.get("return_dict_in_generate") != True:
        UserWarning("return_dict_in_generate set False")
        kwargs["return_dict_in_generate"] = True

    num_return_sequences = kwargs.get("num_return_sequences")
    if (not isinstance(num_return_sequences, int)) or num_return_sequences < 1:
        num_return_sequences = 1
    num_inputs = len(inputs)
    inputs = tokenizer(inputs, padding="longest", return_tensors="pt")
    length = inputs.input_ids.shape[-1]
    output: GenerateBeamDecoderOnlyOutput = model.generate(
        **inputs.to(model.device), **kwargs
    )

    sequences = tokenizer.batch_decode(output.sequences[:, length:])
    scores = (
        softmax(output.sequences_scores.view(-1, num_return_sequences), dim=-1)
        .cpu()
        .numpy()
    )

    results = []
    for i in num_inputs:
        start = int(i * num_return_sequences)
        end = start + num_return_sequences
        results += [{"generated_texts": sequences[start:end], "scores": scores[i]}]

    return results
