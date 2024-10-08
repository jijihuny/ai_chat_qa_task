from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerateBeamDecoderOnlyOutput
from torch.nn.functional import softmax
from typing import Union, Dict, Literal
from numpy import ndarray
from torch import cuda


def get_beam_search_sequences(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, inputs: list[str], **kwargs
) -> list[Dict[Literal["generated_texts", "scores"], Union[list[str], ndarray]]]:
    r"""

    ```python
    from functools import partial

    generate = partial(get_beam_search_sequences, model=model, tokenizer=tokenizer)

    output = generate(inputs=inputs)
    ```

    """
    # only pipeline arg
    kwargs.pop("return_full_text")
    if kwargs.get("return_dict_in_generate") != True:
        UserWarning("return_dict_in_generate set False")
        kwargs["return_dict_in_generate"] = True
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    num_return_sequences = kwargs.get("num_return_sequences")
    if (not isinstance(num_return_sequences, int)) or num_return_sequences < 1:
        num_return_sequences = 1
    num_inputs = len(inputs)
    inputs = tokenizer(inputs, padding="longest", return_tensors="pt")
    length = inputs.input_ids.shape[-1]
    output: GenerateBeamDecoderOnlyOutput = model.generate(
        **inputs.to(model.device),
        **kwargs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    sequences = tokenizer.batch_decode(
        output.sequences[:, length:], skip_special_tokens=True
    )
    scores = (
        softmax(output.sequences_scores.view(-1, num_return_sequences), dim=-1)
        .cpu()
        .numpy()
    )

    results = []
    for i in range(num_inputs):
        start = int(i * num_return_sequences)
        end = start + num_return_sequences
        results += [{"generated_texts": sequences[start:end], "scores": scores[i]}]

    if cuda.is_available():
        cuda.empty_cache()

    return results
