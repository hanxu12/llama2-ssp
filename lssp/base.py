# Base functions for sampling and streaming

import sys
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

llama7b_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = LlamaTokenizer.from_pretrained(llama7b_name)


def create_model(model_name, max_memory, load_in_8bit=True):
    return LlamaForCausalLM.from_pretrained(
        model_name,
        device_map='balanced',
        load_in_8bit = True,
        max_memory=max_memory
    )


def stream_token_if_required(input_ids, stream=False):
    if stream is True:
        output_string = tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True)
        previous_output_string = tokenizer.decode(
            input_ids[0][:-1],
            skip_special_tokens=True)
        sys.stdout.write(output_string[len(previous_output_string):])
        sys.stdout.flush()


TEMPERATURE = 0.5


def get_temperature_distribution(logits, temperature=TEMPERATURE):
    return torch.softmax(logits / temperature, dim=-1)


def sample_fn(logits, temperature=TEMPERATURE):
    probs = get_temperature_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_model(model,
                 input_ids,
                 nb_tokens,
                 display=False,
                 temperature=TEMPERATURE):
    print("input_id shape, ", input_ids.shape)
    for _ in range(nb_tokens):
        print("index: ", _)
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = sample_fn(next_token_logits, temperature)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        if _ > 1 and tokenizer.decode(input_ids[0], skip_special_tokens=True)[-1] == '\n':
            break
        stream_token_if_required(input_ids, stream=display)
    print("input_id shape, ", input_ids.shape)
    return input_ids
