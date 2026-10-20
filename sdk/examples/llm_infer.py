import argparse
import copy
import os
from typing import Mapping, Optional

import torch
from mlsdk import CacheOptions, Context, MNDevice, set_tensor_name_in_module, storage
from mlsdk.experimental.llm.attention_mask import (
    prepare_4d_causal_attention_mask_with_cache_position,
)
from mlsdk.experimental.llm.kv_cache import (
    kv_cache_to_legacy,
    kv_cache_to_plamo,
    kv_cache_to_tensor,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_prompt(tokenizer, prompt, system_prompt):
    if tokenizer.chat_template:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def infer_with_generate(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt")
    # Greedy decoding for simplicity for comparing the results with the compiled version.
    output_ids = model.generate(
        inputs["input_ids"], do_sample=False, max_new_tokens=max_new_tokens
    )
    assert isinstance(output_ids, torch.Tensor)
    return output_ids


def infer_with_compilation(  # NOQA: CFQ002, CFQ001, CFQ004
    *,
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_new_tokens: int,
    run_prefill_on_cpu: bool,
    run_decode_on_cpu: bool,
    device_name: str,
    outdir: str,
    check_intermediate_outputs: bool,
    prepare_attention_mask_on_cpu: bool,
    disable_cache: bool,
    num_compiler_threads: int,
    load_codegen_prefill: Optional[str],
    load_codegen_decode: Optional[str],
) -> torch.Tensor:
    # When we recompile a model, delete the old decompressed "model.app"
    if load_codegen_prefill is None:
        if os.path.exists(outdir + "/prefill/model.app"):
            os.remove(outdir + "/prefill/model.app")
    if load_codegen_decode is None:
        if os.path.exists(outdir + "/decode/model.app"):
            os.remove(outdir + "/decode/model.app")

    is_plamo_model = any("plamo" in a.lower() for a in model.config.architectures)

    # Sharing layouts between the prefill and decode models can cause layout
    # decision issues in the compiler. We create a shallow copy so that each model
    # can use a different layout. We can also use `copy.deepcopy` instead
    # if the additional host memory usage is acceptable.
    def shallow_copy_model_with_copied_tensors(model):
        new_model = copy.copy(model)
        new_model._modules = model._modules.copy()
        new_model._parameters = model._parameters.copy()
        new_model._buffers = model._buffers.copy()

        for name, child in model._modules.items():
            if child is not None:
                new_model._modules[name] = shallow_copy_model_with_copied_tensors(child)
        for name, param in model._parameters.items():
            if param is not None:
                new_model._parameters[name] = copy.copy(param)
        for name, buffer in model._buffers.items():
            if buffer is not None:
                new_model._buffers[name] = copy.copy(buffer)

        return new_model

    prefill_model = model
    decode_model = shallow_copy_model_with_copied_tensors(prefill_model)

    # Check if object is different while its storage is shared
    decode_model_named_paramters = dict(decode_model.named_parameters())
    for name, p1 in prefill_model.named_parameters():
        p2 = decode_model_named_paramters[name]
        assert p1 is not p2
        assert p1.data_ptr() == p2.data_ptr(), name

    def forward_prefill(
        inputs: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        assert all(isinstance(v, torch.Tensor) for v in inputs.values()), {
            k: type(v) for k, v in inputs.items()
        }
        outputs = prefill_model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            past_key_values=None,
            use_cache=True,
        )
        next_input_ids = outputs["logits"].argmax(dim=2)[:, -1:]
        return {
            "next_input_ids": next_input_ids,
            "next_past_key_values": kv_cache_to_tensor(outputs.past_key_values)[
                :, :, :, :, 1:, :
            ],  # Do every operation, including the shifting, for the KV cache on device.
        }

    def forward_decode(
        inputs: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        assert all(isinstance(v, torch.Tensor) for v in inputs.values()), {
            k: type(v) for k, v in inputs.items()
        }
        if is_plamo_model:
            kv_cache_func = kv_cache_to_plamo
        else:
            # @todo (hvy): Stop using the deprecated legacy KV cache format of tuples.
            kv_cache_func = kv_cache_to_legacy
        past_key_values = kv_cache_func(inputs["past_key_values"])

        outputs = decode_model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_input_ids = outputs["logits"].argmax(dim=2)[:, -1:]
        return {
            "next_input_ids": next_input_ids,
            "next_past_key_values": kv_cache_to_tensor(outputs.past_key_values)[
                :, :, :, :, 1:, :
            ],  # Do every operation, including the shifting, for the KV cache on device.
        }

    assert tokenizer.padding_side == "left"
    inputs = tokenizer(
        prompt, return_tensors="pt", padding="max_length", max_length=max_length
    )
    # @todo (hvy): Consider subtracting 1 from the position_ids to match modeling_llama.py.
    assert "position_ids" not in inputs
    inputs["position_ids"] = inputs["attention_mask"].cumsum(1)
    if prepare_attention_mask_on_cpu:
        inputs["attention_mask"] = prepare_4d_causal_attention_mask_with_cache_position(
            inputs["attention_mask"], inputs["position_ids"], model.dtype
        )
    output_ids = inputs["input_ids"]

    device = MNDevice(device_name)
    context = Context(device)
    Context.switch_context(context)
    set_tensor_name_in_module(prefill_model, "prefill_model")
    for p in prefill_model.parameters():
        context.register_param(p)
    for b in prefill_model.buffers():
        context.register_buffer(b)
    set_tensor_name_in_module(decode_model, "decode_model")
    for p in decode_model.parameters():
        context.register_param(p)
    for b in decode_model.buffers():
        context.register_buffer(b)

    compiled_funcs = {}
    if load_codegen_prefill is not None:
        compiled_funcs["prefill"] = context.load_codegen_dir(
            storage.path(load_codegen_prefill)
        )
        input_proxies_allocated = compiled_funcs["prefill"].allocate_input_proxy()
        input_proxies_allocated["input_ids"].load_from(
            inputs["input_ids"].detach(), clone=False
        )
        input_proxies_allocated["attention_mask"].load_from(
            inputs["attention_mask"].detach(), clone=False
        )
        input_proxies_allocated["position_ids"].load_from(
            inputs["position_ids"].detach(), clone=False
        )
        for model_param in prefill_model.parameters():
            model_param_proxy = context.get_registered_value_proxy(model_param)
            model_param_proxy.load_from(model_param)

    decode_input_proxies = None
    if load_codegen_decode is not None:
        compiled_funcs["decode"] = context.load_codegen_dir(
            storage.path(load_codegen_decode)
        )
        decode_input_proxies = compiled_funcs["decode"].allocate_input_proxy()

    for step in range(max_new_tokens):
        if step == 0:
            if not run_prefill_on_cpu and "prefill" not in compiled_funcs:
                # Set codegen internal flags
                options = dict()
                options["envs"] = {
                    "CODEGEN_TIME_SLICE_SCATTERED_INDEXING_BCAST": "1",
                    "CODEGEN_OP_DEF": "Gather=GatherBcast",
                }
                compiled_funcs["prefill"] = context.compile(
                    forward_prefill,
                    inputs,
                    storage.path(outdir + "/prefill"),
                    options=options,
                    cache_options=(
                        CacheOptions(outdir + "/prefill_cache")
                        if not disable_cache
                        else None
                    ),
                    num_compiler_threads=num_compiler_threads,
                )
            forward_for_step = compiled_funcs.get("prefill", forward_prefill)
        else:
            if not run_decode_on_cpu and "decode" not in compiled_funcs:
                compiled_funcs["decode"] = context.compile(
                    forward_decode,
                    inputs,
                    storage.path(outdir + "/decode"),
                    cache_options=(
                        CacheOptions(outdir + "/decode_cache")
                        if not disable_cache
                        else None
                    ),
                    num_compiler_threads=num_compiler_threads,
                    io_spec_constraints=(
                        {
                            "past_key_values": compiled_funcs["prefill"].output_specs[
                                "next_past_key_values"
                            ],
                            # This constraint is to share the layout between decode's
                            # `past_key_values` and `next_past_key_values`.
                            "next_past_key_values": compiled_funcs[
                                "prefill"
                            ].output_specs["next_past_key_values"],
                        }
                        if not run_prefill_on_cpu
                        else None
                    ),
                )
            forward_for_step = compiled_funcs.get("decode", forward_decode)

        outputs = forward_for_step(inputs)

        if check_intermediate_outputs:
            # @todo (hvy): Consider using a more sophisticated check for the outputs.
            if "mncore" in device_name:
                atol = 1.0
            else:
                assert device_name == "pfvm:cpu"
                atol = 5e-3
            n_tokens = inputs["position_ids"].max()
            forward = forward_prefill if step == 0 else forward_decode
            # We treat `past_key_values` as a `TensorProxy` to avoid sending it to the
            # host. Therefore, to perform computations on the host, we need to transfer
            # them explicity.
            inputs_on_host = (
                inputs
                if step == 0
                else {
                    **inputs,
                    "past_key_values": inputs["past_key_values"].cpu().detach(),
                }
            )
            outputs_expected = forward(inputs_on_host)
            # Since we compute argmax inside the compiled function, we only output the
            # last token. For a more accurate comparison, we may need to output the
            # full `logits`.
            next_input_ids = outputs["next_input_ids"]
            next_input_ids_expected = outputs_expected["next_input_ids"]
            next_past_key_values = outputs["next_past_key_values"][
                :, :, :, :, -n_tokens:
            ]
            next_past_key_values_expected = outputs_expected["next_past_key_values"][
                :, :, :, :, -n_tokens:
            ]

            assert torch.allclose(next_input_ids, next_input_ids_expected, atol=atol), (
                step,
                (next_input_ids - next_input_ids_expected).abs().max(),
            )
            assert torch.allclose(
                next_past_key_values, next_past_key_values_expected, atol=atol
            ), (
                step,
                (next_past_key_values - next_past_key_values_expected).abs().max(),
            )

        next_input_ids = outputs["next_input_ids"].cpu()
        if prepare_attention_mask_on_cpu:
            next_attention_mask = inputs["attention_mask"][:, :, -1:, :]
            next_attention_mask = torch.roll(next_attention_mask, shifts=-1, dims=-1)
            next_attention_mask[:, :, :, -1] = 0
        else:
            next_attention_mask = inputs["attention_mask"]
            next_attention_mask = torch.roll(next_attention_mask, shifts=-1, dims=-1)
            next_attention_mask[:, -1] = 1
        next_position_ids = inputs["position_ids"][:, -1:] + 1
        next_past_key_values = outputs["next_past_key_values"]
        inputs = {
            "input_ids": next_input_ids.detach(),
            "attention_mask": next_attention_mask.detach(),
            "position_ids": next_position_ids.detach(),
            "past_key_values": next_past_key_values,
        }

        output_ids = torch.cat([output_ids, next_input_ids], dim=1)

        if next_input_ids.item() == tokenizer.eos_token_id:
            break

        if compiled_funcs.get("decode") is not None:
            if decode_input_proxies is None:
                decode_input_proxies = compiled_funcs["decode"].allocate_input_proxy()
            decode_input_proxies["input_ids"].load_from(
                next_input_ids.detach(), clone=False
            )
            decode_input_proxies["attention_mask"].load_from(
                next_attention_mask.detach(), clone=False
            )
            decode_input_proxies["position_ids"].load_from(
                next_position_ids.detach(), clone=False
            )

    return output_ids[:, max_new_tokens:]


def main(args):
    prompt = args.prompt
    system_prompt = args.system_prompt
    model_name = args.model_name
    max_length = args.max_length
    max_new_tokens = args.max_new_tokens
    run_prefill_on_cpu = args.run_prefill_on_cpu
    run_decode_on_cpu = args.run_decode_on_cpu
    device_name = args.device_name
    outdir = args.outdir
    check_intermediate_outputs = args.check_intermediate_outputs
    prepare_attention_mask_on_cpu = args.prepare_attention_mask_on_cpu
    disable_cache = args.disable_cache
    num_compiler_threads = args.num_compiler_threads
    load_codegen_prefill = args.load_codegen_prefill
    load_codegen_decode = args.load_codegen_decode

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()  # Some models do not return the KV cache in training mode.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # For static KV caching
    tokenizer.truncation_side = "left"

    prompt = prepare_prompt(tokenizer, prompt, system_prompt)

    outputs = infer_with_compilation(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        run_prefill_on_cpu=run_prefill_on_cpu,
        run_decode_on_cpu=run_decode_on_cpu,
        device_name=device_name,
        outdir=outdir,
        check_intermediate_outputs=check_intermediate_outputs,
        prepare_attention_mask_on_cpu=prepare_attention_mask_on_cpu,
        disable_cache=disable_cache,
        num_compiler_threads=num_compiler_threads,
        load_codegen_prefill=load_codegen_prefill,
        load_codegen_decode=load_codegen_decode,
    )
    print(
        "=========== Generated with compilation ==========\n",
        tokenizer.decode(outputs[0]),
    )

    outputs_expected = infer_with_generate(prompt, model, tokenizer, max_new_tokens)
    print(
        "========== Generated with model.generate ==========\n",
        tokenizer.decode(outputs_expected[0]),
    )

    # @todo (hvy): Do not rely on `max_new_tokens` tokens always being generated?
    assert torch.equal(
        outputs[:, -max_new_tokens:], outputs_expected[:, -max_new_tokens:]
    ), "Outputs differed. Check generated outputs above."
    print("Generated outputs matched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default='The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01.',  # NOQA
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a friendly chatbot who is an expert on MN-Core.",
    )
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_compiler_threads", type=int, default=-1)
    parser.add_argument("--run_prefill_on_cpu", action="store_true")
    parser.add_argument("--run_decode_on_cpu", action="store_true")
    parser.add_argument("--device_name", type=str, default="mncore2:auto")
    parser.add_argument(
        "--outdir",
        type=str,
        default="/tmp/mlsdk_llm_infer",
        help="""
        Output directory. Compilation results for prefill and decode are stored in the
        `/prefill` and `/decode` subdirectories, respectively.
        """,
    )
    parser.add_argument("--check_intermediate_outputs", action="store_true")
    parser.add_argument("--prepare_attention_mask_on_cpu", action="store_true")
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--load_codegen_prefill", type=str, default=None)
    parser.add_argument("--load_codegen_decode", type=str, default=None)
    args = parser.parse_args()
    main(args)
