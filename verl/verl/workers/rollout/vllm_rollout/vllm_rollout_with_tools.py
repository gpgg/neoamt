# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import importlib
import logging
import math
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Counter, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm.lora.request import LoRARequest

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import (get_response_mask,
                                         pad_sequence_to_length)
from verl.workers.rollout.tools.base_tool import BaseTool
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _pre_process_inputs, vLLMRollout)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """Dynamically loads a tool from its configuration."""
    module_path, class_name = tool_config["class_path"].rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        
        tool_class = getattr(module, class_name)
        
        # tool_params = OmegaConf.to_container(tool_config.get('params', {}), resolve=True)
        tool_params = tool_config.get('params', {})
        tool_instance = tool_class(**tool_params)
        
        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Failed to instantiate {class_name} with provided parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading tool from {tool_config.class_path}: {e}")
        raise


class vLLMRolloutWithTools(vLLMRollout):
    """
    An advanced vLLM rollout engine capable of handling multiple tools like
    code interpreters and search engines during generation.

    This class extends vLLMRollout by orchestrating a multi-step generation
    process where the language model can emit special tokens to trigger external
    tools. The tool outputs are then fed back into the model to continue
    generation.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        # breakpoint()
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        # breakpoint()
        # 从配置中获取beam search相关参数
        # self.initial_rollouts = self.config.get("initial_rollouts", self.config['n'])
        # self.beam_size = self.config.get("beam_size", 1)
        # self.branch_probability = self.config.get("branch_probability", 0.5)
        # self.entropy_weight = self.config.get("entropy_weight", 0.5)
        
        # 从配置中获取工具设置
        # tools_config = self.config.get("tools", OmegaConf.create({}))
        tools_config = self.config.tools
        print(f"Tools config: {tools_config}")
        # breakpoint()
        # 获取工具通用配置
        self.tool_call_limit = tools_config["call_limit"]
        self.max_tool_workers = tools_config["max_workers"]
        self.tool_timeout = tools_config["timeout"]

        # 其他可能的工具通用配置
        self.tool_retry_count = tools_config["retry_count"]
        self.tool_verbose_logging = tools_config["verbose_logging"]
        self.search_lang_side = tools_config["search_lang_side"] # "src", "tgt", "all", "none"

        
        self.tools: Dict[str, BaseTool] = {}
        if "tool_instances" in tools_config:
            for tool_name, tool_config in tools_config["tool_instances"].items():
                logger.info(f"Loading tool '{tool_name}' from {tool_config['class_path']}")
                try:
                    # breakpoint()
                    tool_instance = _load_tool_from_config(tool_config)
                    self.tools[tool_instance.trigger_tag] = tool_instance
                except Exception as e:
                    logger.error(f"Could not initialize tool '{tool_name}'. Please check your configuration. Error: {e}")
                    if tools_config.get("fail_on_error", False):
                        raise

        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        self.translation_stop_sequences = ["</translation>"]
        self.tag_token_ids = {tag: self.tokenizer.encode(tag, add_special_tokens=False)[0] for tag in self.tools.keys()} # !! we assume that the tag is not split into multiple tokens
        
        self.logprobs = 10 # entropy
        self.initial_entropy_dict = {}  # record initial entropy of active indice

        if not self.tools:
            logger.warning(
                "vLLMRolloutWithTools initialized, but no tools were configured.")

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tool_workers)

    def __del__(self):
        self.executor.shutdown(wait=False)

    def _extract_content(self, text: str, tag: str) -> str:
        """Extracts content from within the last <tag>...</tag> block."""
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            logger.warning(
                f"Could not extract content for tag '{tag}' from text: {text}")
            return ""

    def _execute_tool_with_retry(self, tool, content, lang=None):
        retry_count = 0
        start_time = time.time()
        success = False
        
        while retry_count < self.tool_retry_count:
            try:
                result_text = tool.execute(content, lang=lang)
                if result_text:
                    success = True
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text
                    }
                else:
                    logger.warning(f"Tool({tool.trigger_tag}) returned empty output. Retrying {retry_count + 1}/{self.tool_retry_count}")
                    retry_count += 1
            except Exception as e:
                logger.error(f"Tool({tool.trigger_tag}) execution failed. Retrying {retry_count + 1}/{self.tool_retry_count}: {e}")
                retry_count += 1
        
        execution_time = time.time() - start_time
        logger.warning(f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries. Appending EOS.")
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": ""
        }

    def _calc_entropy(self, logprobs):
        if not logprobs:
            return 0.0
        p_list = [math.exp(l) for l in logprobs]
        entropy = -sum(p * l for p, l in zip(p_list, logprobs))
        return entropy

    @GPUMemoryLogger(role="vllm rollout spmd with tools", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # breakpoint()
        input_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        
        qe = prompts.meta_info.get("qe", False)
        
        batch_size = input_ids.size(0)
        if prompts.meta_info.get("validate", False) or qe:
            dynamic_nums = [1] * batch_size
        # elif qe:
        #     dynamic_nums = [1] * batch_size
        else:
            dynamic_nums = prompts.non_tensor_batch.pop("dynamic_nums", [1] * batch_size)
        print(f"Dynamic nums: {dynamic_nums}")
        print(f"kwargs: {kwargs}")
        # 初始化工具调用统计信息
        tool_metrics = {
            "tools/total_calls": 0,
            "tools/successful_calls": 0,
            "tools/failed_calls": 0,
            "tools/total_execution_time": 0.0,
            "tools/avg_execution_time": 0.0,
            "tools/max_execution_time": 0.0,
            "tools/max_retries": 0,
            "tools/total_retries": 0,
            "tools/call_limit_reached_count": 0,
        }
        
        # 每个工具的统计信息
        calls_per_tool = Counter()
        success_per_tool = Counter()
        total_time_per_tool = Counter()

        non_tensor_batch = prompts.non_tensor_batch
        # get lps in non_tensor_batch
        _lps = [non_tensor_batch["extra_info"][i]["lp"] for i in range(batch_size)]
        
        lps = []
        for lp in _lps:
            if "_" in lp:
                lp = lp.split("_")[0]
            else:
                lp = lp
            lps.append(lp)
        
        src_langs = [lp.split("-")[0] for lp in lps]
        tgt_langs = [lp.split("-")[1] for lp in lps]
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, input_ids[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            prompt_token_ids_list = [vllm_input["prompt_token_ids"] for vllm_input in vllm_inputs]
            
            curr_inputs = []
            init_inputs = []
            result_masks = []
            call_counters = []
            active_indices = []
            curr_lps = []
            curr_src_langs = []
            curr_tgt_langs = []
            for i, ids in enumerate(prompt_token_ids_list):
                rollout_n = dynamic_nums[i]
                for _ in range(rollout_n):
                    curr_inputs.append(ids.copy())
                    init_inputs.append(ids.copy())
                    result_masks.append([])
                    call_counters.append(0)
                    active_indices.append(len(curr_inputs) - 1)
                    curr_lps.append(lps[i])
                    curr_src_langs.append(src_langs[i])
                    curr_tgt_langs.append(tgt_langs[i])
            
            # Track rollouts per original sample
            # rollouts_per_sample = [initial_rollouts] * batch_size  # 每个样本初始有initial_rollouts个rollout
            rollouts_per_sample = dynamic_nums # 每个样本有dynamic_nums[i]个rollout
            # 初始时每个样本有多个索引
            # sample_to_indices = {i: [i * initial_rollouts + j for j in range(initial_rollouts)] for i in range(batch_size)}
            sample_to_indices = {}
            cur_index = 0
            for i in range(batch_size):
                cur_n_rollouts = dynamic_nums[i]
                sample_to_indices[i] = [cur_index + j for j in range(cur_n_rollouts)]
                cur_index += cur_n_rollouts

            max_len = self.config.response_length
            
            iteration_num = 0
            while active_indices:
                # print(f"Iteration {iteration_num}, Active samples: {len(active_indices)}")
                iteration_num += 1
                active_prompts = [curr_inputs[i] for i in active_indices]
                # logger.debug(f"active_indices: {active_indices}")
                # logger.debug(f"active_prompts: {active_prompts}")
                # print("Generating outputs for active prompts...")
                # Update max_tokens for each active sample
                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences + self.translation_stop_sequences,
                    max_tokens=max(1, max((max_len - (len(curr_inputs[i]) - len(init_inputs[i])) for i in active_indices))),
                    detokenize=True,
                    logprobs = self.logprobs,
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )
                # print(f"Generated {len(outputs)} outputs.")
                # breakpoint()
                tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
                next_active_indices = []
                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    generated_tokens = output.outputs[0].token_ids
                    
                    curr_inputs[out_idx].extend(generated_tokens)
                    result_masks[out_idx].extend([1] * len(generated_tokens))
                    
                    finish_reason = output.outputs[0].finish_reason
                    stop_reason = output.outputs[0].stop_reason
                    
                    is_tool_call = finish_reason == 'stop' and stop_reason in self.stop_sequences
                    is_final_translation = finish_reason == 'stop' and stop_reason in self.translation_stop_sequences
                    
                    decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    logger.debug(f"  Sample {out_idx} output:")
                    # logger.debug(f"  Token IDs: {generated_tokens}")
                    logger.debug(f"  Current input: {self.tokenizer.decode(curr_inputs[out_idx], skip_special_tokens=False)}")
                    logger.debug(f"  Output: {decoded_text}")
                    logger.debug(f"  Finish reason: {finish_reason}")
                    logger.debug(f"  Stop reason: {stop_reason}")
                    logger.debug(f"  Is tool call: {is_tool_call}")
                    logger.debug(f"  Tool: {stop_reason.strip('</>') if is_tool_call else 'No tool call'}")
                    if is_tool_call:
                        tag = stop_reason.strip("</>")
                        if call_counters[out_idx] < self.tool_call_limit:
                            call_counters[out_idx] += 1
                            full_text = self.tokenizer.decode(curr_inputs[out_idx])
                            content = self._extract_content(full_text, tag)
                            if content:
                                tool_requests[tag].append({"index": out_idx, "content": content})
                                next_active_indices.append(out_idx)
                                # 更新工具调用计数统计
                                tool_metrics["tools/total_calls"] += 1
                                calls_per_tool[tag] += 1
                        else:
                            logger.warning(f"Tool call limit reached for sample {out_idx}. Appending EOS.")
                            # if eos_token_ids is a list of ids
                            # we extend
                            # else we append
                            if isinstance(eos_token_id, list):
                                curr_inputs[out_idx].extend(eos_token_id)
                                result_masks[out_idx].extend([1] * len(eos_token_id))
                            else:
                                curr_inputs[out_idx].append(eos_token_id)
                                result_masks[out_idx].append(1)
                            tool_metrics["tools/call_limit_reached_count"] += 1
                    
                    elif is_final_translation:
                        logger.warning(f"Generated the final translation. Appending EOS.")
                        if isinstance(eos_token_id, list):
                            curr_inputs[out_idx].extend(eos_token_id)
                            result_masks[out_idx].extend([1] * len(eos_token_id))
                        else:
                            curr_inputs[out_idx].append(eos_token_id)
                            result_masks[out_idx].append(1)
                    elif finish_reason == "length":
                        if len(curr_inputs[out_idx]) - len(init_inputs[out_idx]) < max_len:
                            next_active_indices.append(out_idx)
                    elif finish_reason == "stop": # EOS
                        pass
                # print(f"Tool requests: {tool_requests}")
                # print(f"Num of tool requests: {sum(len(reqs) for reqs in tool_requests.values())}")
                if any(tool_requests.values()):
                    logger.info(f"Processing tool requests: {sum(len(reqs) for reqs in tool_requests.values())} total requests")
                    futures = {}
                    for tag, requests in tool_requests.items():
                        if not requests:
                            continue
                        logger.debug(f"Processing {len(requests)} requests for tool '{tag}'")
                        tool = self.tools[tag]
                        for req in requests:
                            logger.debug(f"Submitting tool request: tool={tag}, idx={req['index']}, content={req['content']}")
                            if self.search_lang_side == "src":
                                lang = curr_src_langs[req["index"]]
                            elif self.search_lang_side == "tgt":
                                lang = curr_tgt_langs[req["index"]]
                            elif self.search_lang_side == "all":
                                lang = "all"
                            future = self.executor.submit(self._execute_tool_with_retry, tool, req["content"], lang=lang)
                            futures[future] = {"index": req["index"], "tag": tag}
                    
                    total_futures = len(futures)
                    completed_futures = 0
                    logger.debug(f"Submitted {total_futures} tool requests for execution")
                    for future in concurrent.futures.as_completed(futures):
                        completed_futures += 1
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            # 解析工具执行结果
                            success = result["success"]
                            retry_count = result["retry_count"]
                            execution_time = result["execution_time"]
                            result_text = result["result"]
                            
                            # 更新统计信息
                            if success:
                                tool_metrics["tools/successful_calls"] += 1
                                success_per_tool[tag] += 1
                                logger.info(f"Tool({tag}) for sample {idx} completed successfully in {execution_time:.2f}s, result length: {len(result_text)}")
                            else:
                                tool_metrics["tools/failed_calls"] += 1
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(f"Tool({tag}) for sample {idx} failed after {retry_count} retries, execution time: {execution_time:.2f}s")
                            
                            tool_metrics["tools/total_execution_time"] += execution_time
                            tool_metrics["tools/max_execution_time"] = max(tool_metrics["tools/max_execution_time"], execution_time)
                            tool_metrics["tools/total_retries"] += retry_count
                            tool_metrics["tools/max_retries"] = max(tool_metrics["tools/max_retries"], retry_count)
                            
                            # 更新每个工具的时间统计
                            total_time_per_tool[tag] += execution_time
                            
                            if not result_text:
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(f"Tool({tag}) for sample {idx} returned empty output, execution time: {execution_time:.2f}s")
                            else:
                                logger.debug(f"Tool({tag}) result: {result_text}")
                        except Exception as e:
                            logger.error(f"Tool({tag}) execution failed for sample {idx}: {e}")
                            result_text = f"Error: Tool({tag}) execution failed with message: {e}"
                            tool_metrics["tools/failed_calls"] += 1
                        
                        logger.debug(f"Tool completion progress: {completed_futures}/{total_futures} ({completed_futures/total_futures*100:.1f}%)")
                        formatted_result = f" <information>\n{result_text}\n</information>"
                        result_tokens = self.tokenizer.encode(formatted_result, add_special_tokens=False)
                        logger.debug(f"Result for tool({tag}), sample {idx} tokenized to {len(result_tokens)} tokens")
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))

                final_active_indices = []
                for idx in next_active_indices:
                    response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if response_len < max_len:
                        final_active_indices.append(idx)
                
                active_indices = final_active_indices
                
            # 确保所有序列不超过max_len
            for idx in range(len(curr_inputs)):
                response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_len > max_len:
                    offset = len(init_inputs[idx])
                    curr_inputs[idx] = curr_inputs[idx][:offset + max_len]
                    result_masks[idx] = result_masks[idx][:max_len]
            
            
            # Reorganize outputs to match original batch structure and select up to num_samples per sample
            output_sequences = []
            output_result_masks = []
            for i in range(batch_size):
                # Get all indices for this sample
                sample_indices = sample_to_indices.get(i, [])
                for idx in sample_indices:
                    output_sequences.append(curr_inputs[idx][len(prompt_token_ids_list[i]):])
                    output_result_masks.append(result_masks[idx])
            
            padded_response_list = []
            padded_result_mask_list = []
            for output_ids, result_mask in zip(output_sequences, output_result_masks):
                # logger.debug(f"len(output_ids): {len(output_ids)}, len(result_mask): {len(result_mask)}, output_ids: {output_ids}, result_mask: {result_mask}")
                
                assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
                # print(f"Output_ids: {output_ids}")
                response = torch.tensor(output_ids)
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                
                result_mask_tensor = torch.tensor(result_mask)
                result_mask_tensor = pad_sequence_to_length(result_mask_tensor, self.config.response_length, 0)
                
                padded_response_list.append(response)
                padded_result_mask_list.append(result_mask_tensor)
            
            response = torch.stack(padded_response_list, dim=0).to(input_ids.device)
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)
            
            
            # repeat the input_ids according to dynamic_nums
            repeated_tensors = defaultdict(list)
            for key in ["input_ids", "attention_mask", "position_ids"]:
                tensor = prompts.batch[key]
                cur = 0
                for i in range(tensor.shape[0]):
                    repeated_tensors[key].append(tensor[i].unsqueeze(dim=0).repeat_interleave(int(dynamic_nums[cur]), dim=0))
                    cur += 1
            r_tensors = {key: torch.cat(val, dim=0) for key, val in repeated_tensors.items()}
            
            assert r_tensors["input_ids"].shape[0] == sum(dynamic_nums), f"r_tensors[input_ids].shape: {r_tensors['input_ids'].shape}, sum(dynamic_nums): {sum(dynamic_nums)}"
            
            input_ids = r_tensors["input_ids"]
            attention_mask = r_tensors["attention_mask"]
            position_ids = r_tensors["position_ids"]
            
            if non_tensor_batch:
                repeated_non_tensor_batch = defaultdict(list)
                for key, val in non_tensor_batch.items():
                    cur = 0
                    v_list = []
                    for i in range(val.shape[0]):
                        if not isinstance(val[i], np.ndarray):
                            for j in range(int(dynamic_nums[cur])):
                                v_list.append(val[i])
                        else:
                            repeated_non_tensor_batch[key].append(np.repeat(np.expand_dims(val[i], axis=0), int(dynamic_nums[cur]), axis=0))
                        cur += 1
                    if len(v_list) > 0:
                        repeated_non_tensor_batch[key]= np.array(v_list, dtype=object)
                non_tensor_batch=repeated_non_tensor_batch
            
            
            final_batch_size = input_ids.size(0)
            seq = torch.cat([input_ids, response], dim=-1)
            
            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(final_batch_size, -1)

            if position_ids.dim() == 3:  # for RoPE scaling like qwen2vl mrope
                delta_position_id = delta_position_id.view(final_batch_size, 1, -1).expand(final_batch_size, position_ids.size(1), -1)
                response_position_ids = position_ids[..., -1:].expand(-1, position_ids.size(1), -1) + delta_position_id
            else:
                response_position_ids = position_ids[..., -1:] + delta_position_id

            final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            loss_mask = loss_mask * response_attention_mask
            
            # 计算平均执行时间
            if tool_metrics["tools/total_calls"] > 0:
                tool_metrics["tools/avg_execution_time"] = tool_metrics["tools/total_execution_time"] / tool_metrics["tools/total_calls"]
                
            # 计算每个工具的平均执行时间和成功率
            tool_specific_metrics = {}
            for tag in self.tools.keys():
                calls = calls_per_tool[tag]
                if calls > 0:
                    tool_specific_metrics[f"tools/{tag}/calls"] = calls
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = total_time_per_tool[tag] / calls
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = success_per_tool[tag] / calls
                else:
                    tool_specific_metrics[f"tools/{tag}/calls"] = 0
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = 0
            
            batch = TensorDict({
                "prompts": input_ids,
                "responses": response,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "loss_mask": loss_mask,
                "position_ids": final_position_ids,
            }, batch_size=final_batch_size)
            
        # 合并所有metrics
        all_metrics = {**tool_metrics, **tool_specific_metrics}
        # 将metrics添加到meta_info中
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics
        data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
        return data_proto