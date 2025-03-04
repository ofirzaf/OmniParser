from groq import Groq
import os
from .utils import is_image_path

# import torch
# import intel_extension_for_pytorch as ipex


import os
# import time
# import transformers

LOCAL_LM = False

if LOCAL_LM:
    import openvino as ov
    import openvino_genai as ov_genai
    from transformers import AutoTokenizer
    from collections import namedtuple
    # import numpy as np


    DecodedResults = namedtuple('DecodedResults', ['perf_metrics', 'scores', 'texts', 'tokens'])


    class LLMPipelineWithHFTokenizer(ov_genai.LLMPipeline):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            model_dir = kwargs['model_dir'] if 'model_dir' in kwargs else args[0]
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        def generate(self, *args, **kwargs):
            texts = kwargs.pop('inputs', None)
            if texts is None:
                texts, args = args[0], args[1:]
            # print(texts, flush=True)
            if kwargs.pop('apply_chat_template', False):
                # print("Applying chat template\n")
                # print(self.tokenizer.apply_chat_template(texts, add_generation_prompt=True, tokenize=False), flush=True)
                inputs = self.tokenizer.apply_chat_template(texts, add_generation_prompt=True, return_tensors='np')
                #print(type(inputs), type(inputs[0][0].item()), flush=True)
                inputs = ov.Tensor(inputs)
            else:
                # print(self.tokenizer(texts, return_tensors='np'), flush=True)
                inputs = ov.Tensor(self.tokenizer(texts, return_tensors='np')['input_ids'])
            out = super().generate(inputs, *args, **kwargs)
            # print('\n\n\n', 'Generated tokens number:', len(out.tokens[0]), flush=True)
            res = DecodedResults(out.perf_metrics, out.scores, self.tokenizer.batch_decode(out.tokens), out.tokens)
            # print(res.texts, flush=True)
            return res
else:
    import requests


def run_groq_interleaved(messages: list, system: str, model_name: str, api_key: str, max_tokens=2048, temperature=0.6):
    """
    Run a chat completion through Groq's API, ignoring any images in the messages.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    
    client = Groq(api_key=api_key)
    # avoid using system messages for R1
    final_messages = [{"role": "user" if "r1" in model_name else "system", "content": system}]

    # print(f"system: {system}")
    print(f"messages: {messages}")
    
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict):
                # For dict items, concatenate all text content, ignoring images
                text_contents = []
                images = []
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if not is_image_path(cnt):  # Skip image paths
                            text_contents.append(cnt)
                        elif "VL" in model_name or True: # Vision Language model
                            images.append(cnt)
                        # print("skipped image path")
                    else:
                        text_contents.append(str(cnt))
                
                if text_contents:  # Only add if there's text content
                    message = {"role": "user", "content": " ".join(text_contents)}
                    final_messages.append(message)
            else:  # str
                message = {"role": "user", "content": item}
                final_messages.append(message)
    
    elif isinstance(messages, str):
        final_messages.append({"role": "user", "content": messages})

    try:
        # completion = client.chat.completions.create(
        #     model="deepseek-r1-distill-llama-70b",
        #     messages=final_messages,
        #     temperature=0.6,
        #     max_completion_tokens=max_tokens,
        #     top_p=0.95,
        #     stream=False,
        #     reasoning_format="raw"
        # )
        
        # response = completion.choices[0].message.content

        if LOCAL_LM:
            scheduler_config = ov_genai.SchedulerConfig()
            # # cache params
            # # scheduler_config.cache_size = 2
            scheduler_config.num_kv_blocks = 4096 // 16
            scheduler_config.dynamic_split_fuse = False
            scheduler_config.max_num_batched_tokens = 4096
            print('initializing pipe')
            pipe = LLMPipelineWithHFTokenizer('C:/Users/sdp/shira/deepseek_model', 'GPU',scheduler_config=scheduler_config)
            # pipe = LLMPipelineWithHFTokenizer('C:/Users/sdp/phi3', 'GPU',scheduler_config=scheduler_config)
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
            print(f'final messages: {final_messages}')
            print("\nbefore generate")
            response = pipe.generate(final_messages, config, apply_chat_template=True).texts[0]
        else:
            parameters = {
                "temperature": 0.6,
                "max_new_tokens": max_tokens,
                "top_p": 0.95,
                "apply_chat_template": True,
                "do_sample": False,
            }
            # print(f"\ninput sent to model: {final_messages}\n")
            json_dist={"inputs": final_messages, "parameters": parameters}
            # if images and len(images) > 0:
            #     json_dist.update({"images": images})
            response = requests.post("http://127.0.0.1:8002/api/generate", json=json_dist).json()['generated_text']
            
        # print(f'\ngot response: {response}\n', flush=True)
        final_answer = response.split('</think>\n')[-1] if '</think>' in response else response
        final_answer = final_answer.replace("<output>", "").replace("</output>", "")
        # print(f'\nfinal answer: {final_answer}')
        token_usage = 0
        
        return final_answer, token_usage
    except Exception as e:
        print(f"Error in interleaved Groq: {e}")

        return str(e), 0