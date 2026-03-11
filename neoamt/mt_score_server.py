
from typing import Optional

import uvicorn
from comet import download_model, load_from_checkpoint
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from neoamt.llm_judge import PROMPT_TEMP_POINT_WISE, PROMPT_TEMP_LIST_WISE, transformers_gen, gen, extract_evaluation

#####################################
# FastAPI server below
#####################################


class QueryRequest(BaseModel):
    src: str
    sys_hypo: str
    ref: Optional[str] = None


class BatchQueryRequest(BaseModel):
    src: list[str]
    sys_hypo: list[str]
    ref: Optional[list[str]] = None


class BatchLLMJudgeRequest(BaseModel):
    src: list[str]
    sys_hypo: list[str]
    ref: list[str]
    neologism: list[str]
    gloss: list[str]
    model_name: list[str]


app = FastAPI()

# cometkiwi_da_xl_model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
# cometkiwi_da_xl_model = load_from_checkpoint(cometkiwi_da_xl_model_path)
# xcomet_xl_model_path = download_model("Unbabel/XCOMET-XL")
# xcomet_xl_model = load_from_checkpoint(xcomet_xl_model_path)

# cometkiwi_da_xl_model_path = download_model("/lustre/miao/models/Unbabel/wmt23-cometkiwi-da-xl")
cometkiwi_da_xl_model = load_from_checkpoint("/lustre/miao/models/Unbabel/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt", local_files_only=True)
# xcomet_xl_model_path = download_model("/lustre/miao/models/Unbabel/XCOMET-XL")
xcomet_xl_model = load_from_checkpoint("/lustre/miao/models/Unbabel/XCOMET-XL/checkpoints/model.ckpt", local_files_only=True)
# xcomet_xxl_model = load_from_checkpoint("/lustre/miao/models/Unbabel/XCOMET-XXL/checkpoints/model.ckpt", local_files_only=True)
# cometkiwi_da_xxl_model = load_from_checkpoint("/lustre/miao/models/Unbabel/wmt23-cometkiwi-da-xxl/checkpoints/model.ckpt", local_files_only=True)

def get_cometkiwi_da_xl_model_dep():
    return cometkiwi_da_xl_model

# def get_cometkiwi_da_xxl_model_dep():
#     return cometkiwi_da_xxl_model

def get_xcomet_xl_model_dep():
    return xcomet_xl_model

# def get_xcomet_xxl_model_dep():
#     return xcomet_xxl_model

# llm_judge_gemma3_27b_it_model_path = "/lustre/miao/models/google/gemma-3-27b-it"
# llm_judge_gemma3_27b_it_tokenizer = AutoTokenizer.from_pretrained(llm_judge_gemma3_27b_it_model_path, use_fast=False)
# llm_judge_gemma3_27b_it_model = AutoModelForCausalLM.from_pretrained(
#     llm_judge_gemma3_27b_it_model_path,
#     device_map="auto",
#     torch_dtype="auto",
# )


@app.post("/cometkiwi_da_xl")
def cometkiwi_da_xl_score_endpoint(request: QueryRequest, model=Depends(get_cometkiwi_da_xl_model_dep)):
    # Perform batch retrieval
    if not request.ref:
        data = [
            {
                "src": request.src,
                "mt": request.sys_hypo,
            }
        ]
    else:
        data = [
            {
                "src": request.src,
                "mt": request.sys_hypo,
                "ref": request.ref,
            }
        ]
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    return {"result": model_output.scores[0]}


@app.post("/xcomet_xl")
def xcomet_xl_score_endpoint(request: QueryRequest, model=Depends(get_xcomet_xl_model_dep)):
    # Perform batch retrieval
    if not request.ref:
        data = [
            {
                "src": request.src,
                "mt": request.sys_hypo,
            }
        ]
    else:
        data = [
            {
                "src": request.src,
                "mt": request.sys_hypo,
                "ref": request.ref,
            }
        ]
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    return {"result": model_output.scores[0]}


@app.post("/batch_cometkiwi_da_xl")
def batch_cometkiwi_da_xl_score_endpoint(request: BatchQueryRequest, model=Depends(get_cometkiwi_da_xl_model_dep)):
    data = []
    for src, mt in zip(request.src, request.sys_hypo):
        data.append({
            "src": src,
            "mt": mt,
        })
    model_output = model.predict(data, batch_size=8, gpus=1)
    scores = model_output.scores
    # check mt is empty string
    for i in range(len(request.sys_hypo)):
        if request.sys_hypo[i].strip() == "":
            scores[i] = 0.0
    return {"result": scores}

# @app.post("/batch_cometkiwi_da_xxl")
# def batch_cometkiwi_da_xxl_score_endpoint(request: BatchQueryRequest, model=Depends(get_cometkiwi_da_xxl_model_dep)):
#     data = []
#     for src, mt in zip(request.src, request.sys_hypo):
#         data.append({
#             "src": src,
#             "mt": mt,
#         })
#     model_output = model.predict(data, batch_size=8, gpus=1)
#     scores = model_output.scores
#     # check mt is empty string
#     for i in range(len(request.sys_hypo)):
#         if request.sys_hypo[i].strip() == "":
#             scores[i] = 0.0
#     return {"result": scores}

@app.post("/batch_xcomet_xl")
def batch_xcomet_xl_score_endpoint(request: BatchQueryRequest, model=Depends(get_xcomet_xl_model_dep)):
    data = []
    for src, mt, ref in zip(request.src, request.sys_hypo, request.ref):
        data.append({
            "src": src,
            "mt": mt,
            "ref": ref,
        })
    model_output = model.predict(data, batch_size=8, gpus=1)
    scores = model_output.scores
    # check mt is empty string
    for i in range(len(request.sys_hypo)):
        if request.sys_hypo[i].strip() == "":
            scores[i] = 0.0
    return {"result": scores}

# @app.post("/batch_xcomet_xxl")
# def batch_xcomet_xxl_score_endpoint(request: BatchQueryRequest, model=Depends(get_xcomet_xxl_model_dep)):
#     data = []
#     for src, mt, ref in zip(request.src, request.sys_hypo, request.ref):
#         data.append({
#             "src": src,
#             "mt": mt,
#             "ref": ref,
#         })
#     model_output = model.predict(data, batch_size=8, gpus=1)
#     scores = model_output.scores
#     # check mt is empty string
#     for i in range(len(request.sys_hypo)):
#         if request.sys_hypo[i].strip() == "":
#             scores[i] = 0.0
#     return {"result": scores}


# @app.post("/batch_llm_judge")
# def batch_llm_judge_score_endpoint(request: BatchLLMJudgeRequest):
#     scores = []
#     max_try = 3
#     for src, mt, ref, neologism, gloss, model_name in zip(request.src, request.sys_hypo, request.ref, request.neologism, request.gloss, request.model_name):
#         # data.append({
#         #     "src": src,
#         #     "mt": mt,
#         #     "ref": ref,
#         #     "neologism": neologism,
#         #     "gloss": gloss,
#         #     "model_name": model_name,
#         # })
#         if model_name == "gemma3-27b-it":
#             prompt = PROMPT_TEMP_POINT_WISE.format(
#                 source_sentence=src,
#                 neologism=neologism,
#                 neologism_meaning=gloss,
#                 reference_translation=ref,
#                 candidate_translation=mt,
#             )
            
#             for _ in range(max_try):
#                 response = gen(
#                     prompt=prompt,
#                     model_path=llm_judge_gemma3_27b_it_model_path,
#                     tokenizer=llm_judge_gemma3_27b_it_tokenizer,
#                     model=llm_judge_gemma3_27b_it_model,
#                     max_new_tokens=5096,
#                     temperature=0.2,
#                     top_p=0.95,
#                 )
                
#                 score_str = extract_evaluation(response)
#                 try:
#                     score = float(score_str)
#                 except Exception as e:
#                     print(f"Error parsing score: {score_str}, error: {e}")
#                     score = None
#                 if score is not None:
#                     break
#             scores.append(score)
#     return {"result": scores}


if __name__ == "__main__":
    # Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)