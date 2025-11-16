
# Multiprocessing example


Below is an example of using multiprocessing in Python.
I like this structure as it does not have any nested methods.
The core functionality is written in sequential pattern and adds very little complexity to the existing code.
To introduce multiprocessing, we just replace the sequential loop with a parallel map.
I used pathos here, but feel free to use any other library that will allow use to achieve this.


```python

    import itertools
import json
import logging
from pathlib import Path
import re
from dotmap import DotMap
import hydra
from omegaconf import DictConfig
import openai
import pandas as pd
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas
import os
import sys
from sgd_dstc8_data_model.dstc_dataclasses import DstcSchema, DstcServiceCall


sys.path.insert(0, os.path.abspath("./src"))
from tod.turns.api_call_turn_csv_row import ApiCallTurnCsvRow

from configs.dm_config import DataModuleConfig
from datamodules.tod_datamodulev2 import TodDataModuleV2
from schema.schema_loader import SchemaLoader

from metric_managers.metric_manager_factory import MetricManagerFactory

from dstc.dstc_domains import DstcDomainBuilder, DstcDomains
from logger.inference_logger_dataclasses import (
    ApiCallInferenceLogData,
    KetodInferenceLogData,
)
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from prompts.nlg_prompt_manager import ChatGptPrompt, NlgPromptFactory
from logger.results_logger import ResultsLogger
from tqdm import tqdm
from my_enums import Steps, TurnRowType, ZsTodConstants
import utils
import data_prep.data_prep_utils as data_prep_utils
from pathos.multiprocessing import ProcessingPool as Pool
from base_datamodule import SimpleTodDataSet
from tod.turns.zs_tod_turn import TodTurnCsvRowFactory


class AutoTod:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.cfg.project_root = Path(self.cfg.project_root)
        self.cfg.raw_data_root = self.cfg.project_root / self.cfg.raw_data_root
        self.tod_turn_row_cls = ApiCallTurnCsvRow
        self.prompt_cls = NlgPromptFactory.get_handler(
            self.cfg.prompt_type, self.cfg.model_type.context_type
        )
        formatter = logging.Formatter(fmt="%(message)s")
        root_logger = logging.getLogger()  # no name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
        self.logger = root_logger
        # self.metric_manager = NlgApiCallMetricManager(self.logger)
        self.metric_manager = MetricManagerFactory.get_metric_manager(
            self.cfg.model_type.context_type, None, self.logger, self.cfg
        )

    def get_prompts(self, schemas):
        schema_loader = SchemaLoader(DstcSchema)
        schemas = schema_loader.get_schemas(self.cfg.raw_data_root)
        tod_turn_row_cls = TodTurnCsvRowFactory.get_handler(self.cfg)
        dm_cfg = DotMap(self.cfg)
        dm_cfg.update(self.cfg.model_type)
        dm_config = DataModuleConfig(tokenizer=None, **dm_cfg)
        dm = TodDataModuleV2(
            dm_config,
            schemas=schemas,
            tod_turn_row_cls=tod_turn_row_cls,
        )
        dm.setup()
        test_datasets = dm.datasets["test"]
        data = test_datasets[0].data
        all_prompts = []
        for item in data:
            prompt = self.prompt_cls.get_prompt(
                item.domains,
                item.schema,
                item.context,
                all_schema=schemas,
                domains_original=item.domains_original,
            )
            all_prompts.append(DotMap(prompt=prompt, item=item))
        return all_prompts

    def query_chatgpt(self, row):
        openai.api_key = os.getenv("CHATGPT_APIKEY")
        messages = [
            {"role": "system", "content": row.prompt.system_prompt},
            {"role": "user", "content": row.prompt.dialog_history},
        ]
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4o",
            temperature=0,
            messages=messages,
            functions=row.prompt.function_prompt,
        )
        msg = response.choices[0].message
        if msg.content:
            pred = msg.content
        else:
            func_call = msg.function_call
            params_dict = json.loads(func_call.arguments)
            if self.cfg.dataset_name == "bitod":
                params_str = self.get_bitod_params_str(params_dict)
            else:
                params_str = ", ".join(
                    [f"'{k}': '{v}'" for k, v in params_dict.items()]
                )
            pred = f"ApiCall(method='{func_call.name}',parameters={params_str})"
        out = KetodInferenceLogData(
            input_text=row.prompt,
            label=row.item.target,
            pred=pred,
            turn_row_type=row.item.turn_row_type,
            is_retrieval=row.item.is_retrieval,
            is_slot_fill=row.item.is_slot_fill,
            dialog_id=row.item.dialog_id,
            turn_id=row.item.turn_id,
            is_multi_domain_api_call=row.item.is_multi_domain_api_call,
            domains=row.item.domains_original,
            complete_kb_call=row.get("complete_kb_call", None),
            ke_method=row.get("ke_method", None),
            ke_params=row.get("ke_params", None),
            ke_api_call_invoke=row.get("ke_api_call_invoke", None),
            is_single_domain=row.get("is_single_domain", None),
            current_user_utterance=row.get("current_user_utterance", None),
            search_results=row.get("search_results", None),
        )
        return out


    def get_chatgpt_responses(self, item_prompts, nlg_metric_manager):
        outputs = self.get_chatgpt_outputs(item_prompts)
        nlg_metric_manager.data = outputs
        nlg_metric_manager.compute_row_wise_metrics()
        df = pd.DataFrame(nlg_metric_manager.data)
        csv_root = os.getcwd() / Path(self.cfg.out_dir)
        csv_root.mkdir(parents=True, exist_ok=True)
        csv_path = csv_root / "chatgpt_inference.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df

    def get_chatgpt_outputs(self, item_prompts):
        if self.cfg.response_path:
            df = pd.read_csv(self.cfg.project_root / self.cfg.response_path)
            return [KetodInferenceLogData(**i) for _, i in df.iterrows()]
        outputs = []
        if self.cfg.is_multi_process:
            outputs = list(
                tqdm(
                    Pool().imap(
                        self.query_chatgpt,
                        item_prompts,
                    ),
                    total=len(item_prompts),
                )
            )
        else:
            outputs = [self.query_chatgpt(row) for row in item_prompts]
        return outputs

    

    def run(self):
        steps = Steps.list()
        schemas = {}
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in steps]:
            schemas.update(d)
        item_prompts = self.get_prompts(schemas)

        metric_manager = MetricManagerFactory.get_metric_manager(
            self.cfg.model_type.context_type, None, self.logger, self.cfg
        )
        responses = self.get_chatgpt_responses(item_prompts, metric_manager)

```