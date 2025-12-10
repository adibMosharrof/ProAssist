import torch
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from mmassist.eval.runners.stream_inference import FrameOutput
from custom.src.prospect.models.dst_proact import DSTProActLlamaForCausalLM
from custom.src.prospect.utils.cache_manager import KVCacheManager

logger = logging.getLogger(__name__)

@dataclass
class DSTFrameOutput(FrameOutput):
    dst_update: Optional[str] = None
    dst_state: Optional[Dict] = None

class DSTStreamRunner:
    """
    Frame-based streaming runner for DST inference.
    
    Features:
    - Sequential frame processing
    - Continuous KV cache management
    - Binary head decisions (Speaking, DST Update)
    - Context overflow handling (Full DST Schema Refresh)
    """
    
    def __init__(
        self,
        model: DSTProActLlamaForCausalLM,
        processor: Any,  # Can be tokenizer or processor
        fps: float = 2.0,
        speaking_threshold: float = 0.5,
        dst_threshold: float = 0.5,
        max_seq_len: int = 4096,
        reserved_seq_len: int = 512,
        device: str = "cuda",
        worker_id: int = 0,
    ):
        self.model = model
        self.processor = processor
        self.fps = fps
        self.speaking_threshold = speaking_threshold
        self.dst_threshold = dst_threshold
        self.max_seq_len = max_seq_len
        self.reserved_seq_len = reserved_seq_len
        self.device = device
        self.worker_id = worker_id
        
        # Initialize Cache Manager
        self.cache_manager = KVCacheManager(context_strategy=None)
    
    @property
    def tokenizer(self):
        """Get tokenizer from processor or use processor directly if it's a tokenizer."""
        if hasattr(self.processor, 'tokenizer'):
            return self.processor.tokenizer
        return self.processor
        
    def run_inference_on_video(self, sample: Dict[str, Any]) -> List[FrameOutput]:
        """
        Run streaming inference on a single video sample.
        """
        embeddings = sample["embeddings"].to(self.device)  # [T, 2048]
        dst_schema = sample["dst"]
        conversation = sample.get("conversation", [])
        clip_start_frame = sample.get("start_frame_idx", 0)
        
        # Track trigger statistics
        stats = {
            "total_frames": len(embeddings),
            "speaking_triggered": 0,
            "dst_triggered": 0,
            "both_triggered": 0,
        }
        
        # Reset cache manager
        self.cache_manager.reset()
        
        # Initialize state
        dst_state = {}  # {"S1": "completed", ...}
        if "initial_dst_state" in sample and sample["initial_dst_state"]:
            dst_state = sample["initial_dst_state"].copy()
            
        # Initial system prompt
        system_prompt = self._build_updated_schema_prompt(dst_schema, dst_state)
        
        # Tokenize system prompt using tokenizer
        messages = [{"role": "system", "content": system_prompt}]
        text_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        last_msg_tokens = self.tokenizer(text_input, return_tensors="pt").input_ids.to(self.device)
        
        outputs = []
        
        import sys
        from tqdm import tqdm
        for frame_idx in tqdm(range(len(embeddings)), desc=f"Worker {self.worker_id} Frames", leave=False, file=sys.__stderr__, position=self.worker_id):
        # for frame_idx in tqdm(range(400), desc=f"Worker {self.worker_id} Frames", leave=False, file=sys.__stderr__, position=self.worker_id):
            # 1. Prepare Input
            frame_embed = embeddings[frame_idx].unsqueeze(0).unsqueeze(0) # [1, 1, 2048]
            
            if last_msg_tokens is not None:
                img_token_id = getattr(self.model.config, 'img_token_id', 49190)
                img_token = torch.tensor([[img_token_id]], device=self.device)
                input_ids = torch.cat([img_token, last_msg_tokens], dim=1)
                last_msg_tokens = None
            else:
                img_token_id = getattr(self.model.config, 'img_token_id', 49190)
                input_ids = torch.tensor([[img_token_id]], device=self.device)
            
            # Get current cache
            kv_cache = self.cache_manager.get_cache()
            
            # 2. Create joint embeddings (ProAssist pattern)
            # Use joint_embed to combine vision + text embeddings
            inputs_embeds = self.model.joint_embed(
                input_ids=input_ids,
                image_embeds=frame_embed
            )
            
            # 3. Single forward pass: get tokens AND binary head logits
            # Returns (token_ids, kv_cache, model_outputs)
            output_ids, kv_cache, model_outputs = self.model.fast_greedy_generate(
                inputs_embeds=inputs_embeds,
                past_key_values=kv_cache,
                max_length=1,  # Just 1 token for frame processing
                output_hidden_states=True,  # Request binary head logits
            )
            
            # 4. Update KV Cache with new state
            self.cache_manager.update_cache(kv_cache)
            cache_len = self.cache_manager.get_cache_length()
            
            # Log cache growth every 50 frames
            if frame_idx % 50 == 0 and frame_idx > 0:
                logger.debug(f"Frame {frame_idx}: KV cache length = {cache_len} tokens")
            
            # Get binary head decisions from model outputs dictionary
            dst_logits = model_outputs.get('dst_update_logits', None)
            if dst_logits is not None:
                dst_prob = torch.sigmoid(dst_logits[:, -1])
                dst_update_triggered = dst_prob > self.dst_threshold
            else:
                dst_update_triggered = False
            
            dst_text = None
            if dst_update_triggered:
                stats["dst_triggered"] += 1
                # DST updates are short (e.g., "S1->start"), so generate up to 10 tokens
                # Convert output token IDs to embeddings for generation
                gen_input_embeds = self.model.get_input_embeddings()(output_ids)
                dst_gen_ids, kv_cache, _ = self.model.fast_greedy_generate(
                    inputs_embeds=gen_input_embeds,
                    past_key_values=kv_cache,
                    max_length=10,
                )
                # Decode generated tokens
                dst_text = self.tokenizer.decode(dst_gen_ids[0], skip_special_tokens=True)
                # Update cache with DST generation results
                self.cache_manager.update_cache(kv_cache)
                dst_state = self._update_state(dst_state, dst_text)

            # 6. Check Speaking Decision
            speaking_logits = model_outputs.get('speaking_logits', None)
            if speaking_logits is not None:
                speaking_prob = torch.sigmoid(speaking_logits[:, -1])
                speaking_triggered = speaking_prob > self.speaking_threshold
            else:
                speaking_triggered = False
            
            gen_text = ""  # Default to empty string (no-talk sentinel)
            if speaking_triggered:
                stats["speaking_triggered"] += 1
                # Generate assistant response (longer, use max_length=50)
                # Convert output token IDs to embeddings for generation
                gen_input_embeds = self.model.get_input_embeddings()(output_ids)
                gen_ids, kv_cache, _ = self.model.fast_greedy_generate(
                    inputs_embeds=gen_input_embeds,
                    past_key_values=kv_cache,
                    max_length=50,
                )
                # Decode generated tokens
                gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                # Update cache with response generation results
                self.cache_manager.update_cache(kv_cache)
            
            # 6. Get Ground Truth References
            # Use frame_idx directly since conversation events are relative to clip start
            # Default to empty string (no-talk sentinel) if no reference text found
            ref_speaking = self._get_reference_text(conversation, frame_idx, "assistant")
            if ref_speaking is None:
                ref_speaking = ""
            ref_dst_update = self._get_reference_text(conversation, frame_idx, "DST_UPDATE")

            # 7. Store Output
            outputs.append(DSTFrameOutput(
                gen=gen_text,
                ref=ref_speaking,
                frame_idx_in_stream=frame_idx,
                timestamp_in_stream=frame_idx / self.fps,
                dst_update=dst_text,
                dst_state=dst_state.copy(),
            ))
            
            # 8. Context Management
            if self._should_refresh_context(kv_cache):
                current_cache_len = self.cache_manager.get_cache_length()
                logger.warning(f"🔄 Context Overflow! Refreshing at frame {frame_idx}")
                logger.warning(f"   Cache length: {current_cache_len} / {self.max_seq_len} (threshold: {self.max_seq_len - self.reserved_seq_len})")
                self.cache_manager.reset() # Reset cache manager
                kv_cache = None
                
                system_prompt = self._build_updated_schema_prompt(dst_schema, dst_state)
                messages = [{"role": "system", "content": system_prompt}]
                text_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                last_msg_tokens = self.tokenizer(text_input, return_tensors="pt").input_ids.to(self.device)
                logger.info(f"   Rebuilt system prompt with DST state: {len(last_msg_tokens[0])} tokens")
                
        # Log sparsity statistics
        if stats["total_frames"] > 0:
            speaking_rate = (stats["speaking_triggered"] / stats["total_frames"]) * 100
            dst_rate = (stats["dst_triggered"] / stats["total_frames"]) * 100
            logger.info(f"📊 Inference Statistics (Video with {stats['total_frames']} frames):")
            logger.info(f"   Speaking triggered: {stats['speaking_triggered']} frames ({speaking_rate:.1f}%)")
            logger.info(f"   DST update triggered: {stats['dst_triggered']} frames ({dst_rate:.1f}%)")
            logger.info(f"   ✅ Sparse generation is {'ACTIVE' if (speaking_rate + dst_rate) < 20 else 'LIMITED'}")
                
        return outputs

    def _get_reference_text(self, conversation: List[Dict], frame_idx: int, role: str) -> Optional[str]:
        """
        Get reference text for a specific role at a specific frame.
        """
        for turn in conversation:
            if turn["role"] == role:
                if "start_frame" in turn and turn["start_frame"] == frame_idx:
                    # For DST_UPDATE, content is a list of dicts, we need to format it
                    if role == "DST_UPDATE":
                        content = turn["content"]
                        if isinstance(content, list) and len(content) > 0:
                            update = content[0]
                            return f"{update['id']}->{update['transition']}"
                    else:
                        return turn["content"]
        return None

    def _update_state(self, current_state: Dict, dst_text: str) -> Dict:
        if not dst_text or "->" not in dst_text:
            return current_state
        try:
            parts = dst_text.split("->")
            step_id = parts[0].strip()
            transition = parts[1].strip()
            
            if transition == "start":
                current_state[step_id] = "in_progress"
            elif transition == "complete":
                current_state[step_id] = "completed"
        except:
            pass
        return current_state

    def _should_refresh_context(self, kv_cache) -> bool:
        if kv_cache is None:
            return False
        current_len = kv_cache[0][0].shape[2]
        return current_len >= (self.max_seq_len - self.reserved_seq_len)

    def _build_updated_schema_prompt(self, dst_schema: List[Dict], current_state: Dict) -> str:
        lines = ["You are a helpful assistant.\n"]
        lines.append("Steps:")
        for step in dst_schema:
            step_id = step["id"]
            step_name = step["name"]
            state = current_state.get(step_id, "not_started")
            lines.append(f"- {step_id}: {step_name} ({state})")
            
        # Sort steps alphanumerically for consistent state string
        sorted_state = sorted(current_state.items(), key=lambda x: (int(x[0][1:]) if x[0][1:].isdigit() else x[0]))
        state_parts = [f"Step {k}: {v}" for k, v in sorted_state]
        state_str = ", ".join(state_parts)
        
        lines.append(f"\nDialogue State:\nCurrent step states - {state_str}")
        return "\n".join(lines)
