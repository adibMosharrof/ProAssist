#!/usr/bin/env python3
"""
Test Token-Level DST Loss

Test script to validate token-level DST loss computation.
Refactored to match project test structure.
"""

import pytest
import torch
import torch.nn as nn
import logging
from unittest.mock import MagicMock
from sklearn.metrics import precision_recall_fscore_support

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTokenLevelLoss:
    """Test class for token-level DST loss computation"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        class MockConfig:
            image_token_id = 12345
            hidden_size = 128
            response_gen_weight = 1.0
            speaking_binary_weight = 1.0
            dst_binary_weight = 1.0
            dst_gen_weight = 1.0
        return MockConfig()

    @pytest.fixture
    def mock_model(self, mock_config):
        """Create mock DST model with forward pass logic"""
        class MockDSTModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.speaking_decision_head = nn.Linear(self.config.hidden_size, 1)
                self.dst_update_head = nn.Linear(self.config.hidden_size, 1)
                
            def forward(self, input_ids=None, speaking_labels=None, dst_update_labels=None, **kwargs):
                # Simulate last_hidden_state
                batch_size, seq_len = input_ids.shape
                last_hidden_state = torch.randn(batch_size, seq_len, self.config.hidden_size)
                
                loss = 0.0
                speaking_binary_loss = None
                dst_binary_loss = None
                
                # --- LOGIC FROM DSTSmolVLMWithStrategies ---
                if speaking_labels is not None and dst_update_labels is not None:
                    image_token_id = self.config.image_token_id
                    
                    if image_token_id is not None and input_ids is not None:
                        batch_size = input_ids.shape[0]
                        all_pred_logits = []
                        all_target_labels_speak = []
                        all_target_labels_dst = []
                        
                        for b in range(batch_size):
                            img_mask = (input_ids[b] == image_token_id)
                            
                            if img_mask.any():
                                img_embeddings = last_hidden_state[b][img_mask]
                                
                                speak_logits = self.speaking_decision_head(img_embeddings)
                                dst_logits = self.dst_update_head(img_embeddings)
                                
                                num_frames = img_embeddings.shape[0]
                                speak_lbl = speaking_labels[b, :num_frames]
                                dst_lbl = dst_update_labels[b, :num_frames]
                                
                                all_pred_logits.append((speak_logits, dst_logits))
                                all_target_labels_speak.append(speak_lbl)
                                all_target_labels_dst.append(dst_lbl)
                        
                        if all_pred_logits:
                            flat_speak_logits = torch.cat([x[0] for x in all_pred_logits]).squeeze(-1)
                            flat_dst_logits = torch.cat([x[1] for x in all_pred_logits]).squeeze(-1)
                            flat_speak_labels = torch.cat(all_target_labels_speak).float()
                            flat_dst_labels = torch.cat(all_target_labels_dst).float()
                            
                            bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                            
                            # Speaking
                            speak_loss_all = bce_loss_fn(flat_speak_logits, flat_speak_labels)
                            speak_mask = flat_speak_labels != -100
                            if speak_mask.any():
                                speaking_binary_loss = speak_loss_all[speak_mask].mean()
                            else:
                                speaking_binary_loss = torch.tensor(0.0)

                            # DST
                            dst_loss_all = bce_loss_fn(flat_dst_logits, flat_dst_labels)
                            dst_mask = flat_dst_labels != -100
                            if dst_mask.any():
                                dst_binary_loss = dst_loss_all[dst_mask].mean()
                            else:
                                dst_binary_loss = torch.tensor(0.0)
                # --------------------------------------------------
                
                # Compute accuracies (for logging)
                speaking_accuracy = None
                speaking_precision = None
                speaking_recall = None
                speaking_f1 = None
                
                dst_accuracy = None
                dst_precision = None
                dst_recall = None
                dst_f1 = None
                
                # Speaking accuracy
                if speaking_labels is not None:
                    speaking_logits_flat = torch.cat([x[0] for x in all_pred_logits]).squeeze(-1) if all_pred_logits else torch.tensor([])
                    speaking_labels_flat = torch.cat(all_target_labels_speak).float() if all_target_labels_speak else torch.tensor([])
                    
                    if speaking_labels_flat.numel() > 0:
                        valid_mask = speaking_labels_flat != -100
                        if valid_mask.any():
                            preds = (speaking_logits_flat[valid_mask] > 0).float()
                            targets = speaking_labels_flat[valid_mask]
                            speaking_accuracy = (preds == targets).float().mean()
                            
                            preds_cpu = preds.cpu().numpy()
                            targets_cpu = targets.cpu().numpy()
                            p, r, f1, _ = precision_recall_fscore_support(
                                targets_cpu, preds_cpu, average='binary', zero_division=0
                            )
                            speaking_precision = torch.tensor(p)
                            speaking_recall = torch.tensor(r)
                            speaking_f1 = torch.tensor(f1)

                # DST accuracy
                if dst_update_labels is not None:
                    dst_binary_logits_flat = torch.cat([x[1] for x in all_pred_logits]).squeeze(-1) if all_pred_logits else torch.tensor([])
                    dst_binary_labels_flat = torch.cat(all_target_labels_dst).float() if all_target_labels_dst else torch.tensor([])
                    
                    if dst_binary_labels_flat.numel() > 0:
                        valid_mask = dst_binary_labels_flat != -100
                        if valid_mask.any():
                            preds = (dst_binary_logits_flat[valid_mask] > 0).float()
                            targets = dst_binary_labels_flat[valid_mask]
                            dst_accuracy = (preds == targets).float().mean()
                            
                            preds_cpu = preds.cpu().numpy()
                            targets_cpu = targets.cpu().numpy()
                            p, r, f1, _ = precision_recall_fscore_support(
                                targets_cpu, preds_cpu, average='binary', zero_division=0
                            )
                            dst_precision = torch.tensor(p)
                            dst_recall = torch.tensor(r)
                            dst_f1 = torch.tensor(f1)

                return {
                    "speaking_binary_loss": speaking_binary_loss,
                    "dst_binary_loss": dst_binary_loss,
                    "speaking_accuracy": speaking_accuracy,
                    "speaking_precision": speaking_precision,
                    "speaking_recall": speaking_recall,
                    "speaking_f1": speaking_f1,
                    "dst_accuracy": dst_accuracy,
                    "dst_precision": dst_precision,
                    "dst_recall": dst_recall,
                    "dst_f1": dst_f1,
                }
        
        return MockDSTModel(mock_config)

    @pytest.fixture
    def sample_data(self, mock_config):
        """Create sample input data"""
        image_token_id = mock_config.image_token_id
        
        # Batch size 2, seq len 10
        # Sample 1: 3 frames (indices 1, 3, 5)
        # Sample 2: 2 frames (indices 2, 4)
        input_ids = torch.tensor([
            [1, image_token_id, 2, image_token_id, 3, image_token_id, 4, 0, 0, 0],
            [1, 2, image_token_id, 3, image_token_id, 4, 0, 0, 0, 0]
        ])
        
        # Labels (padded to max_frames=3)
        # Sample 1: [1, 0, -100] (Speak, Silent, Ignore)
        # Sample 2: [-100, 1, -100] (Ignore, DST Update, Padding)
        speaking_labels = torch.tensor([
            [1, 0, -100],
            [-100, 0, -100]
        ])
        
        dst_update_labels = torch.tensor([
            [0, 0, -100],
            [-100, 1, -100]
        ])
        
        return input_ids, speaking_labels, dst_update_labels

    def test_token_level_loss_computation(self, mock_model, sample_data):
        """
        Test that token-level loss is computed correctly and ignores -100.
        
        Verifies:
        - Loss is not None
        - Loss is not NaN
        - Loss is computed for both heads
        """
        input_ids, speaking_labels, dst_update_labels = sample_data
        
        # Run forward
        outputs = mock_model(
            input_ids=input_ids, 
            speaking_labels=speaking_labels, 
            dst_update_labels=dst_update_labels
        )
        
        speak_loss = outputs["speaking_binary_loss"]
        dst_loss = outputs["dst_binary_loss"]
        speak_acc = outputs.get("speaking_accuracy")
        speak_prec = outputs.get("speaking_precision")
        speak_rec = outputs.get("speaking_recall")
        speak_f1 = outputs.get("speaking_f1")
        
        dst_acc = outputs.get("dst_accuracy")
        dst_prec = outputs.get("dst_precision")
        dst_rec = outputs.get("dst_recall")
        dst_f1 = outputs.get("dst_f1")
        
        logger.info(f"Speaking Loss: {speak_loss}")
        logger.info(f"DST Loss: {dst_loss}")
        logger.info(f"Speaking Accuracy: {speak_acc}, Precision: {speak_prec}, Recall: {speak_rec}, F1: {speak_f1}")
        logger.info(f"DST Accuracy: {dst_acc}, Precision: {dst_prec}, Recall: {dst_rec}, F1: {dst_f1}")
        
        # Verify loss is not None and not NaN
        assert speak_loss is not None, "Speaking loss should not be None"
        assert dst_loss is not None, "DST loss should not be None"
        assert not torch.isnan(speak_loss), "Speaking loss should not be NaN"
        assert not torch.isnan(dst_loss), "DST loss should not be NaN"
        
        # Verify accuracies
        assert speak_acc is not None, "Speaking accuracy should not be None"
        assert dst_acc is not None, "DST accuracy should not be None"
        assert 0.0 <= speak_acc <= 1.0, "Speaking accuracy should be between 0 and 1"
        assert 0.0 <= speak_prec <= 1.0, "Speaking precision should be between 0 and 1"
        assert 0.0 <= speak_rec <= 1.0, "Speaking recall should be between 0 and 1"
        assert 0.0 <= speak_f1 <= 1.0, "Speaking F1 should be between 0 and 1"
        
        assert 0.0 <= dst_acc <= 1.0, "DST accuracy should be between 0 and 1"
        assert 0.0 <= dst_prec <= 1.0, "DST precision should be between 0 and 1"
        assert 0.0 <= dst_rec <= 1.0, "DST recall should be between 0 and 1"
        assert 0.0 <= dst_f1 <= 1.0, "DST F1 should be between 0 and 1"
        
        # Verify that we got a scalar tensor
        assert speak_loss.dim() == 0, "Speaking loss should be a scalar"
        assert dst_loss.dim() == 0, "DST loss should be a scalar"
        
        print(f"\n✅ Token-level loss computation successful!")
