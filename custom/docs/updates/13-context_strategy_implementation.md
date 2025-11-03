# Context Strategy System Implementation

**Date**: November 3, 2025  
**Status**: ✅ Complete

## Overview

Implemented a modular, factory-pattern-based context strategy system for handling KV cache overflow during streaming inference. This replaces the monolithic if-else approach from ProAssist with a clean, extensible architecture.

## Motivation

The original ProAssist code handles context overflow using a large if-else block in `manage_kv_cache()`. We wanted:

1. **Modularity**: Each strategy in its own file/class
2. **Extensibility**: Easy to add new strategies (e.g., DST-enhanced)
3. **Testability**: Each strategy can be tested independently
4. **Configuration**: Strategies configured via YAML
5. **Factory Pattern**: Consistent with existing PROSPECT design (data sources, generators)

## Architecture

### Directory Structure

```
custom/src/prospect/context_strategies/
├── __init__.py                      # Base class + enum
├── drop_all.py                      # Strategy 1: Drop everything
├── drop_middle.py                   # Strategy 2: Keep first + last
├── summarize_and_drop.py            # Strategy 3: Generate summary
└── context_strategy_factory.py     # Factory for instantiation

custom/config/prospect/context_strategy/
├── none.yaml                        # No strategy (stateless)
├── drop_all.yaml                    # Drop all config
├── drop_middle.yaml                 # Drop middle config
└── summarize_and_drop.yaml          # Summarize config
```

### Base Class

```python
class BaseContextStrategy(ABC):
    @abstractmethod
    def should_reduce_cache(self, current_seq_len: int) -> bool:
        """Check if cache should be reduced"""
        pass
    
    @abstractmethod
    def handle_overflow(
        self,
        past_key_values: Any,
        last_msg: Any,
        **context
    ) -> Tuple[Any, Any]:
        """Handle KV cache overflow"""
        pass
```

### Strategy Enum

```python
class ContextStrategy(Enum):
    DROP_ALL = "drop_all"
    DROP_MIDDLE = "drop_middle"
    SUMMARIZE_AND_DROP = "summarize_and_drop"
```

## Implemented Strategies

### 1. Drop All (`drop_all.py`)

**Behavior**:
- Drop ALL KV cache when overflow occurs
- Keep only last message/token
- Start fresh with minimal context

**Use case**: Simplest approach, acceptable quality loss

**Memory**: Resets to ~0 tokens

### 2. Drop Middle (`drop_middle.py`)

**Behavior**:
- Keep initial context (first few frames + dialogue)
- Keep recent context (last N tokens, default 512)
- Drop everything in the middle

**Use case**: Preserve task setup + recent activity

**Memory**: Bounded by `init_len + last_keep_len`

**Key method**:
```python
def set_initial_cache(self, past_key_values):
    """Store initial KV cache after first user input"""
    self.init_kv_cache = past_key_values
```

### 3. Summarize and Drop (`summarize_and_drop.py`)

**Behavior**:
- Generate text summary of progress using VLM
- Model "sees" all frames + dialogue via KV cache
- Drop ALL KV cache
- Keep only text summary as context

**Use case**: Best quality, semantic information preserved

**Memory**: Resets to ~100-500 tokens (summary only)

**Key parameters**:
- `summary_max_length`: Max tokens for summary (default 512)
- `summary_prompt`: Prompt for summarization
- `initial_sys_prompt`: Prepended to summary
- `task_knowledge`: Appended to summary

## Factory Pattern

```python
from prospect.context_strategies.context_strategy_factory import ContextStrategyFactory

strategy = ContextStrategyFactory.create_strategy(
    strategy_type="summarize_and_drop",
    max_seq_len=4096,
    reserved_seq_len=128,
    summary_max_length=512,
    summary_prompt="Summarize progress..."
)
```

## Configuration

### Main Config (`prospect.yaml`)

```yaml
defaults:
  - data_source: proassist_dst
  - generator: baseline
  - model: smolvlm2
  - context_strategy: none  # ← New default
```

### Strategy Configs

**None** (`none.yaml`):
```yaml
type: none
# No KV cache accumulation (current baseline behavior)
```

**Drop All** (`drop_all.yaml`):
```yaml
type: drop_all
# No additional parameters
```

**Drop Middle** (`drop_middle.yaml`):
```yaml
type: drop_middle
last_keep_len: 512  # Recent tokens to keep
```

**Summarize and Drop** (`summarize_and_drop.yaml`):
```yaml
type: summarize_and_drop
summary_max_length: 512
summary_prompt: "Summarize the task progress so far..."
```

## Integration

### VLM Runner

Updated `vlm_stream_runner.py`:

```python
def __init__(
    self,
    ...
    context_strategy_type: str = "none",
    context_strategy_config: Optional[Dict] = None,
    **kwargs
):
    # Create strategy via factory
    if context_strategy_type != "none":
        self.context_strategy = ContextStrategyFactory.create_strategy(
            strategy_type=context_strategy_type,
            max_seq_len=kwargs.get('max_seq_len', 4096),
            **context_strategy_config
        )
```

### Evaluator

Updated `prospect_evaluator.py` to pass strategy config:

```python
context_strategy_type = self.cfg.context_strategy.get("type", "none")
context_strategy_config = dict(self.cfg.context_strategy)
context_strategy_config.pop("type", None)

runner = VLMStreamRunner(
    ...
    context_strategy_type=context_strategy_type,
    context_strategy_config=context_strategy_config,
)
```

## Usage Examples

### Run with Different Strategies

```bash
# Default: no strategy (stateless)
./custom/runner/run_prospect.sh generator=baseline

# Drop all strategy
./custom/runner/run_prospect.sh generator=baseline context_strategy=drop_all

# Drop middle strategy
./custom/runner/run_prospect.sh generator=baseline context_strategy=drop_middle

# Summarize and drop
./custom/runner/run_prospect.sh generator=baseline context_strategy=summarize_and_drop
```

### Override Strategy Parameters

```bash
# Custom last_keep_len for drop_middle
./custom/runner/run_prospect.sh \
    generator=baseline \
    context_strategy=drop_middle \
    context_strategy.last_keep_len=1024

# Custom summary prompt
./custom/runner/run_prospect.sh \
    generator=baseline \
    context_strategy=summarize_and_drop \
    'context_strategy.summary_prompt="Describe what has been completed."'
```

## Future Extensions

### DST-Enhanced Strategy

Next step: Create `dst_enhanced.py` strategy that:
- Uses DST state tracking
- Generates summaries with DST context
- Includes completed/current/upcoming steps in summary

```python
class DSTEnhancedStrategy(BaseContextStrategy):
    def handle_overflow(self, past_key_values, last_msg, **context):
        dst_state = context['dst_tracker'].get_current_state()
        
        summary = f"""
        Completed: {dst_state.completed_steps}
        Current: {dst_state.current_step}
        Next: {dst_state.upcoming_steps[0]}
        """
        
        return None, summary
```

## Benefits

1. ✅ **Clean separation of concerns**: Each strategy is self-contained
2. ✅ **Easy to test**: Mock strategies for unit tests
3. ✅ **Extensible**: Add new strategies without modifying existing code
4. ✅ **Configurable**: All parameters in YAML
5. ✅ **Consistent**: Follows existing PROSPECT patterns (factory, config-driven)
6. ✅ **Type-safe**: Enum for strategy types, abstract base class

## Comparison: Before vs. After

### Before (ProAssist)

```python
def manage_kv_cache(self, frames, past_key_values, last_msg):
    if strategy == ExceedContextHandling.DROP_ALL:
        return None, last_msg
    elif strategy == ExceedContextHandling.DROP_MIDDLE:
        # 30 lines of logic here
        ...
    elif strategy == ExceedContextHandling.SUMMARIZE_AND_DROP:
        # 20 lines of logic here
        ...
```

### After (PROSPECT)

```python
# In runner __init__:
self.context_strategy = ContextStrategyFactory.create_strategy(...)

# In inference loop:
if self.context_strategy.should_reduce_cache(current_len):
    past_key_values, last_msg = self.context_strategy.handle_overflow(
        past_key_values, last_msg, **context
    )
```

## Testing

To test a strategy:

```python
from prospect.context_strategies.drop_all import DropAllStrategy

strategy = DropAllStrategy(max_seq_len=4096, reserved_seq_len=128)

# Test overflow detection
assert strategy.should_reduce_cache(4000) == True
assert strategy.should_reduce_cache(3000) == False

# Test overflow handling
new_kv, new_msg = strategy.handle_overflow(mock_kv, "last message")
assert new_kv is None  # Drop all clears cache
```

## Notes

- Current baseline still uses `context_strategy=none` (stateless processing)
- Strategies are ready but not yet used in production runs
- Will be activated when we implement KV cache accumulation for enhanced models
- All three ProAssist strategies are implemented and tested

## Files Created

1. `custom/src/prospect/context_strategies/__init__.py`
2. `custom/src/prospect/context_strategies/drop_all.py`
3. `custom/src/prospect/context_strategies/drop_middle.py`
4. `custom/src/prospect/context_strategies/summarize_and_drop.py`
5. `custom/src/prospect/context_strategies/context_strategy_factory.py`
6. `custom/config/prospect/context_strategy/none.yaml`
7. `custom/config/prospect/context_strategy/drop_all.yaml`
8. `custom/config/prospect/context_strategy/drop_middle.yaml`
9. `custom/config/prospect/context_strategy/summarize_and_drop.yaml`

## Files Modified

1. `custom/config/prospect/prospect.yaml` - Added context_strategy default
2. `custom/src/prospect/runners/vlm_stream_runner.py` - Added strategy integration
3. `custom/src/prospect/prospect_evaluator.py` - Pass strategy config to runner
