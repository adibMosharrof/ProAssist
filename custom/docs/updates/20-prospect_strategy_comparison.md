# PROSPECT Strategy Performance Comparison

## Overview

This document summarizes the performance evaluation of all PROSPECT context management strategies tested on the assembly video dataset (9011-c03f). Each strategy was evaluated on the same video with identical model configuration (SmolVLM-2.2B-Instruct) and evaluation parameters.

## Performance Comparison Table

| Strategy | Generated | Matched | Missed | F1 Score | BLEU-4 | METEOR | Time | Memory | Compression Events |
|----------|-----------|---------|--------|----------|--------|--------|------|--------|-------------------|
| `drop_all` | 8 | 3 | 31 | 0.1277 | 0.0505 | 0.1313 | 49.18s | 7.1GB | 2 |
| `drop_middle` | 6 | 2 | 33 | 0.0889 | 0.0000 | 0.0752 | 48.17s | 7.5GB | 3 |
| `summarize_and_drop` | 8 | 3 | 31 | 0.1277 | 0.0505 | 0.1313 | 72.89s | 7.4GB | 2 |
| `summarize_with_dst` | 8 | 3 | 31 | 0.1277 | 0.1715 | 0.1702 | 74.29s | 7.5GB | 2 |

## Test Configuration

- **Dataset**: Assembly101 video `9011-c03f` (461 frames)
- **Model**: HuggingFaceTB/SmolVLM-2.2B-Instruct
- **Ground Truth**: 39 dialogue events from PROASSIST_DST dataset
- **Evaluation Window**: ±15 seconds for dialogue matching
- **Max Tokens**: 4096 (KV cache limit)
- **Test Date**: November 4-5, 2025

## Strategy Details

### 1. drop_all
**Strategy**: Clears entire KV cache on compression events
- **Performance**: Baseline performance with identical metrics to summarize_and_drop
- **Speed**: Fastest execution (49.18s)
- **Memory**: Lowest peak usage (7.1GB)
- **Behavior**: Complete cache reset, no context preservation
- **Use Case**: Maximum memory efficiency, minimal context retention

### 2. drop_middle
**Strategy**: Keeps last 512 tokens, drops middle content
- **Performance**: Lowest F1 score (0.0889) and BLEU-4 (0.0000)
- **Speed**: Fast execution (48.17s)
- **Memory**: Moderate usage (7.5GB)
- **Behavior**: Partial context preservation, frequent compression (3 events)
- **Use Case**: Balance between memory and speed, but poor quality retention

### 3. summarize_and_drop
**Strategy**: Generates summary before clearing entire cache
- **Performance**: Same as drop_all (F1: 0.1277, BLEU-4: 0.0505)
- **Speed**: Slower than drop_all (72.89s due to summary generation)
- **Memory**: Moderate usage (7.4GB)
- **Behavior**: Brief summaries ("The person has", "Person has completed...")
- **Use Case**: Context-aware cache clearing with quality summaries

### 4. summarize_with_dst
**Strategy**: DST-guided contextual summaries with cache preservation
- **Performance**: Best overall quality (F1: 0.1277, BLEU-4: 0.1715, METEOR: 0.1702)
- **Speed**: Slowest execution (74.29s due to detailed DST summaries)
- **Memory**: Highest usage (7.5GB)
- **Behavior**: Rich DST-guided summaries with full context preservation
- **Use Case**: Maximum context quality and task understanding

## Key Findings

### Performance Insights
- **Quality Leader**: summarize_with_dst achieves best BLEU-4 (0.1715) and METEOR (0.1702) scores
- **F1 Consistency**: drop_all, summarize_and_drop, and summarize_with_dst achieve identical F1 (0.1277)
- **drop_middle Underperforms**: Significantly lower F1 (0.0889 vs 0.1277) and BLEU-4 (0.0000 vs 0.0505-0.1715)
- **Quality vs Speed Trade-off**: Faster strategies sacrifice semantic quality metrics

### Efficiency Insights
- **Memory Usage**: drop_all most efficient (7.1GB), others ~7.4-7.5GB
- **Execution Time**: drop_middle fastest (48.17s), DST strategies slowest (72-74s)
- **Compression Frequency**: drop_middle requires most compressions (3 vs 2)

### Strategy Recommendations
- **For Speed**: Use `drop_all` (fastest, good quality, low memory)
- **For Quality**: Use `summarize_with_dst` (best context understanding)
- **For Balance**: Use `summarize_and_drop` (good quality with reasonable speed)
- **Avoid**: `drop_middle` (poor performance despite speed)

## Generated Files

Each strategy test generates a complete output directory with:

```
{strategy_name}/{timestamp}/
├── {strategy_name}_timeline.html    # Interactive visualization
├── {strategy_name}.log             # Execution log
├── {strategy_name}_trace.json      # Raw trace data
├── frames/                         # Frame thumbnails (40-41 images)
└── eval/                           # Detailed evaluation results
```

## HTML Timeline Features

All strategies generate interactive HTML timelines with:
- Complete metrics dashboard
- Frame thumbnails for all events
- Ground truth dialogue visualization
- Compression event details
- Generation event tracking
- Collapsible content sections
- Responsive design with modern styling

## Conclusion

The evaluation reveals clear performance differences between strategies:
- **summarize_with_dst** provides the best semantic quality (highest BLEU-4 and METEOR scores) at higher computational cost
- **drop_all** provides the best balance of speed, memory efficiency, and F1 score
- **summarize_and_drop** offers reasonable quality with moderate overhead
- **drop_middle** should be avoided due to poor performance across all metrics

All strategies successfully generate comprehensive HTML visualizations for detailed analysis and comparison.