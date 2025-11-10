# How ProAssist Handles User Types and Metrics

## Data Generation Structure

In ProAssist, **YES, for 1 video they generate 3 different conversation cases** for different user types:

### User Type Configuration
- **Default setting**: `"no_talk@2,talk_some@4,talk_more@4"`
- This means for each video, they generate:
  - 2 conversation variants for **no_talk** users (passive users who follow instructions)
  - 4 conversation variants for **talk_some** users (ask occasional questions ~20% of steps)  
  - 4 conversation variants for **talk_more** users (talkative, ask questions ~40% of steps)

### Data Structure
```json
{
  "video_uid": "...",
  "conversations": [
    {
      "user_type": "no_talk",
      "conversation": [...],
      "auto_quality_eval": {...}
    },
    {
      "user_type": "talk_some", 
      "conversation": [...],
      "auto_quality_eval": {...}
    },
    {
      "user_type": "talk_more",
      "conversation": [...], 
      "auto_quality_eval": {...}
    }
  ]
}
```

## How They Handle User Types in Evaluation

### 1. **Data Filtering** (`filter_and_split.py`)
- Groups conversations by `user_type`
- For each user type, selects the **best quality** conversation based on `auto_quality_eval.final_score`
- Final dataset contains **one conversation per user type** per video

### 2. **Evaluation Process**
- Each conversation variant is treated as a **separate data sample**
- Evaluation is performed **independently** for each user type
- No aggregation of metrics across user types

### 3. **Results Reporting**
Based on `assets/results_all.yaml`, they report:
- **Single set of metrics** per evaluation run
- Metrics include: `AP`, `AR`, `Avg-F1`, `JI`, `bleu4`, `num_m`
- **NOT** separate metrics for each user type
- The `notalk0.5` in results refers to prediction threshold, not user type

## Key Insight
**They treat each user type as a separate data point, not as multiple metrics for the same video.**

For 1 video:
- **Data**: 3 conversation variants (no_talk, talk_some, talk_more)
- **Evaluation**: 3 separate evaluations 
- **Reporting**: 3 separate results, not 3 metrics for 1 result

This approach allows them to:
1. Train models on diverse user interaction patterns
2. Evaluate performance separately for each user type
3. Compare how well the system adapts to different user behaviors