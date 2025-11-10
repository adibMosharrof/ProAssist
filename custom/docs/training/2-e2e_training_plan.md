
# End to End Training Plan


## Requirements

I want to extend the proassist work by incorporating DST to improve context understanding and summarization quality.

The task of the model is to speak at the right time and also say the correct thing.

The DST will be given as an annotation, but the model has to update the state on its own.

The training will involve multi-task learning with both dialogue generation and state tracking objectives.

The DST will replace the summarization task that proassist would do.

My model will have to output two categories: DST and the assistant response.

However, not all frames will require updates, so for each category will have a decision component.

So overall, the model has to output DST update decision, updated DST if the decision was true, speaking decision, and dialog response if the model decides to speak.

The metrics to be tracked will be the F1 score of the speaking decision, accuracy of DST and reponse quality.

During training, the loss has to incorporate the DST update decision loss, DST accuracy loss, speaking decision loss, and response generation loss. 

The speaking and DST update decisions have a high class imbalance, so appropriate weighting or sampling techniques will be necessary to ensure effective learning.

We can start with focal loss, then explore other options.

The response has to be related to the task and context, ensuring relevance and coherence.


### Inputs

The inputs to the model will be the video frames, previous dialog history, and the current dialog state.
The video frames and previous dialog history will be embedded in the KV cache.
The DST and instructions will be added as text prompts.


### Input and Outputs

#### DST


The json file contains the DST with the timestamps. 

We can derive the ground truth state using the timestamps present.

Each item in the DST will be a node and will have a state which will be one of In Progress, Completed, Not Started.

The initial state of all nodes will be not started.

The model has to update the state of certain nodes when it decides to.

The design is similar to the when to speak and response generation style updates.

#### DST Update Decision

It is a binary decision.

#### DST Update

The model outputs the updated states for the nodes it decides to update.
The output of this step should be a valid DST.

#### Speaking Decision

It is a binary decision.

#### Response Generation

The response should be grounded on the video frames, conversation history, dialog state.
It will be free-form text.

It will be evaluated with metrics like BLEU, METEOR, semantic similarity. 
See the metrics in proassist, we already have implemented them.

## Data

DST data: custom/outputs/dst_generated/proassist_label/2025-11-06/17-02-11_gpt-4o_proassist_50rows/assembly101/val.json

The above json contains the full data. The dst data is inside a "dst" key.

We will have separate datasets for training, validation, and testing to ensure robust evaluation.

There are no missing data. The DST annotations were deterministically calculated using the video timestamps.

Read custom/docs/dst_data/proassist_dst_label_plan.md for details on how labels were generated.


## Learning

We are trying to jointly optimize each of the tasks.
We could try curriculum learning where the model first tries to learn the easy items, then over time learns the hard things. I am not sure what the best approach would be for this, so initially lets not incorporate it.

## Metrics

I want to use the same metrics as proassist, so that I can compare performance easily.
I already have things setup in the propect codebase, see the single strategy tests.

custom/src/prospect/tests/run_single_strategy.sh

## Model Architecture

I want to use SmolVLM2 to do the tasks.

custom/src/prospect/tests/run_single_strategy.sh

The above code does direct inference on a single video with different strategies.

I want to train the model and then load the weights for evaluation.

## Training Details

I have two 24GB gpus, Nvidia Titan RTX.

I would like to utilize a batch size that will use up 90% of my GPU memory.

I have not decided whether I want to do full fine-tuning or use LORA.

Initially I will work with SmolVLM2, which will fit my GPU memory.

We can try larger models later.

We also have the options of doing LORA fine-tuning, quantization for efficient memory management if needed.
