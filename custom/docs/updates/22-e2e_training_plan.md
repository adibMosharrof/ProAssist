
# End to End Training Plan


## Requirements

I want to extend the proassist work by incorporating DST to improve context understanding and summarization quality.

The task of the model is to speak at the right time and also say the correct thing.

The DST will be given as an annotation, but the model has to update the state on its own.

The training will involve multi-task learning with both dialogue generation and state tracking objectives.

The DST will replace the summarization task that proassist would do.

So with my approach, the model has to output the DST, speaking decision, and dialog response if the model decides to speak.

The metrics to be tracked will be the F1 score of the speaking decision, accuracy of DST and reponse quality.

During training, the loss has to incorporate the speaking decision loss, DST accuracy loss, and response generation loss. 

Initially lets not add any weights to the different losses.


The speaking decision has a high class imbalance, so appropriate weighting or sampling techniques will be necessary to ensure effective learning.

We can start with focal loss, then explore other options.

The DST will stay the same most of the time, it will mainly update during transitions, so the model should be designed to handle sparse updates efficiently.

The response has to be related to the task and context, ensuring relevance and coherence.


### Inputs

The inputs to the model will be the video frames, previous dialog history, and the current dialog state.
The video frames and previous dialog history will be embedded in the KV cache.
The DST and instructions will be added as text prompts.


### Outputs

#### DST


The tsv file contains the DST with the timestamps. 

Each line in the  tsv file represents a node.

We can derive the ground truth state using the timestamps present.

Each item in the DST will be a node. The DST is a 3-level tree, it will have steps, substeps and actions.

The states will be: In Progress, Completed, Not Started

The model has to label the state of each node.

The initial state of all nodes will be not started.

Instead of updating at each time step, the model should make a decision at every time step whether it should update DST or not. Once it decides to update, it should output the new state. The design is similar to the when to speak and response generation style updates.

Should update DST should be a binary decision.

#### Speaking Decision

It is a binary decision.

#### Response Generation

The response should be grounded on the video frames, conversation history, dialog state.
It will be free-form text.

It will be evaluated with metrics like BLEU, METEOR, semantic similarity. See the metrics in proassist, we already have implemented them.

#### Validations

We have to have a mechanism to handle the hierarchical nature of the DST, ensuring consistency between parent and child nodes.

We also have to make sure we have a valid DST after each update.


## Data

DST data: data/proassist_dst_manual_data/assembly101/assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit.tsv

proassist data: data/proassist/processed_data/assembly101

We will have separate datasets for training, validation, and testing to ensure robust evaluation.

There are no missing data. The DST annotations was generated with LLMs, so there could be some errors in it, but we should not worry about it initially.


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
