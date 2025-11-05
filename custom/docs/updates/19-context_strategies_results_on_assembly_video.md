data/proassist_dst_manual_data/assembly101/assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit.tsv

I have the DST data for the above video.

I want to create a small test with this video for the context strategies.

I want to evaluate the performance of each strategy on this specific assembly task.

I want to create a table for the metrics.

Add a new strategy, which will be summarize_with_dst, where you the model will maintain a DST state and use it to guide the summarization.

The DST will be provided from the TSV files.

The model will have to update the state as it processes frames.

For the initial implementation, we will use the timestamps from the TSV files and update the DST state manually. 

Later, this has to be done automatically during frame processing.

The goal is to see, whether adding the DST state improves the summarization quality.

Right now we will not do any training, we will just run inference on this single video and compare results for the different strategies.