?  *	???̰?A2?
_Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::Map _?L?\s@!wƄ0f?U@)Y?? Zs@1?r[??U@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?|?5^?E@!+E??~y(@)???QImE@1'ߓ?kF(@:Preprocessing2?
KIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleAndRepeat C?i?q???!??|???)L7?A`???1?﹏?4??:Preprocessing2?
lIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::Map::TensorSlice '1?Z??!?C?R???)'1?Z??1?C?R???:Preprocessing2?
ZIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2 :#J{?/??!$??hަ?):#J{?/??1$??hަ?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchV????_??!w??I9Y??)V????_??1w??I9Y??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??W?2ı?!@O7? ??)p_?Q??1??H??}?:Preprocessing2F
Iterator::Model+??????!?8QL???)"??u????1?J?8?c?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q???(??"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.JDESKTOP-PDLAU14: Failed to load libcupti (is it installed and accessible?)