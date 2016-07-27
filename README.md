# deepmark
THE Deep Learning Benchmarks

See: https://github.com/soumith/convnet-benchmarks/issues/101

~~Come back here on June 15th, 2016.~~  
A bit delayed due to y'know -- a lot of co-ordination among groups.

## Networks
### Images
- InceptionV3-batchnorm (http://arxiv.org/abs/1512.00567 , https://gist.github.com/soumith/e18a56b847a8905a84fffb1f32b06db8, https://github.com/Moodstocks/inception-v3.torch)
- Alexnet-OWT
- VGG-D
- ResNet-50 ( http://arxiv.org/abs/1512.03385 , https://github.com/facebook/fb.resnet.torch )

### Video
- C3D - A vgg-style 3D net ( http://vlg.cs.dartmouth.edu/c3d/ )

### Audio
- DeepSpeech2 - Convnet + RNN + FC ( http://arxiv.org/abs/1512.02595 )
- MSR's 5 layer FC net ( https://github.com/Alexey-Kamenev/Benchmarks )

### Text
- Small RNN LSTM ( https://github.com/karpathy/char-rnn/blob/master/train.lua#L38-L48 )
- Large RNN LSTM ( BIG-LSTM in http://arxiv.org/abs/1602.02410 )


### Platform
- Initially multi-GPU with (1 to 4 titan-X cards)
- However, multi-machine, custom hardware, other GPU cards such as AMD, CPUs etc. can and should be accommodated, we will work this out after the initial push.

## Metrics
- Round-trip time for 1 epoch of training (will define an epoch size separately for each network)
- Maximum batch-size that fits (to show and focus on the extra memory consumption that the framework uses)

## Frameworks
Everyone who wants to join-in, but I thought an initial set that is important to cover would be:
- Caffe
- Chainer
- CNTK
- MXNet
- Neon
- Theano
- TensorFlow
- Torch

## Scripts format
- Emit JSON output (so that the README -- or jekyll website can be auto-generated, similar to http://autumnai.com/deep-learning-benchmarks )

## Guarantees
- I will personally to the best of my abilities make sure that the benchmarking is fair and unbiased. The hope is that the community at large will watch these and point-out / fix mistakes.

## Governance
- The benchmarks will be placed at https://github.com/DeepMark/deepmark and other key community members / organizations who want ownership will be welcome to join in proposing new benchmarks that get relevant as the field progresses.
