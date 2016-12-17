### SegNet-Basic:
---

What is Segnet?

* Deep Convolutional Encoder-Decoder Architecture for Semantic Pixel-wise Image Segmentation

 **Segnet** = **(Encoder + Decoder)** +  **Pixel-Wise Classification** layer

##### *[SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation (Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE) arXiv:1511.00561v3](https://arxiv.org/abs/1511.00561)*


What is SegNet-Basic?

* *"In order to analyse SegNet and compare its performance with FCN  (decoder  variants)  we  use  a  smaller  version  of  SegNet, termed SegNet-Basic ,  which  ha  4  encoders  and  4  decoders. All the encoders in SegNet-Basic perform max-pooling and subsampling and the corresponding decoders upsample its input using the  received  max-pooling  indices."*

Basically it's a mini-segnet to experiment / test the architecure with convnets, such as FCN.


 -----

### Steps To Run The Model:
---

1. Run `python model-basic.py` to create `segNet_basic_model` for keras to use.
	
	* `model-basic.py` contains the architecure.

2. 



### Dataset:
---

1. In a different directory run this to download the [dataset from original Implementation](https://github.com/alexgkendall/SegNet-Tutorial).
	* `git clone git@github.com:alexgkendall/SegNet-Tutorial.git`
	* copy the `/CamVid` to here, or change the `DataPath` in `data_loader.py` to the above directory
2. The run `python data_loader.py` to generate these two files:
	
	* `/data/train_data.npz/` and `/data/train_label.npz`
	* This will make it easy to process the model over and over, rather than waiting the data to be loaded into memory.



----


### To Do:
----

	[ ] SegNet-Basic
	[ ] SegNet
	[ ] Test Accuracy
	[ ] Requirements





