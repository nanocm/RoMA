# 📚 Contents

- [News](#news)
- [Abstract](#abstract)
- [Overview](#overview)
- [Evaluation Results](#evaluation-results)
- [Scaling Behavior](#scaling-behavior)
- [Pretraining](#pretraining)
- [Checkpoints](#checkpoints)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

# 🔥News


# 📄Abstract

Recent advances in self-supervised learning for Vision Transformers (ViTs) have fueled breakthroughs in remote sensing (RS) foundation models. However, the quadratic complexity of self-attention poses a significant barrier to scalability, particularly for large models and high-resolution images. While the linear-complexity Mamba architecture offers a promising alternative, existing RS applications of Mamba remain limited to supervised tasks on small, domain-specific datasets. To address these challenges, we propose RoMA, a framework that enables scalable self-supervised pretraining of Mamba-based RS foundation models using large-scale, diverse, unlabeled data. 

# 🔍Overview

<figure>
<img src="assets/image-20250311170540530.png">
<figcaption align = "center"><b>Figure 1: Overview of the RoMA Pretraining Pipeline. 
 </b></figcaption>
</figure>

The input image is first divided into patches, and high-value patches are selected for random rotation using the Adaptive Rotation Encoding Strategy. These patches are then tokenized and processed by the Mamba encoder. The encoded features undergo autoregressive next-token prediction, followed by a multi-scale strategy that computes losses at different scales for gradient updates. RoMA optimally adapts the Mamba architecture for remote sensing, making its encoder a robust feature extractor for diverse downstream tasks.

# ✅Evaluation Results

<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <caption style="caption-side: top; text-align: center; font-weight: bold; margin-bottom: 10px;">
    Results for scene classification, change detection, and semantic segmentation.<br>
    “TR” represents the ratio of training data.<br>
   <sup>★</sup> indicates results from MA3E and MTP.<br>
      <sup>†</sup> denotes our reproduction with their official code.
  </caption>
  <colgroup>
    <col style="width: 25%;"> <!-- Set first column width -->
    <col> <!-- Auto-width for the other columns -->
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
  </colgroup>
<thead>
  <tr>
    <th rowspan="3">Methods</th>
    <th rowspan="3">Publication</th>
    <th rowspan="3">Backbone</th>
    <th rowspan="3">Params</th>
    <th colspan="2">Scene Classification</th>
    <th colspan="1">Change Detection</th>
    <th colspan="1">Semantic Segmentation</th>
  </tr>
  <tr>
    <th>AID</th>
    <th>UCM</th>
    <th>OSCD</th>
    <th>SpaceNetv1</th>
  </tr>
  <tr>
    <th>OA(TR=50%)</th>
    <th>OA(TR=50%)</th>
    <th>F1</th>
    <th>mF1</th>
  </tr>
</thead>
  <tbody>
    <tr>
      <td colspan="8" style="text-align: left;"><em style="color: gray;">Natural Image pretraining</em></td>
    </tr>
    <tr>
      <td>MoCo v3<sup>★</sup></td>
      <td>ICCV'21</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>78.72</td>
      <td>38.34</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DINO<sup>★</sup></td>
      <td>ICCV'21</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>78.51</td>
      <td>40.04</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>MAE<sup>★</sup></td>
      <td>CVPR'22</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>84.21</td>
      <td>52.75</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>SimMIM<sup>★</sup></td>
      <td>CVPR'22</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>83.19</td>
      <td>51.48</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LoMaR<sup>★</sup></td>
      <td>Arxiv'22</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>82.26</td>
      <td>51.89</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>MixMAE<sup>★</sup></td>
      <td>CVPR'23</td>
      <td>Swin-B/W14</td>
      <td>88M</td>
      <td>81.53</td>
      <td>50.63</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ARM<sup>†</sup></td>
      <td>ICLR'25</td>
      <td>Mamba-B</td>
      <td>85M</td>
      <td>81.14</td>
      <td>50.41</td>
      <td>47.28</td>
      <td>77.89</td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left;"><em style="color: gray;">RS Image pretraining</em></td>
    </tr>
    <tr>
      <td>SeCo<sup>★</sup></td>
      <td>ICCV'21</td>
      <td>ResNet-50</td>
      <td>25.6M</td>
      <td>78.26</td>
      <td>47.45</td>
      <td>47.67</td>
      <td>77.09</td>
    </tr>
    <tr>
      <td>CACo<sup>★</sup></td>
      <td>CVPR'23</td>
      <td>ResNet-50</td>
      <td>25.6M</td>
      <td>77.81</td>
      <td>40.53</td>
      <td>52.11</td>
      <td>77.94</td>
    </tr>
    <tr>
      <td>SatMAE<sup>★</sup></td>
      <td>NIPS'22</td>
      <td>ViT-L</td>
      <td>307M</td>
      <td>55.10</td>
      <td>34.28</td>
      <td>52.76</td>
      <td>78.07</td>
    </tr>
    <tr>
      <td>ScaleMAE<sup>★</sup></td>
      <td>ICCV'23</td>
      <td>ViT-L</td>
      <td>307M</td>
      <td>48.46</td>
      <td>28.19</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GFM<sup>★</sup></td>
      <td>ICCV'23</td>
      <td>Swin-B</td>
      <td>88M–</td>
      <td>80.58</td>
      <td>49.73</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>RVSA<sup>★</sup></td>
      <td>TGRS'23</td>
      <td>ViT-B+RVSA</td>
      <td>86M</td>
      <td>84.06</td>
      <td>50.86</td>
      <td>50.28</td>
      <td><strong>79.56</strong></td>
    </tr>
    <tr>
      <td>SatMAE++<sup>†</sup></td>
      <td>CVPR'24</td>
      <td>ViT-L</td>
      <td>307M</td>
      <td>85.98</td>
      <td>55.72</td>
      <td>53.10</td>
      <td>79.21</td>
    </tr>
    <tr>
      <td>MA3E<sup>★</sup></td>
      <td>ECCV'24</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>85.86</td>
      <td>55.69</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>RoMA</td>
      <td>-
</td>
      <td>Mamba-B</td>
      <td>85M</td>
      <td><strong>87.36</strong></td>
      <td><strong>59.45</strong></td>
      <td><strong>55.63</strong></td>
      <td>79.50</td>
    </tr>
  </tbody>
</table>

For implementation of each task, please check the corresponding folder for more details.

* [Scene Classification](https://github.com/nanocm/RoMA//tree/main/Scene%20Classification) 
* [Change Detection](https://github.com/nanocm/RoMA/tree/main/Change%20Detection)
* [Semantic Segmentation](https://github.com/nanocm/RoMA/tree/main/Semantic%20Segmentation)

# 📈Scaling Behavior

Mamba’s performance also improves with increasing model size. We conduct extensive pretraining on four model variants—Tiny, Small, Base, and Large—following the configurations in our code. As shown in Figure 3, larger models consistently achieve superior results on downstream tasks. Although Mamba-Large surpasses Mamba-Base in AID dataset, its performance gain remains limited, likely due to insufficient pretraining. With only 300 epochs on 4 million samples, the training may not be adequate for a 297M-parameter model. Due to experimental constraints, we did not extend pretraining to 800 epochs as in MAE. The OSCD and SpaceNet experiments are ongoing, with updates to follow. However, these results do not alter our key findings: Mamba-based RSFMs pretrained with RoMA demonstrate performance gains as model parameters scale. 

# 🚀Pretraining

For environment setup and pretraining instructions, please refer to [RoMA/requirements.txt](https://github.com/nanocm/RoMA/blob/main/RoMA/requirements.txt)  and [RoMA/train.sh](https://github.com/nanocm/RoMA/blob/main/RoMA/train.sh).

# 🎯Checkpoints

We provide our pretrained weights in -.

# 🔗Citation


# 🤝Acknowledgement

-
