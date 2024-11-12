# Rebuttal

## Response to Reviewer 1 (还有9字符可用)
>7 - 3
>This paper present a new strategy for employing motion-oriented diffusion mechanism, utilizing optical flow as an additional constraint in video generation tasks to enhance consistency between entities. The approach includes improved noise initialization and a new entity-level attention mechanism. The experiments demonstrate the potential of this method.
>Strength:
>- The motivation is clear, and the paper is well-organized to effectively convey key concerns.
>- The two main modules are plug-and-play, making them adaptable to any backbone architecture.
>- The comparison section is informative and comprehensive.
>Weakness:

>- The baseline is better than few methods in some metric, but it’s not detailed.
>- The results are dynamic, however, the visual comparison in the supplementary are still too short to demonstrate the superiority.
>- More metrics should be used to evaluate video quality comprehensively.
>- The MOEA appears to degrade performance in terms of image quality, as indicated by a higher FID score. A more detailed explanation of this phenomenon would be beneficial.
>- A user study could provide valuable insights into human preferences, complementing the numerical evaluations.

Overall, I believe this paper meets the bar for acceptance.



We sincerely thank the reviewer for their insightful comments and are encouraged to hear that they appeciate our clear motivation, plug-and-play module and comprehensive comparison. 

>*W3: More metrics*

We introduce landmark distance, FaceID, and cross-CLIP from Make Your Anchor, ArcFace and Pix2Video. Our method show significant improvements in face and hand consistency.

||LDM(face)↓|LDM(hand)↓|LDM(body)↓|FaceID↑|cross-CLIP↑| 
|-|-|-|-|-|-|
|DisCo|12.07|5.63|7.81|0.356|0.8843|
|MagicAniamte|11.06|5.22|7.18|0.372|0.8913|
|Animate Anyone|10.07|5.17|7.38|0.368|0.8952|
|Champ|10.31|5.06|7.21|0.392|0.8933|
|Ours|8.16|4.03|6.81|0.436|0.9072|

>*W5: User study*

We conducted a user study with 40 video sets and 20 volunteers to rate each video from 1 to 5. Our method is comparable in image quality, and better in video quality.

||Human|Background|Motion Smoothness|Face Consistency|Hand Consistency| Artifacts & Flicking| 
|-|-|-|-|-|-|-|
|DisCo|2.770|2.940|1.225|1.205|1.395|1.150|
|MagicAniamte|3.050|4.080|2.825|2.560|2.085|1.260|
|Animate Anyone|3.235|3.725|2.840|2.850|2.180|2.440|
|Champ|3.725|4.070|3.370|3.285|3.295|2.905|
|Ours|4.050|4.340|3.870|4.150|4.005|3.515|

>*W1: Baseline Metric*

Tab. 1: baseline are various SOTAs, our method exhibits fluctuations around the baseline rather than consistently leading, attributable to minor instability and insignificance in the metrics. 

Tab. 4: baseline is the backbone plus our MOLR module, due to the trade-off between image and video metrics—a phenomenon noted in studies like AnimateDiff—the addition of the MOEA module results in a slight decline in metrics.

>*W4: MOEA*

MOEA enhances consistency among entities across frames rather than improving the fidelity within frames. The slight decrease in FID is due to the trade-offs mentioned earlier. However, this reduction is not relative to backbone because MOLR has improved the FID. 

>*W2: Longer Videos*

We generate long videos by default, and all experiments and user study are conducted on long videos. We cannot provide videos due to rebuttal limits, but we find performance difference between short (4s) and long video (>16s) is minimal. We will add more videos in revision.

||FID|LDM(face)↓|LDM(hand)↓|FaceID↑|cross-CLIP↑| 
|-|-|-|-|-|-|
|short|31.06|8.02|3.98|0.442|0.9085|
|long|31.42|8.16|4.03|0.436|0.9072|

Your review help us refine details and improve evaluations. We hope our response addresses your concerns and will bolster your support for this work.


## Response to Reviewer 2 (还有7字符可用)
> 6 - 4

> This paper proposes an image-based animation method. They employ normal maps, masks, segmentation maps from the SMPL model, and skeletons derived from DWPose as driving signals. Experiment results show they outperform other SOTA methods.

> Strengths
>1. The paper investigates the importance of initial noise in video generation tasks. It proposes a Motion-Oriented Latent Refinement strategy to optimize the initial noise, which regulates the layout while preserving random high-frequency subbands to accommodate deformations.

> 2. They employ a SwinTransformer-like mechanism to perform self-attention in different subspaces, which effectively speeds up inference and shows improvements in metrics.

> 3. The visualization results outperform previous SOTA methods. 

> Weakness
>- It’s unclear what other benefits MOEA offers besides saving time. Why MOEA can result in better consistency? Could you provide a more detailed visualization analysis to clarify its physical significance?

> - The metrics are not significantly better than previous methods, especially for Champ

> - The paper is hard to follow, with many details left unexplained. For example, in line 263, it’s unclear what the final x’ really is. Additionally, since AnimateAnyone is not open-source, how did this paper perform a comparison?

> - The result videos contain several artifacts and temporal flickering. 

Overall, even though this paper lacks many technical details and is difficult to read, it still provides some contributions and analysis, and the visual results look ok. Therefore, I am inclined to accept it and look forward to the author's response.

We are grateful for reviewer’s insightful comments and are encouraged by their recognition of our module and visualization.

>*W1: MOEA*

MOEA is designed to enhance entity consistency, e.g. invariant face details. In Fig. 1, we mark the preceding and following frames' hand tokens as $H_1,H_2$(in blue), and background tokens as $B_1,B_2$(in orange). We aim to analyze the activation from $H_1$ to $H_2$. Here, $X\in R^{f,h,w}$  with entity window size $\{ f',h',w'\}$. We ignore the residual connections, simplify the softmax to $1/n$, and remove the unrelated elements. 

① Entity Attn: $H_1→H_2$
$$
\begin{aligned}X_{{H_{2}}}&=Attn(Q_{{H_{2}}},K_{{H_{1}}})⋅V_{{H_{1}}}\\\\&=\frac{Q_{{H_{2}}}⋅K_{{H_{1}}}^{-1}}{f'h'w'}⋅V_{{H_{1}}}\end{aligned}
$$
② Spatial & Temporal Attn: $H_1→(B_1,B_2)→H_2$
$$
\begin{aligned}
X_{H_{2}}& =Attn(Q_{H_2},K_{B_1})⋅Attn(Q_{B_1},K_{H_1})⋅V_{H_1}+Attn(Q_{H_2},K_{B_2})⋅Attn(Q_{B_2},K_{H_1})⋅V_{H_1} \\\\
&=\frac{Q_{H_{2}}⋅K_{B_{1}}^{-1}}f⋅\frac{Q_{B_{1}}⋅K_{H_{1}}^{-1}}{hw}⋅V_{H_{1}}+\frac{Q_{H_{2}}⋅K_{B_{2}}^{-1}}{hw}⋅\frac{Q_{B_{2}}⋅K_{H_{1}}^{-1}}f⋅V_{H_{1}} \\\\
&<\frac{Q_{H_2}⋅K_{H_1}^{-1}}f⋅\frac1{hw}⋅V_{H_1}+\frac{Q_{H_2}⋅K_{H_1}^{-1}}{hw}⋅\frac1f⋅V_{H_1}\quad since\ Q_HK_B^{-1}<Q_HK_H^{-1}<1,Q_BK_H^{-1}<Q_HK_H^{-1} \\\\
&=2⋅\frac{Q_{H_2}⋅K_{H_1}^{-1}}{fhw}⋅V_{H_1} \\\\
&<\frac{Q_{H_2}⋅K_{H_1}^{-1}}{f'h'w'}⋅V_{H_1}\quad since\ fhw\gg f'h'w'
\end{aligned}
$$

In addition, the value of ① is 4.8e-2 and ② is 1.3e-3 on TikTok. Therefore, MOEA is much better at interaction. 

>*W2: Comparisons*

Since FVD and similar metrics only measure overall video consistency, not entity consistency, we conducted additional evalution and user study. Due to space limits, we respectfully request the reviewer to refer to our response of Reviewer UxcL W3, W5 for more detail.

>*W3: Details*

We will check all details and open source our code.

x´(revision from L259 to L263): We combine the LL from $x^1$ with the LH, HL, HH from $x^k$ to form $x^k´= \{x^1_{ll}, x^k_{lh}, x^k_{hl}, x^k_{hh}\}$. Then, we get $x´= \{x^1´, \ldots, x^k´, \ldots\}$ and apply IDWT to it.

Animate Anyone: We reproduce it's enhanced version by MooreThreads’ git repo. 

>*W4: artifacts and flickering*

In user study, despite it is challenging, we have made improvements. 

Your review helped us clarify details and improve evaluations. We’re thrilled you’re considering our work and hope our response alleviates your concerns and reinforces your support, which is crucial to us.

## Response to Reviewer 3 (还有24字符可用)
> 5 - 4

> Summary This paper proposes a method for enhancing consistency between entities in Human animation by introducing a motion-oriented diffusion mechanism. To explore the limitations of the parameter-based motion module in modeling visual consistency, they conduct a statistical analysis of cosine similarity among the tokens. To address challenges such as misaligned character shapes, movements, rotations and deformations among entities, they propose a motion-oriented latent refinement strategy and an entity attention mechanism.

> Pos.1. This paper conducts a thorough theoretical and experimental analysis of the deficiencies in existing motion modules and proposes a motion-oriented mechanism to enhance the consistency of entities in videos

> Neg.1. The proposed method introduces a new pretrained model along with additional computational complexity, but the quantitative experimental results are not convincing enough to demonstrate its effectiveness

> Neg.2. The qualitative experimental results do not reveal any fundamental difference from the Champ model

> Q1. Was the model used for evaluation on the TikTok dataset trained on the TikTok training set alone, or was it trained on both the TikTok and Champ datasets together?

> Q2. For the ablation study in Table 3, I think the baseline for the ablation study is Champ, if I'm not mistaken. Judging by the metrics, this ablation study was evaluated on TikTok, so why is there such a large gap between the metrics of your baseline and Champ?

We are grateful for the reviewer’s insightful comments and encouraged by their recognition of our thorough analysis and motion-oriented mechanism.

>*Q1: Dataset*

The Champ full dataset (Champ-5328) consists of 5,328 videos, while the released subset (Champ-832) consists of 832 video. The TikTok dataset have 335 videos. Our model is trained on the combination of Champ-832 and TikTok-335.

>*Q2: Metric Gap*

As you correctly observed, the baseline in Tab. 3 is Champ. To ensure a fair comparison in our ablation, all configs were trained using the same combination of Champ-832 and TikTok-335. The metric gap is due to the original Champ using the larger Champ-5328. However, even with only 1/5 of the dataset, our method still outperforms Champ. We have requested more data from the Champ team; their open-sourcing process is ongoing. Meanwhile, we are collecting a comparable dataset and will include futher results in the revision.

>*N1: OmniMotion*

Using OmniMotion increases computation, which is fine for predefined scenarios but not for real-time demands. We are considering zero-shot RAFT as an alternative. Experiments show that RAFT achieves comparable performance, we attribute it to the tolerance of patch-level flow alignment and MOEA window. However, OmniMotion performs better in a few corner cases, we will clarify it in the revision.

||FID↓|FVD↓|LDM(face)↓|LDM(hand)↓|FaceID↑|cross-CLIP↑|
|-|-|-|-|-|-|-|
|RAFT|31.27|155.92|8.52|4.14|0.428|0.9012|
|OmniMotion|31.42|152.74|8.16|4.03|0.436|0.9072|

>*N1,N2: Results*

For quantitative metrics, since FVD and similar measures only overall video consistency and not entity consistency, our results do not show a significant improvement over Champ.

For qualitative results, we cannot provide videos due to rebuttal limit, so we strongly hope the reviewer to compare the faces or hands in the supplementary. We will highlight the main differences in revision.

We conducted additional quantitative experiments and a user study to clearly demonstrate our effectiveness. Due to space constraints, we respectfully ask the reviewer to refer to our response to Reviewer UxcL W3, W5 for more details. These results underscore our method’s effectiveness in maintaining visual consistency, especially in facial and hand features.

Overall, your insightful review has significantly advanced our evaluation and ablation. We hope our response addresses your concerns and will enhance your support for our work, which is crucial to us.

## Response to Reviewer 4 （还剩71字符可用）
> 6 - 5

> This paper proposes a motion-oriented pipeline to enhance the performance of existing SD-based human video animation models. The key idea is to use motion flow to guide the refinement of latent features and the operation of attention mechanisms. I have some questions as follows. 
> (1) In Figure 4, it seems that users can transfer various motion representations, such as skeleton sequences, dense poses, normal maps, etc., to the proposed motion flow through the flow extraction module. However, I couldn't find details on how this transfer process is performed or specifics about the flow extraction module in the paper.

> (2) The proposed model uses both the pose sequence and the motion flow a> s motion guidance simultaneously. What roles do these components play in the model? Is it possible to use motion flow to guide the animation without inputting the pose sequence? Additionally, since the model already uses the pose sequence as guidance, I am somewhat concern about the usefulness of the motion flow.

> (3) I would encourage the authors to provide more qualitative ablation study to verify the key components.

> (4) Is the Motion-oriented Latent Refinement performed at every denoising step in the diffusion model, or only at the first step? If it is only performed at the first step, please provide more evidence to verify the necessity of this module.


We are immensely thankful to the reviewer for their perceptive observations and thoughtful remarks. Below, we address the comments to clarify and improve our manuscript.

>*W1:Motion flow Extraction*

We will clarify it in revision. At L210, we extract flow using OmniMotion. This involves composing videos from depth, normal map, semantic, and skeleton sequences and inputting them into OmniMotion. The outputs are motion flows $\mathcal{F}_d$, $\mathcal{F}_n$, $\mathcal{F}_s$, and $\mathcal{F}_k$, which we then average to obtain the final $\mathcal{F}$. 

>*W2: Role of Motion flow*

Pose representation guides the body shape in each frame, and motion flow guides the visual consistency between frames. These two elements are complementary rather than substitutive. While theoretically possible to convert between them, it is difficult for diffusion models to learn this conversion. Experiments show that combining both elements significantly outperforms using just one.

||FID↓|FVD↓|FaceID↑|cross-CLIP↑|
|-|-|-|-|-|
|Pose Sequence|33.69|181.81|0.392|0.8933|
|First Pose+Motion Flow|34.12|175.39|0.383|0.8958|
|Combination|31.42|152.74|0.436|0.9072|


>*W3: More qualitative ablation*

We are unable to provide more videos due to rebuttal limits, but we strongly hope the reviewer to compare the faces or hands in the supplementary materials. More qualitative ablations and highlighted differences will be provided in the revision. Additionally, we conducted further comparisons to clearly demonstrate our effectiveness. We respectfully ask the reviewer to refer to our responses to Reviewer UxcL W3, W5 for more details.

>*W4: Step of MOLR*

We conduct MOLR at the noise initialization stage due to its significant impact on results, as evidenced in L244-L246. Noise changes affect all denoising steps.Your insightful suggestion has motivated us to ablate the step of MOLR.Experiments show that applying MOLR early can slightly improve video quality, but using it at every step greatly reduces quality. This occurs because entity tokens are more similar in the early steps and diverge significantly later on. We will include these interesting findings in the revision.

||FID↓|SSIM↑|FVD↓|FaceID↑|cross-CLIP↑|
|-|-|-|-|-|-|
|Before Denoising|31.42|0.791|152.74|0.436|0.9072|
|+ early 5 steps|31.51|0.784|151.92|0.439|0.9069|
|+ early 10 steps|34.82|0.714|176.02|0.372|0.8873|
|+ all 20 steps|48.92|0.622|301.92|0.310|0.8392|


|MOLR|MOEA|LDM(face)↓|LDM(hand)↓|FaceID↑|cross-CLIP↑| 
|-|-|-|-|-|-|
|||10.72|5.12|0.380|0.8914|
|✓||9.39|4.42|0.412|0.9024|
||✓|8.78|4.24|0.424|0.9052|
|✓|✓|8.16|4.03|0.436|0.9072|



<!-- 
① Entity Attn: $H_1→H_2$
$$
\begin{aligned}X_{{H_{2}}}&=Attn(Q_{{H_{2}}},K_{{H_{1}}})⋅V_{{H_{1}}}\\\\&=\frac{Q_{{H_{2}}}⋅K_{{H_{1}}}^{-1}}{f^{'}h^{'}w^{'}}⋅V_{{H_{1}}}\end{aligned}
$$
② Spatial & Temporal Attn: $H_1→(B_1,B_2)→H_2$
$$
\begin{aligned}
X_{H_{2}}& =Attn(Q_{H_{2}},K_{B_{1}})⋅V_{B_{1}}+Attn(Q_{H_{2}},K_{B_{2}})⋅V_{B_{2}} \\\\
&=Attn(Q_{H_2},K_{B_1})⋅Attn(Q_{B_1},K_{H_1})⋅V_{H_1}+Attn(Q_{H_2},K_{B_2})⋅Attn(Q_{B_2},K_{H_1})⋅V_{H_1} \\\\
&=\frac{Q_{H_{2}}⋅K_{B_{1}}^{-1}}f⋅\frac{Q_{B_{1}}⋅K_{H_{1}}^{-1}}{hw}⋅V_{H_{1}}+\frac{Q_{H_{2}}⋅K_{B_{2}}^{-1}}{hw}⋅\frac{Q_{B_{2}}⋅K_{H_{1}}^{-1}}f⋅V_{H_{1}} \\\\
&<\frac{Q_{H_2}⋅K_{H_1}^{-1}}f⋅\frac1{hw}⋅V_{H_1}+\frac{Q_{H_2}⋅K_{H_1}^{-1}}{hw}⋅\frac1f⋅V_{H_1}\quad since\ Q_HK_B^{-1}<Q_HK_H^{-1}<1,Q_BK_H^{-1}<Q_HK_H^{-1} \\\\
&=2⋅\frac{Q_{H_2}⋅K_{H_1}^{-1}}{fhw}⋅V_{H_1} \\\\
&<\frac{Q_{H_2}⋅K_{H_1}^{-1}}{f^{'}h^{'}w^{'}}⋅V_{H_1}\quad since\ fhw\gg f^{'}h^{'}w^{'}
\end{aligned}
$$ -->