

Parameter-Efficient Fine-Tuning for HAR:
Integrating LoRA and QLoRA into Transformer
## Models
Irina SEREGINA
## Univ. Grenoble Alpes,
## Grenoble, France
Irina.Seregina@etu.univ-grenoble-alpes.fr
Philippe LALANDA
## Univ. Grenoble Alpes,
## Grenoble, France
philippe.lalanda@imag.fr
German VEGA
## Univ. Grenoble Alpes,
## Grenoble, France
german.vega@imag.fr
Abstract—Human   Activity   Recognition   (HAR)   is   a   foun-
dational   task   in   ubiquitous   computing   with   applications   in
health  monitoring,  smart  environments,  and  human–computer
interaction.  While  recent  advances  in  self-supervised  learning
and transformer-based architectures have significantly improved
HAR  performance,  adapting  large  pretrained  models  to  new
domains  remains  a  practical  challenge  due  to  limited  compu-
tational  resources  on  target  devices.  This  papers  investigates
parameter-efficient fine-tuning techniques, specifically Low-Rank
Adaptation (LoRA) and Quantized LoRA, as scalable alternatives
to  full  model  fine-tuning  for  HAR.  We  propose  an  adaptation
framework  built  upon  a  Masked  Autoencoder  backbone  and
evaluate  its  performance  under  a  Leave-One-Dataset-Out  val-
idation  protocol  across  five  open  HAR  datasets.  Our  experi-
ments demonstrate that both LoRA and QLoRA can match the
recognition  performance  of  full  fine-tuning  while  significantly
reducing  the  number  of  trainable  parameters,  memory  usage,
and training time. Further analyses reveal that LoRA maintains
robust performance even under limited supervision and that the
adapter rank provides a controllable trade-off between accuracy
and  efficiency.  QLoRA  extends  these  benefits  by  reducing  the
memory  footprint  of  frozen  weights  through  quantization,  with
minimal  impact  on  classification  quality.
## I.  INTRODUCTION
Deep  learning  has  become  a  cornerstone  of  modern  arti-
ficial  intelligence,  particularly  in  pervasive  computing  appli-
cations  [1].  Unlike  traditional  machine  learning  approaches
that  rely  on  handcrafted  features,  deep  models  automatically
extract  representations  from  raw  sensor  time  series,  making
them  especially  effective  for  high-dimensional  and  heteroge-
neous  data.  A  major  breakthrough  was  the  introduction  of
Transformers—self-attention  architectures  that  capture  long-
range  dependencies  without  recurrence  [2].  Originally  de-
signed  for  natural  language  processing,  they  have  since  been
adapted to sensor-based time-series tasks [3]. Their scalability
and  representational  power  make  them  attractive  for  perva-
sive  computing;  however,  they  typically  require  large  train-
ing  datasets  to  converge  and  involve  millions  of  parameters,
which  limits  their  deployment  on  resource-constrained  de-
vices. Moreover, in pervasive computing, labeled data are often
scarce  due  to  costly  annotation  and  high  context  variability,
further exacerbating the data-hungry nature of Transformers.
To  mitigate  label  scarcity,  practitioners  increasingly  adopt
a  two-stage  pipeline:  pre-training  on  large,  heterogeneous
corpora   followed   by   task-specific   fine-tuning.   Pre-training
yields transferable, general-purpose features, while fine-tuning
adapts  them  to  downstream  tasks  with  limited  supervision.
This approach has proved effective in many pervasive applica-
tions—for example, self-supervised pre-training on large wear-
able  collections  before  Human  Activity  Recognition  (HAR)
adaptation  [4],  or  pre-training  on  multi-sensor  sleep  datasets
prior  to  specialization  on  smaller  clinical  cohorts  [5].  Be-
yond  representation  transfer,  pre-training  effectively  enlarges
the  usable  training  signal  by  leveraging  related—though  not
identical—domains and contexts. This label efficiency is espe-
cially valuable for Transformers, whose strong capacity entails
substantial data requirements.
The purpose of fine-tuning is to adapt a (pretrained) model
to  a  target  domain  by  updating  its  parameters.  Full  fine-
tuning, where all parameters are adjusted, often yields strong
performance  but  comes  at  high  computational  and  memory
costs, which limits its applicability to resource-constrained de-
vices. To address this, parameter-efficient fine-tuning (PEFT)
methods  have  been  proposed.  Techniques  such  as  Low-Rank
Adaptation (LoRA) or adapters update only a small fraction of
additional parameters, while keeping most pretrained weights
frozen. This approach reduces training and storage costs sub-
stantially while preserving accuracy and reducing the amount
of labeled data required for adaptation, making it attractive for
deployment in real-world settings.
In  this  paper,  we  study  human  activity  recognition  and
incorporate Low-Rank Adaptation (LoRA) into a Transformer
backbone  pretrained  via  Masked  Autoencoding  (MAE).  We
show   that   this   integration   is   both   feasible   and   effective,
offering  a  practical  path  to  scalable,  low-cost—and  data-
efficient—personalization  in  HAR  and,  more  broadly,  per-
vasive  computing.  The  choice  of  HAR  as  our  application
domain  is  motivated  by  the  growing  need  for  systems  that
robustly  adapt  to  non-stationary,  real-world  conditions.  As
deployments  scale  across  users,  devices,  and  contexts,  HAR
models  must  deliver  accurate,  personalized  predictions  on
resource-constrained  mobile  and  wearable  platforms—often
arXiv:2512.17983v1  [cs.LG]  19 Dec 2025

with limited labeled data and under strict privacy constraints.
Conventional supervised models, even when pretrained, often
fail  to  generalize  without  costly  full-model  adaptation.  We
further extend the study to QLoRA, which introduces quantiza-
tion to reduce memory footprint and improve inference speed,
enabling practical deployment on edge and IoT devices.
More specifically, we investigate whether integrating Trans-
formers and (Q)LoRA fine-tuning enhances the adaptability of
HAR models under domain shifts and low-resource conditions.
We  benchmark  the  proposed  approach  on  multiple  publicly
available  HAR  datasets  and  evaluate  its  performance  using  a
Leave-One-Dataset-Out validation protocol across five hetero-
geneous HAR benchmarks.
The key contributions of this work are then as follows:
-  We  introduce  the  integration  of  LoRA  into  transformer-
based   HAR   models   pretrained   with   MAE,   enabling
parameter-efficient fine-tuning for domain adaptation.
-  We  extend  this  approach  with  QLoRA,  reducing  the
memory footprint of frozen weights through quantization
and enabling faster, more resource-efficient deployment.
-  We  conduct  extensive  experiments  under  a  Leave-One-
Dataset-Out validation protocol across five heterogeneous
HAR datasets, demonstrating that LoRA maintains robust
performance while providing a tunable trade-off between
accuracy and efficiency via adapter rank. QLoRA further
enhances   deployability   by   lowering   resource   require-
ments with minimal impact on accuracy.
The paper is structured as follows: Section 2 reviews related
work  on  HAR,  and  fine-tuning  strategies  with  a  focus  on
parameter-efficient adaptation, including LoRA and QLORA.
Section  3  describes  the  proposed  methodology,  including  the
model architecture, the integration of LoRA and QLoRA, and
dataset preprocessing. Section 4 describes the implementation
and  section  V  presents  experimental  results  and  comparative
analyses between full fine-tuning, LoRA, and QLoRA. Finally,
Section 5 concludes the paper with a summary of findings and
directions for future work.
## II.  BACKGROUND AND  RELATED WORK
## A.  Human Activity Recognition
Human Activity Recognition (HAR) is the automatic identi-
fication of physical activities from data collected by wearable
or ambient sensors such as accelerometers and gyroscopes cite
[6],  [7].  It  supports  applications  in  healthcare,  personalized
fitness,  rehabilitation,  workplace  safety,  and  smart  homes.
Over the years, HAR has progressed from shallow classifiers
with  handcrafted  features  to  deep  learning  models  such  as
CNNs, RNNs, and transformers, which can learn spatiotempo-
ral  representations  directly  from  raw  multivariate  signals  [8].
Yet   HAR   suffers   from   pronounced   domain   shift:   models
trained on one dataset often degrade on unseen users, devices,
placements, sampling rates, or behaviors [9]. In HAR, domain
shifts commonly arise due to:
-  Covariate shift: the input distribution P (x) changes while
P (y|x) remains the same (e.g., older adults walking more
slowly than younger adults);
-  Label  shift:  the  label  distribution P (y)  changes  while
P (x| y)  remains  the  same  (e.g.,  datasets  dominated  by
low- vs. high-intensity activities);
-  Concept  shift:  the  relationship P (y | x)  changes  (e.g.,
identical  motion  patterns  labeled  differently  across  con-
texts).
Hence, even with strong pretrained representations, fine-tuning
is fundamental to adapt models to the target domain (user, de-
vice, placement) and to recover robustness under heterogeneity
and noise.
B.  Fine Tuning: basics
Full  fine-tuning  is  the  process  of  updating
all  parameters
of  a  pretrained  neural  network  when  adapting  it  to  a  new
task  or  domain  (as  usually  defined  in  SE  [10],  [11]).  It
remains  the  standard  and  most  flexible  method  for  adapting
large models, especially when there is a significant mismatch
between  the  source  and  target  data  distributions.  A  typical
full-adaptation pipeline is: (1) initialize a pretrained backbone
(general-purpose representations); (2) add a task-specific head
(e.g., a softmax classifier for HAR); (3) unfreeze all layers (as
opposed to feature extraction); and (4) train end-to-end on the
target  data  (e.g.,  with  cross-entropy),  so  gradients  update  the
entire network and both low- and high-level features adapt.
Full fine-tuning is particularly effective when the target task
differs  significantly  from  pretraining,  often  yielding  higher
accuracy  in  transfer  scenarios  with  small  datasets  or  strong
domain  shifts  [12],  [13],  because  updating  all  layers  can
realign both low- and high-level features to the new input–label
distribution.  It  maximizes  domain-specific  learning  capacity
but at the expense of higher compute, sensitivity to initializa-
tion,  and  risk  of  forgetting  pretrained  knowledge.  However,
full fine-tuning is rarely suited for edge deployment: updating
all  weights  is  memory-  and  compute-intensive,  and  storing
a  separate  model  for  each  user  or  context  makes  large-scale
personalization impractical.
C.  Fine Tuning: Advanced techniques
In addition to full fine-tuning, a variety of alternative strate-
gies  have  been  developed  to  adapt  pretrained  models  to  new
tasks  or  domains.  These  methods  are  particularly  relevant  in
scenarios where computational resources are limited or where
parameter efficiency is essential. In particular, let us mention
the following approaches:
-  Feature Extraction (Frozen Backbone): In this approach,
the pretrained model is used as a fixed feature extractor.
All  weights  in  the  backbone  are  frozen,  and  only  the
newly added task-specific head is trained on a new data
source.  This  strategy  is  computationally  efficient  and
requires minimal memory, making it well-suited for low-
resource  settings  or  when  the  target  dataset  is  small.
However,  its  adaptability  is  limited  because  the  frozen
backbone cannot adjust to the characteristics of the new
data distribution.
-  Partial  fine-tuning  refers  to  the  selective  unfreezing  and
training  of  only  a  subset  of  model  parameters,  typically

the  top  layers  of  the  encoder  or  normalization  layers.
This technique seeks a balance between performance and
resource usage. It allows the model to adapt its high-level
representations  to  the  new  domain  while  preserving  the
general features learned during pretraining.
-  Adapter  layers  introduce  small,  trainable  modules  into
the  frozen  pretrained  model.  These  modules  are  typi-
cally  lightweight  feed-forward  networks  inserted  within
the transformer architecture, often between attention and
feed-forward blocks [14]. During the training phase, only
the  parameters  of  these  adapter  modules  are  updated,
while the rest of the model remains unchanged.
Other fine-tuning approaches include:
-  BitFit — updates only the bias terms of a neural network,
achieving competitive performance in some NLP tasks.
-  Prompt and Prefix Tuning — adapts transformer models
by learning trainable prompt or prefix vectors.
-  Hypernetwork-Based Tuning — uses a smaller auxiliary
neural  network  to  generate  task-specific  weights  for  the
main model.
These  techniques  provide  a  rich  set  of  tools  for  model
adaptation. However, each of them presents limitations when
applied  to  the  HAR  domain  and  IoT-focused  deployment
scenarios. BitFit significantly reduces the number of trainable
parameters  but  offers  limited  expressive  capacity,  especially
for  non-language  tasks  like  time-series  sensor  classification,
where  bias-only  updates  are  unlikely  to  capture  complex  do-
main shifts. Prompt and Prefix Tuning were primarily designed
for autoregressive language models and rely on manipulating
input  embeddings  or  attention  keys;  this  mechanism  does
not directly translate to time-series transformers, where input
semantics  and  temporal  structure  differ  significantly  from
NLP settings. Hypernetwork-based tuning, while flexible and
effective  in  some  multitask  scenarios,  introduces  additional
computational  overhead  by  requiring  a  separate  network  to
generate weights, which contradicts the memory and runtime
constraints typical of embedded HAR systems.
D.  LoRA
Low-Rank Adaptation (LoRA) [15] is a parameter-efficient
fine-tuning  technique  in  which  additional  low-rank  trainable
matrices are inserted into existing layers, rather than updating
all original parameters. Formally, for a linear map with weight
## W ∈R
d
out
## ×d
in
, LoRA parameterizes a low-rank update ∆W =
BA  with  rank r ≪  min(d
out
## ,d
in
),  where B ∈R
d
out
## ×r
and
## A∈R
r×d
in
. The forward pass becomes
y  = Wx  +
α
r
BAx,
keeping W  frozen  and  training  only A,B  (typically  with A
initialized  to  zero  so  the  initial  function  is  unchanged).  At
inference time, the update can be merged via
## W ← W  +
α
r
## BA.
This  approach  substantially  reduces  the  number  of  trainable
weights  while  maintaining  performance  close  to  that  of  full
fine-tuning. Recent surveys discuss the various design choices
underlying  LoRA,  including  the  selection  of  insertion  points
for  the  adapters,  the  choice  of  rank,  and  the  resulting  trade-
offs between accuracy and efficiency [16], [17]. Comparative
studies  further  indicate  that  LoRA  and  full  fine-tuning  con-
verge to different parameter configurations, which may explain
differences in their ability to generalize to unseen data [18].
Although  LoRA  was  originally  introduced  for  large  lan-
guage  models,  it  has  since  gained  popularity  in  computer
vision [19], [20], speech processing [21], [22], and time series
forecasting  [23],  [24].  LoRA  is  modular,  meaning  it  can  be
added or removed from a pretrained model without altering the
original weights, and it has a low memory footprint, requiring
very  little  additional  storage  for  the  trainable  parameters.
These properties make it especially appealing for HAR.
E.  QLoRA
QLoRA  extends  LoRA  by  combining  it  with  4-bit  quanti-
zation  for  base  model  weights  (using  the  NF4  quantization
format),  keeping  LoRA  matrices  in  higher  precision.  This
reduces memory usage and allows fine-tuning of large models
on  a  single  consumer-grade  GPU  without  significant  accu-
racy  loss.  Key  innovations  include  NF4  quantization,  double
quantization  to  compress  quantization  constants,  and  paged
optimizers to reduce memory peaks during backpropagation. It
has been demonstrated that QLoRA can train very large mod-
els to state-of-the-art quality while using much less memory.
A  well-known  example  is  the  Guanaco  model  family,  which
showed  top  results  on  the  Vicuna-bench  benchmark  and,  in
some cases, matched or even beat models fine-tuned in full 16-
bit precision, but with a fraction of the hardware requirements
## [25].
IR-QLoRA  [26]  replaces  certain  operations  with  integer-
only arithmetic to improve stability when working with quan-
tized  weights.  Another  method,  LoftQ  [27],  integrates  the
quantization  process  with  LoRA  fine-tuning  so  that  the  two
steps  work  together,  which  can  improve  accuracy  and  some-
times even outperform standard QLoRA. These developments
aimed at making quantized fine-tuning more stable and more
accurate, especially for tasks where every bit of memory and
compute power matters.
While  direct  applications  of  QLoRA  to  HAR  are  not  yet
common, parameter-efficient fine-tuning with quantization has
been  actively  explored  in  time-series  modeling  [28],  mak-
ing  QLoRA  a  natural  candidate  for  future  HAR  systems.
For  resource-constrained  HAR  scenarios,  QLoRA  provides
an  interesting  trade-off  between  performance  and  efficiency,
enabling scalable personalization in IoT deployments.
## III.  APPROACH
## A.  Overview
The  objective  of  this  work  is  to  evaluate  three  fine-tuning
strategies for sensor-based HAR—full fine-tuning, LoRA, and
QLoRA—and  to  quantify  their  trade-offs  in  recognition  ac-
curacy, compute, and memory when adapting a high-capacity

TABLE I: Summary of datasets characteristics
## Dataset
#  of
samples
#  of
users
Adopted  DevicesSampling  rateDevice  positionActivities
## HHAR85,5679
Smartphones: Samsung Galaxy S3 mini,
Samsung Galaxy S3, LG Nexus 4,
## Samsung Galaxy S+
Smartwatches: LG watches,
## Samsung Galaxy Gears
from 50 Hz
to 200 Hz
## Smartphones: Waist
## Smartwatches: Wrist
## Biking, Sitting, Standing,
## Walking, Upstairs, Downstairs
MotionSense17,23124Apple iPhone 6s50 Hz
## Waist
## Downstairs, Upstairs, Sitting,
## Standing, Walking, Running
RealWorld356,42715
## Samsung Galaxy S4
LG G Watch R
## 50 Hz
Smartphones: Head, Chest, Upper arm,
## Waist, Thigh, Shin
## Smartwatches: Forearm
## Downstairs, Upstairs, Lying,
## Sitting, Standing, Jumping,
## Walking, Running
UCI10,29930Samsung Galaxy S II50 HzWaist
## Walking, Upstairs, Downstairs,
## Sitting, Standing, Lying
PAMAP215,1778Colibri wireless IMU sensors100 HzWaist, Chest, Wrist
## Rope Jumping, Lying, Sitting,
## Standing, Walking, Running,
Cycling, Nordic walking,
## Upstairs, Downstairs,
Vacuum cleaning, Ironing
model  to  new  domains.  We  ground  the  study  in  a  state-of-
the-art  Transformer  backbone  pretrained  with  a  Masked  Au-
toencoder (MAE), a strong choice for multivariate time series
[3]. Our deployment target is not ultra-constrained microcon-
trollers,  but  edge-class  devices—smartphones,  smartwatches,
and  embedded  gateways—that  can  execute  Transformer  in-
ference  yet  offer  limited  headroom  for  on-device  training
(RAM/VRAM, bandwidth, and battery). Given the backbone’s
size,  efficient  adaptation  is  therefore  both  challenging  and
necessary to meet these edge constraints.
Also, all methods are evaluated under a common protocol,
ensuring that the comparison is both rigorous and representa-
tive of realistic deployment conditions. To our knowledge, this
is the first systematic study in HAR that simultaneously imple-
ments and benchmarks full fine-tuning, LoRA, and QLoRA on
the same backbone, thereby offering novel insights into their
relative merits.
## B.  Pre-training Strategy
Training  a  model  on  a  single  dataset  rarely  captures  the
diversity of sensing conditions across devices, placements, and
user  populations,  often  resulting  in  poor  generalization.  To
address this, we adopt cross-dataset pretraining on five widely
used HAR datasets (see Table I):
1)  Heterogeneity   Human   Activity   Recognition   (HHAR)
[29]:  9 participants wearing 8 smartphones (waist pouch) and
4 smartwatches performed 6 activities. Accelerometer and gy-
roscope were sampled at 50–200 Hz across 12 heterogeneous
devices.
2)  MotionSense  [30]:   24  subjects  carried  an  iPhone  6s
in  the  front  pocket;  accelerometer,  gyroscope,  and  attitude
were  recorded  at  50 Hz  over  6  activities  (walking,  jogging,
stairs, sitting, standing). Controlled protocol with phone-only
sensing.
3)  RealWorld [31]:  15 subjects (18 h total) with a Galaxy
S4 and LG G Watch R at 7 body locations collected accelerom-
eter/gyroscope  at  50 Hz  over  8  activities  in  unconstrained
outdoor  settings.  Notable  class  imbalance  (e.g.,  standing  vs.
jumping) and cross-position variability.
4)  UCI Human Activity Recognition [32]:  30 subjects wore
a  Galaxy  S  II  on  the  waist  (50 Hz)  to  perform  6  activities
in  a  controlled  lab.  A  canonical  benchmark  with  low  device
variability and well-defined conditions.
5)  PAMAP2  [33]:  9  subjects  performed  12  daily/exercise
activities  wearing  three  IMUs  (ankle,  chest,  wrist)  with  ac-
celerometer/gyroscope/magnetometer   at   100 Hz.   Controlled
setup with multi-sensor, multi-location recordings.
We  downsampled  all  datasets  to  50  Hz,  consistent  with
evidence  that  20–50  Hz  is  optimal  for  smartphone  HAR
and  that  accelerometer/gyroscope  suffice;  higher  rates  add
cost   with   marginal   gains   [34].   Each   dataset   was   sensor-
wise  z-normalized  independently  to  prevent  small  corpora
being  dominated  by  larger  ones  (e.g.,  UCI  vs.  HHAR).  To
minimize  position-induced  domain  shift  and  focus  on  data
scarcity,  we  retained  only  waist-mounted  recordings  present
across   datasets.   Signals   were   segmented   into   128-sample
(2.56 s) windows with 50% overlap over the six accelerome-
ter/gyroscope  channels  [?].  Datasets  were  then  combined  by
taking the union of activity labels.
## C.  Evaluation Strategy
In  order  to  evaluate  fine  tuning,  we  use  a  strategy  called
”Leave-One-Dataset-Out”   (LODO)   [9].   Precisely,   At   each
fold,  one  dataset  is  considered  as  left-out  dataset,  while  the
remaining ones are used to create a pre-trained model.
Once  pretrained,  the  encoder  is  fine-tuned  on  the  dataset
that was excluded during pretraining. This protocol is rotated
across  all  combinations,  ensuring  that  each  dataset  serves  as
the  target  domain  exactly  once.  The  LODO  design  provides
a principled and rigorous way to assess cross-dataset transfer
in HAR. Unlike random within-dataset splits, which primarily
test interpolation, LODO explicitly measures extrapolation to
unseen  sources—a  scenario  that  closely  mirrors  real-world
IoT deployments where models must adapt to new hardware,

environments,  or  user  populations.  This  strategy  therefore
offers a comprehensive assessment of how different fine-tuning
strategies perform under realistic domain shift conditions.
Full fine-tuning serves as the baseline: it updates all parame-
ters of the pretrained MAE and thus provides an upper bound
on  recognition  accuracy.  Against  this  baseline,  we  compare
LoRA  and  QLoRA.  These  methods  are  evaluated  under  the
LODO protocol described above, ensuring that the comparison
is  both  rigorous  and  representative  of  realistic  deployment
conditions.
## IV.  IMPLEMENTATION
A.  MAE and Transformer-based architecture
In our framework, the backbone model is a Transformer en-
coder–decoder architecture, while the training pipeline follows
the  Masked  Autoencoder  (MAE)  paradigm.  It  is  important
to  emphasize  this  distinction:  the  Transformer  specifies  the
architectural  building  blocks  (attention  layers,  feed-forward
networks,  residual  connections),  whereas  MAE  defines  the
self-supervised  learning  strategy  (masking  input  patches  and
reconstructing them). Together, they provide a powerful com-
bination  for  learning  robust  spatiotemporal  representations
from multivariate sensor data.
a)  Input  representation  and  patching:  Let X ∈R
## L×C
denote  a  sensor  window  of  length L  with C  channels.  The
MAE  pipeline  first  partitions  the  signal  into  non-overlapping
patches of length P , producing
## L
## P
patches per channel. Each
patch  is  projected  into  a d-dimensional  embedding  and  aug-
mented with positional encodings to retain temporal order:
z
i
= Linear(X
i:i+P
) + p
i
,   z
i
## ∈R
d
## .
During pretraining, a fraction m = 75% of patch tokens is
masked at random. The encoder processes only the visible to-
kens, while the decoder receives the concatenation of encoded
visible tokens and learned mask tokens at masked positions.
b)  Encoder  (Transformer  backbone):  The  encoder  con-
sists  of  six  Transformer  blocks,  each  combining  multi-head
self-attention  (MSA),  a  position-wise  feed-forward  network
(FFN), residual connections, and layer normalization (LN), as
illustrated by Figure 1. Formally, let Z
## (l)
## ∈R
## T×d
denote the
sequence of token embeddings entering the l-th layer:
## ˆ
## Z
## (l)
## = Z
## (l)
## + MSA
##  
## LN(Z
## (l)
## )
## 
## ,
## Z
## (l+1)
## =
## ˆ
## Z
## (l)
## + FFN
##  
## LN(
## ˆ
## Z
## (l)
## )
## 
## .
The  attention  sublayer  aggregates  contextual  information
across  patches,  while  the  FFN  refines  token  representations
independently.  Stacking  these  blocks  produces  progressively
more abstract and robust embeddings of the input sequence.
c)  Decoder  (MAE  pipeline):   The  decoder  mirrors  the
encoder structure. It restores the original sequence by inserting
mask tokens, concatenating them with encoded visible embed-
dings, and predicting the missing patches.
## Fig. 1: Detailed Encoder Architecture.
d)  Pretraining  objective  (MAE):  Let
## ˆ
Y   denote  decoder
predictions  and Y  the  ground-truth  patches.  The  objective  is
the mean squared error (MSE) over the masked positions M:
## L
## MAE
## =
## 1
## |M|
## X
i∈M


## ˆ
## Y
i
## − Y
i


## 2
## 2
## .
e)  Downstream   classification   head:    For   HAR   fine-
tuning,  the  MAE  decoder  is  discarded  and  a  lightweight
classification head is attached to the encoder output. This head
is a small multilayer perceptron (MLP) with BatchNorm and
Dropout,  projecting  the  encoder  embedding  dimension  to K
activity  classes.  It  is  randomly  initialized  and  trained  during
the fine-tuning phase.
B.  LoRA Integration
LoRA  adapters  are  inserted  into  the  encoder’s  linear  pro-
jections,  specifically  within  the  feed-forward  layers  and  the
attention  mechanism.  The  pretrained  weight  matrices  remain
frozen, while the adapters introduce a small number of train-
able  parameters.  During  fine-tuning,  only  these  adapters  and
the  classification  head  receive  gradients,  ensuring  parameter
efficiency without altering the backbone’s core architecture.
1)  LoRA   in   Feed-Forward   Layers:   Within   Transformer
blocks,  FFNs  refine  token  representations  through  nonlinear
transformations.  A  standard  FFN  is  implemented  as  a  two-
layer structure alternating between linear projection and non-
linear activation:
FFN(X) = σ(XW
## 1
+ b
## 1
## )W
## 2
+ b
## 2
## ,
where W
## 1
## ∈R
d×h
projects  the  embedding  into  a  higher-
dimensional  hidden  space h, σ(·)  is  a  nonlinear  activation,
and W
## 2
## ∈R
h×d
maps back to the original dimension d.
With  LoRA,  modifications  are  applied  to  the  linear  trans-
formations. Instead of updating the full matrices W
## 1
and W
## 2
## ,
they are augmented with compact low-rank adapters:
## W
## ′
## 1
## = W
## 1
## + A
## 1
## B
## 1
## ,  W
## ′
## 2
## = W
## 2
## + A
## 2
## B
## 2
## ,
where A
i
## ,B
i
are trainable matrices with rank r ≪ min(d,h).
This  design  preserves  the  pretrained  knowledge  encoded
in W
## 1
and W
## 2
,  while  the  adapters  provide  a  lightweight
mechanism for domain-specific adaptation. Since the nonlinear
activations  are  left  unchanged,  the  expressive  properties  of
the  original  FFN  are  maintained,  and  the  adapted  output  is
seamlessly propagated to subsequent layers.

TABLE  II:  Cross-dataset  recognition  performance  for  different  fine-tuning  strategies.  The  dataset  in  the  first  column  is  the
target used for fine-tuning; all other datasets are pooled to pre-train the model (LODO).
DatasetFine-tuning  MethodAccuracyF1  MacroPrecisionRecall
HHARFull Fine-Tuning0.9810.9800.9810.981
HHARLoRA0.9600.9580.9610.960
HHARQLoRA0.9550.9520.9560.955
RealWorldFull Fine-Tuning0.9070.9070.9170.910
RealWorldLoRA0.8590.8610.8750.866
RealWorldQLoRA0.8600.8610.8750.866
PAMAPFull Fine-Tuning0.8530.8540.8670.863
PAMAPLoRA0.8290.8250.8550.821
PAMAPQLoRA0.8290.8240.8550.821
SenseFull Fine-Tuning0.9610.9580.9600.961
SenseLoRA0.9560.9500.9530.954
SenseQLoRA0.9550.9490.9520.953
UCIFull Fine-Tuning0.9680.9690.9700.971
UCILoRA0.9520.9530.9550.956
UCIQLoRA0.9530.9530.9560.956
2)  LoRA  in  Attention  Mechanisms:   Self-attention  layers
project  each  token  sequence X ∈R
n×d
into  queries,  keys,
and values:
## Q = XW
## Q
## ,  K = XW
## K
## ,  V  = XW
## V
## ,
with  projection  matrices W
## Q
## ∈R
d×d
q
## , W
## K
## ∈R
d×d
k
,  and
## W
## V
## ∈R
d×d
v
. The attention output is then:
Attention(Q,K,V ) = softmax
## 
## QK
## ⊤
## √
d
k
## 
## V,
followed by a projection back to dimension d:
Output = Attention(Q,K,V )W
## O
## ,  W
## O
## ∈R
d
v
## ×d
## .
With  LoRA,  each  weight  matrix W  is  kept  frozen  and  a
low-rank update is added:
## W
## ′
## = W + ∆W,∆W  = AB,
where A ∈R
d×r
and B ∈R
r×k
are  trainable  matrices,
reducing  the  number  of  trainable  parameters  from d× k  to
r(d + k).
Concretely, in the attention mechanism we obtain:
## [Q,K,V ] = X
##  
## [W
## Q
## ,W
## K
## ,W
## V
## ] + [A
## Q
## B
## Q
## ,A
## K
## B
## K
## ,A
## V
## B
## V
## ]
## 
## ,
Output = Attention(Q,K,V ) (W
## O
## + A
## O
## B
## O
## ).
Each pair (A
## ·
## ,B
## ·
) thus represents a LoRA adapter inserted
into the corresponding projection, enabling efficient adaptation
while leaving the main Transformer backbone intact.
C.  Quantized LoRA Extension
Quantized  Low-Rank  Adaptation  (QLoRA)  builds  directly
on the LoRA framework by combining low-rank adapters with
quantization of the pretrained weight matrices. The motivation
is  to  further  reduce  memory  usage  while  retaining  the  rep-
resentational  power  of  large  Transformer  backbones,  thereby
enabling fine-tuning on hardware with resource constraints.
In our implementation, all major projection layers of the en-
coder—namely the attention projections (W
## Q
## ,W
## K
## ,W
## V
## ,W
## O
## )
and  the  feed-forward  projections  (W
## 1
## ,W
## 2
)—are  stored  in  a
4-bit  quantized  format.  These  quantized  weights  are  frozen
during  adaptation,  while  the  LoRA  adapters  (A,B)  and  the
classification head are trained in higher precision (e.g., FP16
or BF16).
Formally, let W ∈R
d×k
denote a pretrained weight matrix.
Instead  of  storing W  in  full  precision,  it  is  compressed  into
a  4-bit  representation
## ̃
W  via  blockwise  quantization.  During
forward  propagation,
## ̃
W  is  dequantized  into  an  approximate
floating-point representation
## ˆ
W , and the effective update is:
## W
## ′
## =
## ˆ
W + α· (AB),
where A ∈R
d×r
and B ∈R
r×k
are  trainable  low-rank
matrices  and α  is  a  scaling  factor.  Gradients  are  backprop-
agated through
## ˆ
W , but optimizer updates are restricted to the
LoRA adapters and the classification head. The quantized base
weights
## ̃
W  remain  fixed,  preserving  the  knowledge  encoded
during large-scale pretraining.
To  maintain  stability,  certain  components  such  as  layer
normalization  parameters,  positional  embeddings,  and  patch
embeddings  are  kept  in  higher  precision.  This  precaution
avoids  numerical  instabilities  and  ensures  that  quantization
errors   do   not   accumulate   across   layers.   Overall,   QLoRA
provides  a  balance  between  three  key  requirements:  preserv-
ing  representational  fidelity,  reducing  the  memory  footprint
of  frozen  weights,  and  enabling  deployability  on  memory-
constrained devices such as wearables and IoT platforms.
## V.  RESULTS
## A.  Overview
This  section  evaluates  the  empirical  performance  of  the
fine-tuning  methods  presented  earlier.  All  experiments  are
conducted  on  the  same  MAE-Transformer  backbone  under

a  unified  training  setup  and  evaluation  protocol  to  ensure
comparability.  Each  model  is  fine-tuned  for  50  epochs  on
five benchmark HAR datasets using a Leave-One-Dataset-Out
(LODO)  validation  scheme.  Within  each  target  dataset,  70%
of  the  data  is  used  for  fine-tuning  and  30%  is  reserved  for
validation.
We analyze several practical aspects that are critical for real-
world deployment:
-  Recognition quality using common metrics;
-  Parameter   efficiency:   number   of   trainable   parameters
introduced by each method;
-  Computational cost: training time measured in seconds;
-  Memory  usage:  parameter  storage  and  buffer  memory
during fine-tuning.
B.  Recognition Accuracy across Datasets
In this section, we evaluate the recognition performance of
each fine-tuning strategy across the five open-source datasets
presented  earlier  (HHAR,  RealWorld,  PAMAP,  Sense,  and
UCI). Each dataset presents a different level of difficulty due
to variations in sensor modalities, sampling frequencies, user
populations,  and  activity  granularity.  All  models  were  evalu-
ated  in  the  LODO  setting  to  simulate  realistic  domain  shift
scenarios, where the model must generalize to data collected
from  a  previously  unseen  source.  We  report  four  commonly
used  classification  metrics:  Accuracy  (overall  proportion  of
correct predictions), Macro-F1 score (balance between preci-
sion and recall averaged across classes), Precision (proportion
of correct positive predictions), and Recall (proportion of true
positives correctly identified).
Table  II  reports  a  detailed  comparison  of  these  metrics
across  datasets  and  fine-tuning  methods.  For  each  entry,  the
target dataset corresponds to the held-out domain in the LODO
setting, i.e., the model is trained on the four remaining datasets
and  evaluated  on  the  unseen  one.  As  expected,  full  fine-
tuning  consistently  achieves  the  highest  recognition  accuracy
and  macro-F1  scores  across  all  benchmarks.  However,  both
LoRA and QLoRA perform competitively, often matching or
closely approaching full fine-tuning performance, particularly
on  larger  or  less  noisy  datasets  such  as  HHAR  and  UCI.
Notably, QLoRA shows a slight underperformance compared
to LoRA in certain settings (e.g., PAMAP), likely due to the
approximation error introduced by quantization. Nevertheless,
the performance gap remains relatively small (typically within
1–2%),  indicating  that  the  temporal  representations  learned
during MAE pretraining are sufficiently robust to tolerate low-
bit adaptation mechanisms.
Overall, these results validate the effectiveness of parameter-
efficient tuning strategies in HAR: despite significantly fewer
trainable  weights  and  reduced  memory  usage,  both  LoRA
and QLoRA maintain strong classification performance across
diverse domains.
C.  Trainable Parameters and Model Efficiency
While  recognition  performance  is  essential  for  evaluating
a  fine-tuning  method,  it  is  equally  important  to  consider
its  efficiency  in  terms  of  computational  cost  and  memory
usage—particularly in resource-constrained environments such
as  mobile  or  embedded  HAR  systems.  This  subsection  ex-
amines the parameter footprint and training overhead of each
method  to  better  highlight  the  trade-offs  between  adaptation
quality and efficiency. Specifically, we report:
-  the number of trainable parameters and the total param-
eter count (trainable + frozen),
-  training time over 50 epochs (in seconds),
-  and  peak  memory  consumption  during  fine-tuning  (in
megabytes).
Table III summarizes the number of trainable and total param-
eters  required  for  each  fine-tuning  strategy.  These  values  are
architecture-dependent  and  remain  constant  across  datasets,
since  all  experiments  rely  on  the  same  MAE  backbone  and
identical  low-rank  adapter  configurations  (e.g.,  fixed  rank r
for LoRA/QLoRA).
Fine-tuning  MethodTrainable  ParamsTotal  Params
Full Fine-Tuning2,210,8572,210,857
LoRA428,8322,636,105
QLoRA428,8322,636,105
TABLE III: Trainable and total parameters for each fine-tuning
strategy.
Full  fine-tuning  updates  all  weights  of  the  pretrained  en-
coder and classification head, totaling approximately 2.2 mil-
lion parameters. In contrast, LoRA and QLoRA freeze the pre-
trained weights and optimize only a small number of adapter
parameters  (about  428,000),  yielding  a  fivefold  reduction  in
the number of trainable weights.
Interestingly,   the   total   parameter   count   for   LoRA   and
QLoRA   is   slightly   higher   than   for   full   fine-tuning.   This
occurs because LoRA introduces additional low-rank matrices
(A, B)  alongside  the  frozen  pretrained  weights W .  Since
these  adapters  are  added  as  external  components  rather  than
replacing  the  original  weights,  the  full-precision  parameters
are  still  retained.  In  other  words,  LoRA  and  QLoRA  extend
the model architecture instead of overwriting existing weights.
This explains why TrainableLoRA + TotalFull > Total
LoRA
, as
adapter  weights  function  as  parallel  branches  within  selected
layers (e.g., attention and feed-forward projections) rather than
as direct substitutions.
D.  Memory consumption
Next,  we  analyze  memory  consumption  during  adaptation.
As  shown  in  Table  IV,  both  LoRA  and  QLoRA  allocate  ap-
proximately 1.64 MB for trainable parameters, corresponding
to  the  total  size  of  the  low-rank  adapter  matrices  introduced
by  the  fine-tuning  strategy.  The  column  Frozen  Param  (MB)
quantifies  the  memory  occupied  by  the  pretrained  encoder’s
frozen  weights,  while  Trainable  Param  (MB)  indicates  the
size of the newly added trainable components. Together, these
two quantities define the static memory footprint of the model
parameters.
In practice, however, fine-tuning also incurs additional tem-
porary  costs,  reported  in  the  Buffer  Memory  (MB)  column.

This  value  captures  the  peak  size  of  intermediate  memory
buffers  required  during  training,  including  those  for  activa-
tions,  gradients,  and  extra  computations  introduced  by  quan-
tization mechanisms.
## Method
## Frozen
Param  (MB)
## Trainable
Param  (MB)
## Buffer
Memory  (MB)
LoRA10.061.640.01
QLoRA6.221.644.82
TABLE IV: Memory consumption for LoRA and QLoRA.
In  the  case  of  LoRA,  buffer  memory  usage  is  negligible
(0.01 MB), since all operations rely on standard floating-point
arithmetic without additional processing of frozen weights. By
contrast, QLoRA introduces a memory–computation trade-off:
quantizing the frozen weights reduces their static footprint by
about 40% (from 10.06 MB to 6.22 MB), but requires on-the-
fly dequantization during forward passes.
In  our  CPU-based  implementation,  this  process  involves
storing  auxiliary  scale  factors  and  performing  extra  matrix
operations to reconstruct floating-point tensors at runtime. As a
result, buffer usage rises sharply to 4.82 MB—several hundred
times higher than that of LoRA.
This  overhead  largely  reflects  the  limitations  of  our  sim-
plified   implementation.   Optimized   GPU-based   versions   of
QLoRA  (e.g.,  with  fused  low-bit  kernels  and  better  memory
management) handle dequantization far more efficiently, dras-
tically reducing buffer costs and yielding substantially greater
overall memory savings.
E.  Training time
Training  time  is  another  key  efficiency  metric,  as  shown
in Table V. As expected, full fine-tuning consistently requires
the longest time across datasets, owing to the need to update
all model weights and maintain large optimizer states. LoRA
generally achieves faster training than QLoRA, likely because
QLoRA   introduces   additional   overhead   from   quantization
and dequantization during forward passes. Nevertheless, both
methods provide substantial efficiency gains compared to full
fine-tuning.
DatasetFull  Fine-TuningLoRAQLoRA
## HHAR277225052598
## REALWORLD565750735195
## PAMAP117010111037
## SENSE145312991378
## UCI927881875
TABLE  V:  Training  time  (in  seconds)  for  each  fine-tuning
method across datasets.
It is important to note that total training time also depends
on the size and complexity of the dataset. Datasets with more
samples or longer input sequences require additional iterations
per epoch and larger memory usage during training, which can
substantially increase runtime. This explains the large variation
in  training  time  observed  even  within  the  same  fine-tuning
strategy across datasets. For instance, REALWORLD requires
over  5600  seconds  for  full  fine-tuning,  whereas  the  smaller
UCI dataset completes training in under 1000 seconds.
Although LoRA reduces the number of trainable parameters
by  more  than  80%,  the  resulting  decrease  in  training  time  is
relatively  modest  (around  10%).  This  limited  speedup  arises
because  LoRA  does  not  reduce  the  number  of  operations
during forward and backward passes: the frozen encoder still
participates fully in computation, and partial backward graphs
must  be  constructed  for  adapter  training.  In  addition,  LoRA
introduces extra matrix multiplications, which partially offset
its  efficiency  gains.  In  practice,  the  majority  of  training  time
is dominated by backbone computations and batch processing
rather than parameter updates. Furthermore, training efficiency
is  influenced  by  several  factors  such  as  model  depth,  batch
size,  the  number  of  normalization  and  activation  layers,  and
the choice of optimizer—most of which remain unaffected by
parameter sparsity.
F.  Impact of LoRA Rank on Accuracy and Efficiency
To   better   understand   the   trade-offs   between   adaptation
quality  and  efficiency  in  LoRA,  we  analyze  the  effect  of
the  adapter  rank—a  key  hyperparameter  that  determines  the
expressive  capacity  of  the  low-rank  updates.  The  rank  con-
trols  the  dimensionality  of  the  matrices A  and B  in  each
LoRA  module  and  thus  directly  impacts  both  the  number
of  trainable  parameters  and  the  computational  cost  of  fine-
tuning. Higher ranks increase the number of trainable weights
and may enhance model expressiveness, whereas lower ranks
reduce  memory  usage  and  accelerate  training,  but  can  limit
performance.
To  explore  this  trade-off,  we  conducted  a  focused  ex-
periment   on   the   UCI   dataset   by   fine-tuning   the   pre-
trained  MAE  encoder  with  LoRA  adapters  of  varying  ranks:
{8, 16, 20, 32, 48, 64}.  For  each  configuration,  we  measured
the macro-averaged F1 score after 50 epochs of training, along
with the total training time in seconds. The results are reported
in Table VI and illustrated in Figure 2.
LoRA  RankF1  MacroTraining  Time  (sec)
## 80.9352916
## 160.9395934
## 200.9434945
## 320.9531971
## 480.96021015
## 640.96491049
TABLE  VI:  Effect  of  LoRA  matrix  rank  on  classification
quality and training speed (UCI dataset).

Fig.  2:  F1  Macro  and  training  time  as  a  function  of  LoRA
adapter rank.
As expected, classification quality improves monotonically
with increasing rank. Macro-F1 scores rise from 0.9352 at rank
8  to  0.9649  at  rank  64,  indicating  that  higher-rank  adapters
capture more nuanced task-specific representations. This gain,
however,  comes  at  the  expense  of  training  efficiency:  total
runtime  increases  from  916  seconds  (rank  8)  to  over  1049
seconds   (rank   64).   These   results   highlight   a   clear   trade-
off:  smaller  ranks  enable  faster  and  more  memory-efficient
adaptation, whereas larger ranks improve accuracy at the cost
of higher computation.
Notably,  performance  gains  saturate  around  ranks  48–64,
suggesting diminishing returns beyond this point. In practice,
moderate ranks (e.g., 32 or 48) may therefore provide the best
balance between accuracy and efficiency, particularly in time-
or memory-constrained deployment scenarios.
Although  this  experiment  was  conducted  only  on  the  UCI
dataset,  the  observed  trend  is  likely  to  generalize  to  other
HAR domains,  given the shared temporal  structure of sensor
data. Overall, the findings emphasize the importance of tuning
LoRA  hyperparameters  to  match  both  task  complexity  and
resource constraints.
G.  Robustness to Training Data Size
Earlier in our evaluation, we showed that LoRA significantly
reduces  the  number  of  trainable  parameters  compared  to  full
fine-tuning. This opens the possibility of performing effective
adaptation even when only a limited amount of labeled training
data is available.
To  explore  this  hypothesis,  we  fine-tuned  models  using
various  train/test  splits,  ranging  from  70/30  to  30/70,  and
recorded  the  resulting  accuracy  on  the  test  set.  The  results
are summarized in Table VII.
SplitFull  FT  AccuracyLoRA  AccuracyLoRA  /  Full  FT
## 70/300.96760.95280.9847
## 60/400.96510.95080.9852
## 50/500.96180.94830.9859
## 40/600.95430.94160.9867
## 30/700.93800.92570.9869
TABLE  VII:  Accuracy  comparison  under  different  train/test
splits.
As expected, both full fine-tuning and LoRA exhibit a grad-
ual decline in recognition accuracy as the training set becomes
smaller.  However,  the  drop  in  LoRA  accuracy  is  relatively
modest compared to full fine-tuning, and the ratio of LoRA to
full  fine-tuning  performance  (Table  VII)  remains  remarkably
stable, even improving as the training size decreases.
This  suggests  that  LoRA  maintains  a  higher  degree  of
data  efficiency:  it  generalizes  well  even  with  fewer  training
examples.  In  practical  scenarios  where  collecting  large-scale
labeled data is expensive or impractical, LoRA offers a com-
pelling alternative that delivers strong performance with fewer
parameters and less supervision.
## H.  Synthesis
In  summary,  both  LoRA  and  QLoRA  demonstrate  sub-
stantial  efficiency  gains  over  full  fine-tuning,  while  preserv-
ing competitive recognition performance. As expected, LoRA
significantly  reduces  the  number  of  trainable  parameters  and
achieves faster training times by freezing the pretrained back-
bone  and  optimizing  only  lightweight  adapter  layers.  This
makes  it  particularly  attractive  for  rapid  model  adaptation  in
resource-constrained  environments,  as  well  as  for  few-shot
learning scenarios where only limited labeled data is available.
QLoRA  builds  upon  this  foundation  by  quantizing  the
frozen  weights,  thereby  further  reducing  the  static  memory
footprint  without  degrading  classification  quality.  Although
our implementation  shows a  modest  increase  in  buffer mem-
ory,  QLoRA  maintains  the  same  number  of  trainable  param-
eters as LoRA while substantially lowering the size of frozen
weights.  This  makes  QLoRA  especially  well  suited  for  de-
ployments where memory capacity is the primary bottleneck,
such as on low-power IoT devices.
## VI.  CONCLUSION
In   this   paper,   we   addressed   the   challenge   of   adapting
deep  learning  models  for  human  activity  recognition  (HAR)
under limited computational and memory resources. Since full
fine-tuning  is  often  impractical  in  real-world  deployments,
we  investigated  parameter-efficient  alternatives—LoRA  and
QLoRA—against  the  full  fine-tuning  baseline.  All  methods
were evaluated on the same transformer-based backbone using
a  Leave-One-Dataset-Out  protocol  across  five  HAR  bench-
marks,  simulating  deployment  to  unseen  domains.  Results
show  that  LoRA  and  QLoRA  achieve  competitive  perfor-
mance,  typically  within  1–2%  of  full  fine-tuning,  while  re-
quiring  far  fewer  trainable  parameters.  Importantly,  LoRA
also  reduces  the  amount  of  labeled  data  needed  to  adapt  a
pretrained  model,  enabling  effective  fine-tuning  under  scarce

supervision. LoRA reduced trainable weights more than five-
fold and improved training time, while QLoRA further lowered
memory  usage  by  about  40%  through  quantization  of  frozen
weights.
Additional analyses revealed that LoRA’s accuracy improves
with  adapter  rank  up  to  a  plateau  around  rank  32,  and  that
it  remains  robust  under  reduced  supervision,  retaining  over
98%  of  full  fine-tuning  accuracy  even  with  limited  labeled
data;  in  practice,  this  lowers  the  labeled-data  requirement  to
reach  a  target  accuracy.  These  findings  confirm  that  LoRA
and  QLoRA  offer  scalable  and  data-efficient  personalization
strategies for HAR.
Future  directions  include  extending  evaluation  to  CNN  or
hybrid  backbones,  exploring  online  and  continual  learning
settings,  and  automating  adapter/quantization  configurations
based  on  deployment  constraints.  Another  ongoing  line  of
work is to integrate LoRA into a federated learning pipeline,
used  in  a  hybrid  fashion  with  conventional  fine-tuning  (e.g.,
on selected layers or rounds) to limit client divergence (client
drift) and to enable faster global model convergence [35]
## REFERENCES
[1]  C.  Becker,  C.  Julien,  P.  Lalanda,  and  F.  Zambonelli,  “Pervasive  com-
puting   middleware:   current   trends   and   emerging   challenges,”
## CCF
TransactionsonPervasiveComputingandInteraction, vol. 1, 02 2019.
[2]  A.  Vaswani,  N.  Shazeer,  N.  Parmar,  J.  Uszkoreit,  L.  Jones,  A.  N.
Gomez,  Ł.  Kaiser,  and  I.  Polosukhin,  “Attention  is  all  you  need,”  in
AdvancesinNeuralInformationProcessingSystems,  vol.  30,  2017.
[Online]. Available: https://arxiv.org/abs/1706.03762
[3]  S. Ek, F. Portet, and P. Lalanda, “Transformer-based models to deal with
heterogeneous  environments  in  human  activity  recognition,”Personal
andUbiquitousComputing, pp. 1–14, 2023.
[4]  S. Ek, R. Presotto, G. Civitarese, F. Portet, P. Lalanda, and C. Bettini,
“Comparing  self-supervised  learning  techniques  for  wearable  human
activity  recognition,”CCFTransactionsonPervasiveComputingand
Interaction, vol. 7, 324–341, 2025.
[5]  R.  Kawabata,  K.  Horie,  P.  Lalanda,  and  H.  Kitagawa,  “Developing
switching  mechanism  to  address  challenging  subjects  in  sleep  stage
scoring,”   inProceedingsoftheIEEEInternationalConferenceon
HealthcareInformatics(ICHI), 2025, pp. 501–507.
[6]  C.  Jobanputra,  J.  Bavishi,  and  N.  Doshi,  “Human  activity  recognition:
A survey,”ProcediaComputerScience, vol. 155, pp. 698–703, 2019.
## [7]  A. Stisen, H. Blunck, S. Bhattacharya, T. S. Prentow, M. B. Kjærgaard,
A.  Dey,  T.  Sonne,  and  M.  M.  Jensen,  “Smart  devices  are  different:
Assessing  and  mitigating  mobile  sensing  heterogeneities  for  activity
recognition,” in13thACMConferenceonEmbeddedNetworkedSensor
Systems, 2015, p. 127–140.
[8]  N. S. Khan and M. S. Ghani, “A survey of deep learning-based models
for human activity recognition,”WirelessPersonalCommunications, vol.
120, pp. 1593–1635, 2021.
[9]  R. Presotto, S. Ek, G. Civitarese, F. Portet, P. Lalanda, and C. Bettini,
“Combining  public  human  activity  recognition  datasets  to  mitigate
labeled   data   scarcity,”   in
2023IEEEInternationalConferenceon
SmartComputing(SMARTCOMP).IEEE, 2023. [Online]. Available:
https://arxiv.org/abs/2306.13735
[10]  P. Lalanda and C. Marin, “A domain-configurable development environ-
ment for  service-oriented applications,”IEEESoftware, vol.  24, no.  6,
pp. 31–38, 2007.
[11]  J. Estublier, G. Vega, P. Lalanda, and T. Leveque, “Domain specific engi-
neering environments,” in
200815thAsia-PacificSoftwareEngineering
Conference, 2008, pp. 553–560.
[12]  J. Dodge, G. Ilharco, R. Schwartz, A. Farhadi, H. Hajishirzi, and N. A.
Smith, “Fine-tuning pretrained language models: Weight initializations,
data orders, and early stopping,”arXivpreprintarXiv:2002.06305, 2020.
[13]  V. Parthasarathy, X. Huang, J. Yu, Y. Li, B. Xu, and G. Jin, “The ultimate
guide to fine-tuning llms from basics to breakthroughs,”arXivpreprint
arXiv:2408.13296, 2024.
## [14]  N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe,
A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer
learning  for  nlp,”  in
InternationalConferenceonMachineLearning
(ICML), 2019, pp. 2790–2799.
[15]  E.  J.  Hu,  Y.  Shen,  P.  Wallis,  Z.  AllenZhu,  Y.  Li,  S.  Wang,  L.  Wang,
and  W.  Chen,  “Lora:  Low–rank  adaptation  of  large  language  models,”
InternationalConferenceonLearningRepresentations(ICLR), 2021.
[16]  Z.  Liu,  G.  Yan,  Y.  Zhangetal.,  “A  survey  on  lora  of  large  language
models,”arXivpreprintarXiv:2407.11046,  2024.  [Online].  Available:
https://arxiv.org/abs/2407.11046
## [17]  L.    Wang,    X.    Zhou
etal.,    “Parameter    efficient    fine    tuning:
A   comprehensive   survey,”arXivpreprintarXiv:2404.13506,   2024.
[Online]. Available: https://arxiv.org/abs/2404.13506
[18]  S.  Biderman,  S.  Ghoshetal.,  “Lora  vs  full  fine-tuning:  An  illusion
of   equivalence,”arXivpreprintarXiv:2410.21228,   2024.   [Online].
Available: https://arxiv.org/abs/2410.21228
## [19]  L. Mi, W. Wang, W. Tu, Q. He, R. Kong, X. Fang, Y. Dong, Y. Zhang,
Y.  Li,  M.  Li,  H.  Dai,  G.  Chen,  and  Y.  Liu,  “V-lora:  An  efficient  and
flexible system boosts vision applications with lora lmm,” 2024.
[20]  Z.  Cui,  A.  Tian,  Z.  Ying,  and  J.  Lu,  “Ac–lora:  Auto  component  lora
for personalized artistic style image generation,” inEighthInternational
ConferenceonComputerGraphicsandVirtuality(ICCGV), 2025.
[21]  Z.   Song,   J.   Zhuo,   Y.   Yang,   Z.   Ma,   S.   Zhang,   and   X.   Chen,
“Lora-whisper:  Parameter-efficient  and  extensible  multilingual  asr,”  in
Interspeech.2024, 09 2024, pp. 3934–3938.
[22]  W. Liu, Y. Qin, Z. Peng, and T. Lee, “Sparsely shared lora on whisper
for  child  speech  recognition,”  inIEEEInternationalConferenceon
Acoustics,SpeechandSignalProcessing(ICASSP), 2024.
[23]  D.  Gupta,  A.  Bhatti,  S.  Parmar,  C.  Dan,  Y.  Liu,  B.  Shen,  and  S.  Lee,
“Low-rank  adaptation  of  time  series  foundational  models  for  out-of-
domain modality forecasting,”InternationalConferenceonMultimodal
Interaction(ICMI), 2024.
[24]  W.  Ruan,  W.  Chen,  X.  Dang,  J.  Zhou,  W.  Li,  X.  Liu,  and  Y.  Liang,
“St-lora: Low-rank adaptation for spatio-temporal forecasting,” 2024.
[25]  T.  Dettmers,  A.  Pagnoni,  A.  Holtzman,  and  L.  Zettlemoyer,  “Qlora:
Efficient finetuning of quantized llms,”arXivpreprintarXiv:2305.14314,
- [Online]. Available: https://arxiv.org/abs/2305.14314
## [26]  H.   Qin
etal.,   “Accurate   lora-finetuning   quantization   of   llms   via
integer-only   reparameterization,”
arXivpreprintarXiv:2402.05445,
- [Online]. Available: https://arxiv.org/abs/2402.05445
[27]  Y.   Li,   Y.   Zhouetal.,   “Loftq:   Lora-fine-tuning-aware   quantization
for  large  language  models,”arXivpreprintarXiv:2310.08659,  2023.
[Online]. Available: https://arxiv.org/abs/2310.08659
[28]  D.Guptaetal.,“Low-rankadaptationoftimeseries
foundationalmodelsforout-of-domainmodalityforecasting,”
arXivpreprintarXiv:2405.10216,2024.[Online].Available:
https://arxiv.org/abs/2405.10216
## [29]  A. Stisen, H. Blunck, S. Bhattacharya, T. S. Prentow, M. B. Kjærgaard,
A.  Dey,  T.  Sonne,  and  M.  M.  Jensen,  “Smart  devices  are  different:
Assessing  and  mitigating  mobile  sensing  heterogeneities  for  activity
recognition,” inProceedingsofthe13thACMConferenceonEmbedded
NetworkedSensorSystems, New York, NY, USA, 2015, p. 127–140.
[30]  M.   Malekzadeh,   R.   G.   Clegg,   A.   Cavallaro,   and   H.   Haddadi,
“Protecting  sensory  data  against  sensitive  inferences,”  in
## Proceedings
ofthe1stWorkshoponPrivacybyDesigninDistributedSystems,
ser.  W-P2DS’18.New  York,  NY,  USA:  ACM,  2018,  pp.  2:1–2:6.
[Online]. Available: http://doi.acm.org/10.1145/3195258.3195260
[31]  T.  Sztyler  and  H.  Stuckenschmidt,  “On-body  localization  of  wearable
devices:  An  investigation  of  position-aware  activity  recognition,”  in
2016IEEEInternationalConferenceonPervasiveComputingand
Communications(PerCom), 2016, pp. 1–9.
[32]  D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, “A public
domain  dataset  for  human  activity  recognition  using  smartphones,”  in
21stEuropeanSymposiumonArtificialNeuralNetworks,ESANN2013,
Bruges,Belgium,April24-26,2013, 2013.
[33]  A.  Reiss  and  D.  Stricker,  “Introducing  a  new  benchmarked  dataset  for
activity monitoring,” in201216thInternationalSymposiumonWearable
Computers.    IEEE, 2012, pp. 108–109.
[34]  W. Sousa Lima, E. Souto, K. El-Khatib, R. Jalali, and J. Gama, “Human
activity recognition using inertial sensors in a smartphone: An overview,”
Sensors, vol. 19, p. 3213, 07 2019.
[35]  S. Ek, F. Portet, P. Lalanda, and G. Vega, “Evaluation and comparison
of  federated  learning  algorithms  for  human  activity  recognition  on
smartphones,”PervasiveandMobileComputing, vol. 87, 2022.