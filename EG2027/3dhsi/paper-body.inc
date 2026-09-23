\title[Agentic Contact: Scene-Side Grounding for Zero-Shot HSI Synthesis]%
      {Agentic Contact: Scene-Side Grounding for Zero-Shot Human-Scene Interaction Synthesis}

% Anonymous conference submission; replace with the assigned submission ID.
\author[SUBMISSION ID]{SUBMISSION ID}

\begin{document}

\teaser{
  \centering
  \includegraphics[width=0.9\linewidth]{figures/teaser.jpg}
  \caption{Overview and representative results. Given an interaction
    instruction, a calibrated scene view, and reconstructed scene geometry,
    our method generates a human interaction image, checks and refines
    body-part-specific scene contact regions, and fits a scene-aligned 3D human. The bottom
    row shows synthesized interactions across different scenes.}
  \label{fig:teaser}
}

\maketitle
%-------------------------------------------------------------------------
\begin{abstract}
Populating reconstructed 3D scenes with humans requires not only plausible body poses but also accurate contact with the surfaces involved in the intended interaction. Pretrained image generators can provide diverse visual interaction hypotheses, but the scene regions that individual body parts should contact remain implicit. Incorrectly inferred contact targets can therefore draw a fitted 3D body toward the wrong scene surfaces. However, the interaction priors of these same image generators can be leveraged to recover explicit contact information. We present \textbf{SceneInteract}, a zero-shot method that builds on this insight to explicitly ground body-part-specific contact regions on scene surfaces. Given a language instruction, a calibrated RGB scene view, and a reconstructed scene mesh, our method first synthesizes a visual interaction hypothesis using a pretrained image generator, then uses image generation again to propose body-part-specific contact regions on the original scene view. A vision-language model evaluates these regions against the generated interaction and provides targeted feedback to correct missing or misplaced contacts. The refined regions are projected onto the reconstructed mesh and used as geometric constraints for scene-aware 3D body fitting. Experiments across diverse interactions demonstrate that SceneInteract achieves lower contact distances between participating body parts and their corresponding scene surfaces, as well as higher overall vision-language scores for rendered interactions, than the evaluated baselines.

%-------------------------------------------------------------------------
\begin{CCSXML}
<ccs2012>
<concept>
<concept_id>10010147.10010371.10010396</concept_id>
<concept_desc>Computing methodologies~Shape modeling</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10010147.10010178.10010224.10010225.10010227</concept_id>
<concept_desc>Computing methodologies~Scene understanding</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
\end{CCSXML}

\ccsdesc[500]{Computing methodologies~Shape modeling}
\ccsdesc[300]{Computing methodologies~Scene understanding}

\printccsdesc   
\end{abstract}  
%-------------------------------------------------------------------------
\section{Introduction}

Populating reconstructed 3D scenes with virtual people supports character staging in games and immersive environments, while scene-aligned interaction poses can serve as keyframes for longer animations~\cite{li2024genzi,li2026genhsi}. In these applications, a plausible body must be grounded in the scene geometry: a seated person needs support from the furniture beneath them, and someone using a treadmill must place the feet on the belt and the hands on the appropriate rails. These contacts must remain plausible from different viewpoints. Generating such configurations from a language instruction could reduce manual character placement, but requires inferring both the body pose and the specific scene surfaces it should touch.

Learning-based methods model body-scene proximity, contact, and interaction semantics~\cite{zhang2020place,hassan2021posa,zhao2022coins}, but their learned priors reflect the coverage of their training data. Pretrained image and language models offer broader priors for unfamiliar scenes and actions. Recent methods use generated human images and geometric fitting to synthesize interactions~\cite{li2024genzi,liu2026funhsi}, or combine pretrained human, scene, and contact estimates to reconstruct them~\cite{muralidhar2025physic}. These approaches reduce dependence on paired 3D interaction data, while leaving the geometric grounding of their predictions to subsequent fitting.

Contact is central to that grounding. PROX~\cite{hassan2019prox} established the value of scene contact and interpenetration constraints for body fitting. PhySIC~\cite{muralidhar2025physic} initializes contact on human vertices, GenZI~\cite{li2024genzi} infers body-part placement from generated image hypotheses, and FunHSI~\cite{liu2026funhsi} associates body parts with functional scene elements. These representations guide fitting, but can leave the particular scene patch required by a depicted interaction unresolved. A body contact label does not specify its scene correspondence, while an extended functional surface can admit several contact locations.

We focus on explicitly recovering these scene-side correspondences. Given an interaction hypothesis, our goal is to identify the surface region that each participating body part should contact and verify its placement before fitting the body. This distinction matters because an incorrect target can draw an otherwise plausible pose toward the wrong location. A single contact prediction may omit a required region, place it on an adjacent surface, or assign it an inappropriate extent. Making these regions explicit allows such errors to be inspected and corrected before they become geometric constraints.

We present \textbf{Agentic Contact}, a zero-shot method that synthesizes a static SMPL-X~\cite{pavlakos2019smplx} body from a language instruction, a calibrated RGB view, and a reconstructed scene mesh. A coarse interaction graph specifies the participating body parts and scene objects, and a generated human image provides a visual hypothesis of the action. An image generation model then proposes body-part-specific scene contact regions. A vision-language model checks their placement against the reference interaction and the original scene, providing targeted feedback for refinement. The resulting regions are projected onto the reconstructed mesh as 3D targets. A scene-aware optimization adjusts a monocular body estimate to satisfy these targets while penalizing penetration and retaining a plausible pose. The method uses pretrained models without task-specific training or fine-tuning on paired 3D human-scene interactions.

We evaluate the method on 23 interactions in 15 ScanNet++~\cite{yeshwanth2023scannetpp} scenes. Ablations show that verifying and correcting contact regions improves localization and final 3D contact accuracy compared with single-shot estimation. Against PROX and PhySIC fitted to the same generated image, our method achieves lower contact distance; its rendered interactions also receive higher overall vision-language scores than those of PROX, PhySIC, and GenZI.

Our contributions are:
\begin{itemize}
\item An agentic procedure for localizing, verifying, and correcting body-part-specific contact regions on scene surfaces.
\item A zero-shot synthesis method that converts these regions into geometric constraints for fitting humans in reconstructed scenes.
\item An evaluation with annotated scene contact regions, controlled ablations, and baseline comparisons that examines contact accuracy and rendered interaction quality.
\end{itemize}

%-------------------------------------------------------------------------
\section{Related Work}
\label{sec:related-work}

\subsection{Learning-based human-scene interaction synthesis}

Early approaches to human-scene interaction synthesis learned how scene geometry and semantics constrain human placement. Li et al.~\cite{li2019putting} combined image-derived pose information with scene geometry to generate training examples for predicting plausible human configurations in indoor environments. Subsequently, PLACE~\cite{zhang2020place} modeled body-scene proximity through distances between scene basis points and the human surface, allowing body configurations to be recovered from predicted spatial relationships. POSA~\cite{hassan2021posa} introduced a complementary body-centric representation that associates human mesh vertices with contact probabilities and semantic scene labels. Building on semantic control, COINS~\cite{zhao2022coins} learned compositional interactions specified through action-object pairs, including combinations of atomic interactions absent from its training examples.

Related methods extend scene-conditioned synthesis to motion. HUMANISE~\cite{wang2022humanise} introduced language-annotated interactions by aligning captured motion with reconstructed scenes and learned to generate motion conditioned on both language and geometry. SceneDiffuser~\cite{huang2023scenediffuser} integrated scene-conditioned generation, physical objectives, and goal-directed planning within diffusion-based sampling. TeSMo~\cite{yi2024tesmo} adapted a pretrained text-to-motion model through scene-aware fine-tuning to support navigation and interaction in different environments. These approaches demonstrate the value of learned interaction priors, although their coverage depends on the scenes, actions, and body configurations represented during training. Our work addresses static interaction synthesis using pretrained components without additional training on paired 3D human-scene interactions.

\subsection{Contact estimation and scene-constrained reconstruction}

Contact provides geometric evidence for resolving ambiguities in human reconstruction. PROX~\cite{hassan2019prox} incorporated scene contact and interpenetration constraints into monocular body fitting, encouraging plausible relationships between the estimated human and the surrounding surfaces. Subsequent work developed richer contact supervision and prediction. RICH~\cite{huang2022rich} supplied dense body contact annotations and introduced BSTRO to infer vertex-level contact from a single image. DECO~\cite{tripathi2023deco} extended dense contact estimation to unconstrained images through body-part and scene-context reasoning. PhySIC~\cite{muralidhar2025physic} combines pretrained human and scene estimates with DECO contact initialization, then refines the reconstruction and vertex-level contact through geometric optimization.

Contact has also been studied beyond labels on the human surface. HOT~\cite{chen2023hot} detects image regions where human-object contact occurs and associates them with the participating body parts. PICO~\cite{cseke2025pico} establishes dense correspondences on both body and object meshes, retrieving object geometry and contact information to guide joint reconstruction from natural images. These methods demonstrate the importance of identifying both sides of a contact relationship. Our setting requires inferring contact targets for a requested interaction in an existing reconstructed scene. We localize these targets as body-part-specific scene regions and check their placement before using them to constrain body fitting.

\subsection{Foundation-model-guided interaction synthesis}

Pretrained image and language models provide interaction priors without requiring a task-specific 3D generative model. GenZI~\cite{li2024genzi} synthesizes human hypotheses in multiple rendered scene views and optimizes a consistent 3D body against the resulting image evidence. Its geometric objective encourages contact and discourages penetration, while the intended body-part placement is inferred implicitly from the generated hypotheses. GenHSI~\cite{li2026genhsi} uses image generation and vision-language reasoning to construct contact-aware 3D interaction keyframes, which subsequently guide the generation of longer human-scene interaction videos. FunHSI~\cite{liu2026funhsi} explicitly identifies functional and supporting scene elements, associates them with body parts through a contact graph, and refines an estimated body against their geometry. It also uses a generator-critic process to assess generated human images and correct inconsistencies before body fitting.

Related human-object methods use pretrained models to infer affordances and contact structure. InteractAnything~\cite{zhang2025interactanything} combines language-model guidance and feedback with diffusion-based affordance parsing to synthesize interactions with open-set objects. HOI-PAGE~\cite{li2025hoipage} represents part-level interaction relationships through an affordance graph, using generated reference videos and contact constraints to optimize human-object motion. Together, these approaches establish image generation, explicit contact reasoning, and iterative feedback as useful tools for interaction synthesis.

The distinction relevant to our work is the spatial specificity of the contact target and the stage at which it is verified. A predicted contacting body region still requires a corresponding scene location; similarly, identifying a functional element can leave several admissible contact locations within an extended surface. This motivates our focus on localizing the particular scene regions suggested by a reference interaction. Agentic Contact applies vision-language feedback directly to these body-part-specific regions, allowing their placement and extent to be corrected before projection onto the reconstructed mesh. The resulting targets connect the visual interaction hypothesis to the geometric constraints used for static body fitting.

%-------------------------------------------------------------------------
% Queue the overview early so the full-width float appears on the next page.
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/methodology.jpg}
  \caption{Overview of method. Given a language instruction, a calibrated scene view, and a reconstructed mesh, we generate a Scene Interaction Graph and an enhanced interaction prompt. A vision-language model selects one of five generated human images to guide contact localization and SMPL-X initialization. The estimated contact regions are mapped onto the scene mesh and used to refine the body through scene-aware optimization. Two views show the resulting interaction, with corresponding contacts highlighted in matching colors.}
  \label{fig:method-overview}
\end{figure*}

\section{Method}
\label{sec:method}

\subsection{Problem formulation and overview}
\label{sec:method-overview}

Given a language instruction $y$, a calibrated RGB scene view $I$, and a reconstructed scene mesh $M_s$, our goal is to synthesize a static human body that performs the requested interaction with appropriate scene contact. The camera calibration comprises intrinsics $K$ and world-to-camera extrinsics $(R_{cw},\mathbf{t}_{cw})$. The scene remains fixed throughout synthesis; instructions involving object manipulation are represented by a static body configuration interacting with the object in its reconstructed state.

We represent the human using SMPL-X~\cite{pavlakos2019smplx}, with parameters
\begin{equation}
\theta=(\mathbf{t},R,\mathbf{p},\boldsymbol{\beta},s),
\end{equation}
where $\mathbf{t}$ and $R$ specify global translation and orientation, $\mathbf{p}$ is the articulated body pose, $\boldsymbol{\beta}$ describes body shape, and $s$ is an isotropic scale. The resulting body vertices are denoted by $V(\theta)$.

Figure~\ref{fig:method-overview} illustrates the method. We first derive a coarse interaction specification and generate an image of a human performing the instruction. This generated image supplies evidence for both body initialization and scene contact localization. An image generation model proposes body-part-specific contact regions, which a vision-language model evaluates and refines through targeted feedback. The resulting regions are grounded on the reconstructed mesh and used to constrain body optimization.

We express both the reconstructed scene and the monocular body estimate in the calibrated camera frame during optimization, then transform the fitted body back to scene coordinates.

\subsection{Interaction specification and human initialization}
\label{sec:method-initialization}

A vision-language model interprets the instruction and scene view to produce a Scene Interaction Graph (SIG),
\begin{equation}
G=(\mathcal{H},\mathcal{S},\mathcal{E}),
\end{equation}
where $\mathcal{H}$ contains human body-part nodes, $\mathcal{S}$ contains the target object and, when required, the floor, and $\mathcal{E}$ specifies their contact relationships. Each edge $e\in\mathcal{E}$ associates a body part with a scene element. Human nodes follow a fixed vocabulary covering the hands, arms, feet, legs, hips, back, and head. Each node maps to a predefined SMPL-X vertex subset $B_e$, such as palm-side vertices for a hand or sole-facing vertices for a foot.

The graph specifies which body parts participate in the interaction while leaving their precise contact locations unresolved. For example, an interaction with a treadmill may require both hands and one foot to contact the object. The corresponding handle and belt regions are localized later using the generated interaction as visual evidence.

Alongside the graph, the vision-language model produces a detailed enhanced prompt describing the intended pose and body-part contact relationships. Conditioned on this prompt and the original scene view, an image generation model produces five candidate human images. A vision-language model compares the candidates against the enhanced prompt and the original view, selecting the frame that best satisfies the requested interaction and contact relationships while preserving the scene and maintaining a plausible human configuration. The selected frame, denoted by $\hat I$, guides both body initialization and scene contact localization.

We recover an initial body from $\hat I$ using GVHMR~\cite{shen2024gvhmr}, supplying the calibrated focal length and retaining its camera-relative estimate. For the static input, we repeat the generated image to form a short constant sequence and take the first recovered body. This yields initial parameters $\theta_0$, with $s_0=1$. The same generated image guides contact localization, so the initial pose and scene targets come from a shared visual hypothesis.

\subsection{Agentic scene contact localization}
\label{sec:method-contact}

Figure~\ref{fig:agentic-contact} illustrates one proposal-and-correction sequence.

\begin{figure*}[t]
  \centering
  \input{figures/agentic_contact_labels.tex}
  \caption{Agentic contact estimation. The generated human image guides body-part-specific contact proposals on the original scene view. The evaluator detects a missing left-hand contact, requests its addition, and checks the revised masks. Acceptance means the visual check passed; it does not certify geometric correctness.}
  \label{fig:agentic-contact}
\end{figure*}

For each contact edge, we assign a distinct color to the participating body part. The generated interaction image $\hat I$ suggests where that part touches the scene, while the original calibrated view $I$ supplies the scene coordinates in which a contact mask must be drawn. Each mask marks a proposed contact footprint on a scene surface. The same proposal, assessment, and refinement procedure handles all contact edges, including floor support.

\emph{Propose and composite.} An image generation model receives $\hat I$, $I$, and the graph and paints one colored region per required contact. We extract the regions by their assigned colors and composite the masks onto the untouched scene view. This preserves the calibrated canvas for inspection and projection, but it does not itself correct a misplaced mask.

\emph{Check and correct.} A vision-language model compares the composite with $\hat I$ and $I$. It checks for missing or extra contacts, body-part assignment, surface location, and region size. When it rejects a proposal, its feedback asks the image generation model for specific changes, such as adding a missing hand contact or moving one onto the correct rail, while retaining regions judged satisfactory. The revised masks are extracted, composited, and checked again.

The loop stops when the evaluator accepts the masks or the round limit is reached. We denote the accepted mask for edge $e$ by $\Omega_e$. Acceptance is an automated judgment rather than proof of correct placement. A run that reaches the limit without acceptance is recorded as a failure.

\subsection{Grounding contact regions in 3D}
\label{sec:method-grounding}

The calibrated camera connects the image-space contact masks to the reconstructed scene. For a scene point $\mathbf{x}=(X,Y,Z)^\top$ expressed in camera coordinates, let $f_x,f_y$ be the focal lengths and $(c_x,c_y)$ the principal point of the zero-skew intrinsics $K$. Its image location is
\begin{equation}
\pi_K(\mathbf{x})=\left(f_x\frac{X}{Z}+c_x,\;f_y\frac{Y}{Z}+c_y\right).
\end{equation}
We rasterize the scene to identify the visible mesh faces covered by each mask. Visibility handling prevents surfaces behind the observed scene surface from being selected. We also filter disconnected surface components to reduce the inclusion of unrelated geometry.

For each contact edge $e$, sampling the retained surface produces a scene target set $P_e$. This set is associated with the predefined body vertex subset $B_e$, establishing an explicit body-part-to-scene-region constraint. The target is a surface region within which the body can establish contact; it does not prescribe a fixed correspondence for every body vertex. All target sets remain fixed during optimization.

% Sources: 3DHSI/06_Evaluate_Interaction/ablation_metrics.csv (geometry, CLIP, and diversity);
% output_init/vlm.csv, output_round1/vlm.csv, and output/vlm.csv (VLM).
% GVHMR entropy: author-provided recalculation, 2.510 (not reflected in the CSV).
% Baselines: 3DHSI/evaluation_comparison/average_metrics.csv.
% PROX CLIP: author-provided recalculation, 0.2710 (not reflected in the CSV).
\begin{table*}[t]
  \centering
  \caption{Quantitative results over 23 interactions. GVHMR, single-shot contacts, the full method, PROX, and PhySIC use the same generated human image; GenZI may synthesize another pose, so its distance to image-conditioned reference contacts is omitted (dash). Contact distance is in centimetres and conditional penetration depth in millimetres. VLM criteria range from 1 to 5, with Overall their mean. Diversity across different interactions is descriptive. Bold indicates the highest values and lowest distance and penetration values; ties are bolded.}
  \label{tab:quantitative}
  \small
  \setlength{\tabcolsep}{3pt}
  \renewcommand{\arraystretch}{1.1}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrrr@{}}
    \toprule
    & \multicolumn{2}{c}{Geometry} & Semantics & \multicolumn{5}{c}{VLM evaluation (1 to 5)} & \multicolumn{2}{c}{Diversity} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-9}\cmidrule(l){10-11}
    Setting & Contact$\downarrow$ & Pen.$\downarrow$ & CLIP$\uparrow$ & Obj.$\uparrow$ & Act.$\uparrow$ & Cont.$\uparrow$ & Phys.$\uparrow$ & Overall$\uparrow$ & Entropy & Spread \\
    & (cm) & (mm) & & & & & & & & \\
    \midrule
    PROX~\cite{hassan2019prox} & 15.68 & 13.77 & 0.2710 & 4.52 & 3.57 & 3.04 & 3.35 & 3.62 & 2.427 & 2.574 \\
    GenZI~\cite{li2024genzi} & -- & 17.95 & 0.2592 & 1.91 & 1.22 & 1.04 & 1.61 & 1.45 & 2.487 & 2.411 \\
    PhySIC~\cite{muralidhar2025physic} & 16.56 & 13.97 & 0.2704 & 4.61 & 3.57 & 3.09 & 3.30 & 3.64 & 2.533 & 2.534 \\
    \midrule
    GVHMR initialization & 16.30 & 17.11 & 0.2704 & 4.22 & 2.96 & 2.35 & 3.13 & 3.16 & 2.510 & 2.651 \\
    Single-shot contacts & 6.98 & 5.26 & 0.2712 & 4.70 & 3.61 & 3.04 & 3.43 & 3.70 & \textbf{2.548} & 2.575 \\
    Full method & \textbf{3.87} & \textbf{5.11} & \textbf{0.2725} & \textbf{5.00} & \textbf{3.91} & \textbf{3.52} & \textbf{3.87} & \textbf{4.08} & \textbf{2.548} & \textbf{2.681} \\
    \bottomrule
  \end{tabular*}
\end{table*}


\subsection{Scene-aware body optimization}
\label{sec:method-optimization}

Starting from $\theta_0$, we optimize translation, global orientation, articulated pose, and scale while holding body shape $\boldsymbol{\beta}_0$ fixed. The objective combines contact and scene-penetration penalties with regularization:
\begin{equation}
\label{eq:method-total-loss}
\begin{aligned}
\mathcal{L}(\theta)={}&\lambda_c\mathcal{L}_{\mathrm{contact}}+\lambda_p\mathcal{L}_{\mathrm{pen}}+\lambda_{\mathrm{pose}}\mathcal{L}_{\mathrm{pose}}\\
&+\lambda_{\mathrm{orient}}\mathcal{L}_{\mathrm{orient}}+\lambda_h\mathcal{L}_{\mathrm{height}}+\lambda_s\mathcal{L}_{\mathrm{self}}.
\end{aligned}
\end{equation}

The contact term measures the mean squared distance from each body contact region to its associated scene target:
\begin{equation}
\label{eq:method-contact-loss}
\mathcal{L}_{\mathrm{contact}}=\frac{1}{|\mathcal{E}|}\sum_{e\in\mathcal{E}}\frac{1}{|B_e|}\sum_{i\in B_e}\min_{\mathbf{q}\in P_e}\left\|V_i(\theta)-\mathbf{q}\right\|_2^2.
\end{equation}
The distance is directed from body to scene. This encourages the body contact region to meet its target without requiring the body to cover the entire target surface.

To discourage scene penetration, we evaluate sampled scene points using the differentiable body signed-distance field provided by VolumetricSMPL~\cite{mihajlovic2025volumetricsmpl}. Let $\phi_\theta(\mathbf{x})$ be negative inside the body, and let $\delta\geq0$ denote a clearance margin. For the scene sample set $\mathcal{Q}_s$, define the active set
\begin{equation}
\mathcal{A}_\theta=\{\mathbf{x}\in\mathcal{Q}_s\mid\phi_\theta(\mathbf{x})<\delta\}.
\end{equation}
The penetration penalty averages only over active violations:
\begin{equation}
\label{eq:method-penetration-loss}
\mathcal{L}_{\mathrm{pen}}=\begin{cases}
\displaystyle\frac{1}{|\mathcal{A}_\theta|}\sum_{\mathbf{x}\in\mathcal{A}_\theta}\left(\delta-\phi_\theta(\mathbf{x})\right),&|\mathcal{A}_\theta|>0,\\[6pt]
0,&\text{otherwise}.
\end{cases}
\end{equation}
Consequently, non-violating scene samples do not dilute the penalty.

Regularization limits departures from the initial human configuration. We penalize the mean squared change in articulated pose and the squared angular displacement from the initial global orientation. A height prior constrains the stature of the scaled, fixed-shape body:
\begin{equation}
\mathcal{L}_{\mathrm{height}}=\left(\frac{s\,h_{\mathrm{can}}-h_{\mathrm{tgt}}}{\sigma_h}\right)^2,
\end{equation}
where $h_{\mathrm{can}}$ is its canonical height at unit scale, $h_{\mathrm{tgt}}$ is the target stature, and $\sigma_h$ controls the tolerated deviation. A bounded scale parameterization further restricts size changes. Finally, a self-intersection penalty discourages collisions between body triangles~\cite{tzionas2016capturing}.

We minimize the objective with Adam in two stages. The first updates translation and scale while preserving the initial orientation and articulation. The second also optimizes orientation and body pose, allowing local adjustments to satisfy the contact constraints. The scene geometry, body shape, and target regions remain fixed throughout. The optimized body is transformed back to scene coordinates for output and rendering.

%-------------------------------------------------------------------------
\section{Experiments and Results}
\label{sec:experiments}

We first test whether agentic refinement improves 2D contact masks and whether those changes improve the fitted body's 3D contact. We then compare the complete method with reference-conditioned reconstruction baselines and with an instruction-conditioned synthesis method. The evaluation uses geometric measurements and automated assessments of rendered interactions.

\subsection{Experimental setup}
\label{sec:exp-setup}

\paragraph{Scenes and interactions.}
We evaluate on 23 interaction instructions across 15 reconstructed indoor scenes from ScanNet++~\cite{yeshwanth2023scannetpp}. Each interaction is specified by a language instruction, a calibrated RGB view, and the corresponding scene mesh. The instructions cover support interactions, such as sitting on furniture; functional interactions, such as operating a faucet or reaching for a lamp; and configurations requiring several simultaneous contacts, such as climbing a ladder or running on a treadmill. The geometric evaluation includes 82 body-part-to-scene contact relationships.

\paragraph{Reference contact annotations.}
We manually annotate the scene regions that participating body parts should contact. Annotations are drawn on the calibrated scene view, using the generated interaction image to identify the intended contact configuration. The regions include a small margin around the apparent contact footprint to accommodate uncertainty in its extent. They are created independently of the predicted contact masks and used only for evaluation.

These annotations assess fidelity to the reference interaction. Since a language instruction can admit several valid poses, a different but plausible contact configuration may receive a larger localization error. We therefore interpret contact distance alongside semantic and rendered-interaction assessments.

\paragraph{Implementation.}
We access all image generation and vision-language models through APIs. Gemini 3.8 Flash generates the Scene Interaction Graph and enhanced prompt. Nano Banana 2 generates five candidate human images per interaction, and Gemini 3.8 Flash selects the reference image according to the enhanced prompt and original scene view. The selected image also supplies the GVHMR initialization. In the contact loop, GPT Image 2 generates the contact masks and GPT-5.6 assesses them and provides corrective feedback. Contact refinement runs for at most ten rounds. The procedure completed without failure on all 23 evaluated interactions. Gemini 3.8 Flash evaluates the final rendered interactions.

Body optimization uses Adam with a learning rate of $10^{-3}$ for 2,000 iterations. The first 400 iterations optimize translation and scale; the remaining 1,600 also update orientation and body pose. The contact, penetration, pose, orientation, height, and self-intersection weights are $800$, $60$, $250$, $100$, $1$, and $10^{-3}$, respectively. The self-intersection term is disabled during the first stage, and the penetration clearance margin is zero. Each scene contact target contains 2,048 surface samples. Body shape remains fixed throughout optimization.

\paragraph{Comparison protocols.}
PROX~\cite{hassan2019prox} and PhySIC~\cite{muralidhar2025physic} use the same generated interaction image $\hat I$ as our method. Manual contact annotations are defined for this depicted interaction, so contact distance compares how accurately these methods fit that shared reference. PhySIC additionally receives the original scene image, calibrated camera parameters, and depth rendered from the reference mesh, with scene scale fixed during optimization. PROX uses a scene TSDF constructed from mesh-rendered depth maps and registered ScanNet++ camera poses.

GenZI~\cite{li2024genzi} instead runs its synthesis pipeline: given the instruction and prepared scene, it generates its own human hypotheses and may produce a different valid realization of the interaction. We provide a scene TSDF built from registered views and a manually specified target-object anchor. Since the manual contact annotations refer to $\hat I$, we do not score GenZI's distance to those reference contacts. We retain reference-independent scene-penetration, CLIP, and rendered-interaction assessments, and interpret this as a comparison of complete synthesis outputs rather than of fitting to the same depicted pose.

We use the authors' implementations for all baselines. Apart from the input adaptations described above, we retain their original configurations and run each method to completion without imposing an additional computation budget.

\subsection{Evaluation metrics}
\label{sec:exp-metrics}

\paragraph{Image-space contact localization.}
We compare the first-round contact masks with the final masks after agentic refinement, using the same generated interaction image and manual annotations. For predicted mask $\Omega^{\mathrm{pred}}$ and reference region $\Omega^{\mathrm{gt}}$, containment is defined as
\begin{equation}
C=\frac{|\Omega^{\mathrm{pred}}\cap\Omega^{\mathrm{gt}}|}{|\Omega^{\mathrm{pred}}|}.
\end{equation}
Containment measures the fraction of a prediction lying within the annotated region. It accommodates the annotation margin without requiring the prediction to fill the entire reference mask. However, containment alone can reward an excessively small prediction, so we also report centroid distance and the number of missing contacts.

Centroid distance measures the Euclidean separation between predicted and reference mask centroids in pixels. In the image-space results reported here, localization averages describe nonempty predictions, while omitted contacts are reported separately as misses. For bilateral contacts, the evaluation permits a left/right reassignment when it reduces the summed centroid distance by more than $2\%$ of the image diagonal. The same rule is applied to both settings.

\paragraph{Geometric contact accuracy.}
Manual contact masks are projected onto the reconstructed scene to obtain reference target regions for our method, its ablations, PROX, and PhySIC. GenZI may depict a different valid pose, so this reference-conditioned measure is not reported for it. For contact edge $e$, we measure the directed mean distance from its body contact vertices $B_e$ to the sampled reference surface $S_e$:
\begin{equation}
D_e=\frac{1}{|B_e|}\sum_{i\in B_e}\min_{\mathbf{q}\in S_e}\|V_i-\mathbf{q}\|_2.
\end{equation}
Distances are averaged first over contact edges within an interaction and then over interactions. Every interaction therefore receives equal weight. We report contact distance in centimetres.

\paragraph{Scene penetration.}
We sample the scene surface within the evaluation crop and query the signed-distance field of the fitted body. Mean penetration is the average magnitude of negative signed distances, computed only over penetrating samples and set to zero when none penetrate. It measures collision severity rather than collision volume.

\paragraph{Diversity.}
For descriptive context, we cluster SMPL-X parameter vectors across the 23 \emph{different} instructions and scenes. After standardizing translation, orientation, pose, and shape features, we apply $k$-means with $k=15$ and report cluster-occupancy entropy and within-cluster spread. These set-level statistics do not measure the diversity of outputs for one instruction or the correctness of a pose, so we do not treat a higher value as a quality ranking.

\paragraph{Rendered-interaction quality.}
We evaluate ten rendered views around each placed human using CLIP~\cite{radford2021clip} and a vision-language rubric. All methods use the same render cameras for each interaction. CLIP measures cosine similarity between normalized image and instruction embeddings, averaged across views and then interactions. Gemini 3.8 Flash receives the instruction and the ten rendered views jointly, assigning scores from 1 to 5 for target-object correctness, action correctness, contact and spatial relationships, and physical plausibility. The overall score is the arithmetic mean of these four criteria.

This assessment is separate from contact-mask verification: it evaluates the final rendered body-scene configuration rather than intermediate contact proposals. The VLM scores provide an automated perceptual assessment and complement the geometric measurements.

\subsection{Contact localization results}
\label{sec:exp-contact-localization}

Table~\ref{tab:contact-localization} compares single-shot contact estimation with agentic refinement before any 3D body optimization.

\begin{table}[t]
  \centering
  \caption{Image-space contact localization against manual reference regions. Localization averages use nonempty predictions; missing contacts are counted separately.}
  \label{tab:contact-localization}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lccc@{}}
    \toprule
    Setting & Containment$\uparrow$ & Centroid$\downarrow$ & Misses$\downarrow$ \\
    & & (px) & \\
    \midrule
    Single-shot & 0.60 & 27.6 & 3 \\
    Agentic & \textbf{0.74} & \textbf{19.5} & \textbf{0} \\
    \bottomrule
  \end{tabular}
\end{table}

Agentic refinement improves both localization measures and removes the three omissions observed in the single-shot predictions. Mean containment increases from 0.60 to 0.74, while centroid distance decreases from 27.6 to 19.5 pixels. The results indicate that refinement improves the placement of predicted regions as well as their completeness.

The image-space gains matter because the selected masks become 3D fitting targets. A small displacement near a thin rail can select a different mesh surface when projected. The next subsection tests whether the improved mask localization carries through to final body contact; Figure~\ref{fig:agentic-contact} illustrates one missing-contact correction.

\subsection{Effect on 3D body fitting}
\label{sec:exp-ablation}

The 2D mask results above establish that refinement changes the contact targets. We now test whether those changes improve the final 3D body. Table~\ref{tab:quantitative} includes three settings that separate geometric fitting from contact refinement. \emph{GVHMR initialization} uses the recovered body without scene-aware optimization. \emph{Single-shot contacts} applies the optimizer using first-round contact estimates. \emph{Full method} uses the refined contact regions. The latter two settings share the human initialization and optimization configuration.

\paragraph{Effect of scene-aware optimization.}
Using first-round contacts, fitting reduces mean reference-contact distance from 16.30 to 6.98~cm and mean penetration from 17.11 to 5.26~mm relative to the GVHMR initialization. The VLM contact score rises from 2.35 to 3.04. This comparison shows what scene-aware fitting contributes before the contact targets are refined.

\paragraph{Effect of agentic refinement.}
With the same human initialization and optimizer, replacing first-round masks with refined masks further lowers mean contact distance from 6.98 to 3.87~cm. Mean penetration remains similar (5.26 versus 5.11~mm), while the VLM contact score rises from 3.04 to 3.52. Together with the 2D containment increase and elimination of three missed masks in Table~\ref{tab:contact-localization}, these results show that the complete refinement procedure improves contact localization and downstream body fitting.

The other VLM criteria also rise, giving overall scores of 3.16, 3.70, and 4.08 for the initialization, single-shot, and full settings. CLIP and the descriptive across-interaction diversity statistics change little; neither explains the main contact-distance gain.

\subsection{Comparison with existing methods}
\label{sec:exp-comparison}

Table~\ref{tab:quantitative} also compares the full method with PROX, PhySIC, and GenZI. GenZI's reference-contact distance is omitted because its generated pose need not match the shared image; its VLM contact score assesses visible body-scene relationships in its own rendered output.

For the shared generated interaction, the full method has a mean contact distance of 3.87~cm, versus 15.68~cm for PROX and 16.56~cm for PhySIC. It also has lower mean penetration. PhySIC receives reference scene geometry, so the contact comparison is less affected by differences in scene reconstruction. These measurements assess fitting to the depicted contact configuration, not the entire instruction-conditioned synthesis problem.

Across all rendered outputs, the full method receives the highest VLM overall score, 4.08, compared with 3.62 for PROX, 3.64 for PhySIC, and 1.45 for GenZI. Its contact and physical-plausibility scores agree in direction with the geometric measures for the shared-reference methods.

On the ScanNet++ scene renders, we observed poor human inpainting from GenZI's Stable Diffusion 2 backbone. Artifacts and limited photorealism in these reconstructed views make human insertion challenging. The resulting image hypotheses provide unreliable guidance for body fitting and likely contribute to GenZI's low object and action scores. This comparison therefore reflects differences in image generation as well as downstream body fitting.

CLIP scores differ little among the full method, PROX, and PhySIC (0.2725, 0.2710, and 0.2704). Broad image-text agreement therefore does not capture their larger differences in reference-contact accuracy. The diversity statistics describe variation across the tested interactions, not the number of valid alternatives for one instruction.

\bibliographystyle{eg-alpha-doi}
\bibliography{references}

\end{document}
