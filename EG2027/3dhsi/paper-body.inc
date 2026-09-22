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
    our method generates a human reference, verifies body-part-specific
    scene-side contact regions, and fits a scene-aligned 3D human. The bottom
    row shows synthesized interactions across different scenes.}
  \label{fig:teaser}
}

\maketitle
%-------------------------------------------------------------------------
\begin{abstract}
Synthesizing humans in reconstructed 3D scenes requires accurate contact between each body part and the surface it should touch. Foundation models can generate plausible images of an interaction, but recovering these correspondences in metric 3D remains difficult. We present Agentic Contact, a zero-shot method that synthesizes a static SMPL-X body in accurate contact with the scene from a language instruction, a calibrated RGB view, and a reconstructed scene mesh, without training on paired 3D human-scene interactions. Our method explicitly represents contact as body-part-specific regions on the scene. An image model proposes these regions using a generated human image as reference, and a vision-language model checks their placement and requests targeted corrections. The resulting regions are projected onto the mesh as 3D contact targets. A scene-aware optimization refines a monocular body estimate to satisfy these targets while penalizing penetration. Across 23 interactions in 15 ScanNet++ scenes, verification improves contact accuracy compared with single-shot contact estimation. The full method achieves better contact accuracy and higher overall scores in a vision-language evaluation of rendered interactions than PhySIC, PROX, and GenZI.

%-------------------------------------------------------------------------
\begin{CCSXML}
<ccs2012>
<concept>
<concept_id>10010147.10010371.10010352</concept_id>
<concept_desc>Computing methodologies~Animation</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
\end{CCSXML}

\ccsdesc[500]{Computing methodologies~Animation}

\printccsdesc   
\end{abstract}  
%-------------------------------------------------------------------------
\section{Introduction}

Reconstructed 3D scenes provide detailed models of real environments, but populating them with humans requires understanding how the body interacts with the surrounding geometry. A person sitting on a table must be supported by its surface; a person using a treadmill must place the supporting foot on the belt and the hands on the appropriate rails. Such relationships determine whether a synthesized interaction is convincing beyond a single viewpoint. Producing these configurations from language instructions would simplify character placement in games, virtual environments, and digital content creation. The challenge is to translate the requested action into a plausible body pose with accurate contact on the scene.

Learning-based methods have made substantial progress in human-scene interaction synthesis. PLACE~\cite{zhang2020place} learns body-scene proximity, while POSA~\cite{hassan2021posa} models contact probabilities and scene semantics on the human body. COINS~\cite{zhao2022coins} introduces compositional control through action-object specifications. In the dynamic setting, HUMANISE~\cite{wang2022humanise} and TeSMo~\cite{yi2024tesmo} generate scene-aware human motion conditioned on language. These methods learn useful relationships between human configurations, scene geometry, and interaction intent. Their learned priors, however, reflect the coverage of their training data, making transfer to unfamiliar objects, environments, and actions a continuing challenge.

Foundation models offer broader visual and semantic priors acquired from large image and language collections. Recent interaction methods use these pretrained capabilities to reduce dependence on task-specific 3D interaction data. GenZI~\cite{li2024genzi} synthesizes interactions by generating human hypotheses in scene images and fitting a consistent 3D body. PhySIC~\cite{muralidhar2025physic} combines pretrained human, scene, and contact estimates with geometric optimization to reconstruct interactions from a single image. FunHSI~\cite{liu2026funhsi} uses functionality reasoning and generated human images to synthesize task-directed interactions. Related human-object methods, including InteractAnything~\cite{zhang2025interactanything} and HOI-PAGE~\cite{li2025hoipage}, further demonstrate how pretrained models can supply affordance and contact information. These approaches expand the range of interactions that can be specified, while leaving the geometric grounding of their predictions to subsequent fitting.

Contact is central to that grounding. PROX~\cite{hassan2019prox} established the value of scene contact and interpenetration constraints for resolving ambiguities in human pose estimation. Subsequent approaches differ in how they determine the contact relationships. PhySIC~\cite{muralidhar2025physic} uses DECO~\cite{tripathi2023deco} to predict contacting human vertices and refines their relationship to nearby scene geometry during optimization. GenZI~\cite{li2024genzi} obtains body-part placement implicitly from generated image hypotheses, together with geometric constraints encouraging contact and discouraging penetration. FunHSI~\cite{liu2026funhsi} explicitly associates body parts with functional scene elements and optimizes contact with those elements. These representations provide different forms of guidance, but do not necessarily specify the particular scene patch required by a depicted interaction. A body contact label leaves its intended scene correspondence unspecified, while an extended functional surface can admit several possible contact locations.

We focus on explicitly recovering these scene-side correspondences. Given an interaction hypothesis, our goal is to identify the surface region that each participating body part should contact and verify its placement before fitting the body. This distinction matters because an incorrect target can draw an otherwise plausible pose toward the wrong location. A single contact prediction may omit a required region, place it on an adjacent surface, or assign it an inappropriate extent. Making these regions explicit allows such errors to be inspected and corrected before they become geometric constraints.

We present \textbf{Agentic Contact}, a zero-shot method that synthesizes a static SMPL-X~\cite{pavlakos2019smplx} body from a language instruction, a calibrated RGB view, and a reconstructed scene mesh. A coarse interaction graph specifies the participating body parts and scene objects, and a generated human image provides a visual hypothesis of the action. An image model then proposes body-part-specific scene contact regions. A vision-language model checks their placement against the reference interaction and the original scene, providing targeted feedback for refinement. The resulting regions are projected onto the reconstructed mesh as 3D targets. A scene-aware optimization adjusts a monocular body estimate to satisfy these targets while penalizing penetration and retaining a plausible pose. The method uses pretrained models without task-specific training or fine-tuning on paired 3D human-scene interactions.

We evaluate the method on 23 interactions in 15 ScanNet++~\cite{yeshwanth2023scannetpp} scenes. Ablations show that verifying and correcting contact regions improves localization and final 3D contact accuracy compared with single-shot estimation. The full method achieves better contact accuracy and higher overall scores in a vision-language evaluation of rendered interactions than PROX, PhySIC, and GenZI. The PhySIC comparison includes reference scene geometry to reduce differences arising from scene reconstruction.

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
  \caption{Overview of method. Given a language instruction, a calibrated scene view, and a reconstructed mesh, we generate a Scene Interaction Graph and an enhanced interaction prompt. A vision-language model selects one of five generated human frames to guide contact localization and SMPL-X initialization. The estimated contact regions are mapped onto the scene mesh and used to refine the body through scene-aware optimization. Two views show the resulting interaction, with corresponding contacts highlighted in matching colors.}
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

Figure~\ref{fig:method-overview} illustrates the method. We first derive a coarse interaction specification and generate an image of a human performing the instruction. This reference image supplies evidence for both body initialization and scene contact localization. An image model proposes body-part-specific contact regions, which a vision-language model evaluates and refines through targeted feedback. The resulting regions are grounded on the reconstructed mesh and used to constrain body optimization.

We perform geometric processing in the calibrated camera frame. Scene points are transformed as
\begin{equation}
\mathbf{x}_c=R_{cw}\mathbf{x}_w+\mathbf{t}_{cw},
\end{equation}
placing the scene and the camera-relative human estimate in a common metric coordinate system. After optimization, the body is transformed back into scene coordinates.

\subsection{Interaction specification and human initialization}
\label{sec:method-initialization}

A vision-language model interprets the instruction and scene view to produce a Scene Interaction Graph,
\begin{equation}
G=(\mathcal{H},\mathcal{S},\mathcal{E}),
\end{equation}
where $\mathcal{H}$ contains human body-part nodes, $\mathcal{S}$ contains the target object and, when required, the floor, and $\mathcal{E}$ specifies their contact relationships. Each edge $e\in\mathcal{E}$ associates a body part with a scene element. Human nodes follow a fixed vocabulary covering the hands, arms, feet, legs, hips, back, and head. Each node maps to a predefined SMPL-X vertex subset $B_e$, such as palm-side vertices for a hand or sole-facing vertices for a foot.

The graph specifies which body parts participate in the interaction while leaving their precise contact locations unresolved. For example, an interaction with a treadmill may require both hands and one foot to contact the object. The corresponding handle and belt regions are localized later using the generated interaction as visual evidence.

Alongside the graph, the vision-language model produces a detailed enhanced prompt describing the intended pose and body-part contact relationships. Conditioned on this prompt and the original scene view, an image model generates five candidate human frames. A vision-language model compares the candidates against the enhanced prompt and the original view, selecting the frame that best satisfies the requested interaction and contact relationships while preserving the scene and maintaining a plausible human configuration. The selected frame, denoted by $\hat I$, guides both body initialization and scene contact localization.

We recover an initial body from $\hat I$ using GVHMR~\cite{shen2024gvhmr}, supplying the calibrated focal length and retaining its camera-relative estimate. For the static input, we repeat the reference image to form a short constant sequence and take the first recovered body. This yields initial parameters $\theta_0$, with $s_0=1$. The same reference image guides contact localization, so the initial pose and scene targets are inferred from a shared visual hypothesis.

\subsection{Agentic scene contact localization}
\label{sec:method-contact}

Figure~\ref{fig:agentic-contact} illustrates the contact proposal, verification, and correction process.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/agentic_contact.jpg}
  \caption{Agentic contact estimation. Given the generated human reference, the original scene view, and the Scene Interaction Graph, an image model proposes body-part-specific contact regions. A vision-language model checks the proposed masks against the reference interaction and the original scene. In this example, it requests the missing left-hand contact while retaining the existing regions. The revised proposal is evaluated again and accepted. The refinement rounds are shown from left to right.}
  \label{fig:agentic-contact}
\end{figure*}

For each object-contact edge, we estimate a mask identifying the scene surface that the corresponding body part should touch. The generated reference $\hat I$ provides evidence about the intended contact, while the original view $I$ defines the image coordinates in which the region must be localized. A mask represents the contact footprint on the object surface, rather than the visible silhouette of the body part or the full extent of the object.

An image model receives the reference, original scene view, and contact specification, and proposes a distinct colored region for each required body-part contact. We extract the resulting masks and display them on the original view for evaluation. Let
\begin{equation}
\mathcal{Z}^{(r)}=\{Z_e^{(r)}\mid e\in\mathcal{E}_{\mathrm{obj}}\}
\end{equation}
denote the object-contact masks at refinement round $r$, where $\mathcal{E}_{\mathrm{obj}}$ contains the graph edges involving the target object.

A vision-language evaluator compares these proposals with the reference interaction and the original scene. It checks whether required contacts are present, whether additional contacts have been introduced, and whether each region has the correct body-part assignment, surface location, and spatial extent. A region is assessed against the contact depicted in the reference, rather than accepted solely because it occupies a plausible part of the object.

The evaluator returns an acceptance decision $d^{(r)}$ and a correction instruction $a^{(r)}$:
\begin{equation}
(d^{(r)},a^{(r)})=\mathcal{V}\!\left(\hat I,I,G,\mathcal{Z}^{(r)}\right).
\end{equation}
When correction is required, the image model receives the previous proposal together with the feedback:
\begin{equation}
\mathcal{Z}^{(r+1)}=\mathcal{P}\!\left(\hat I,I,G,\mathcal{Z}^{(r)},a^{(r)}\right).
\end{equation}
Here, $\mathcal{P}$ includes image-based proposal generation and mask extraction. Feedback specifies local changes, such as moving a hand region onto the appropriate rail, reducing an oversized footprint, or adding a missing contact. Regions judged correct are instructed to remain unchanged.

We retain the current masks when the evaluator accepts them or when the maximum number of refinement rounds is reached. The selected masks are denoted by $\Omega_e$. This feedback process makes contact localization an explicit intermediate estimate that can be inspected and corrected before it influences the body fit.

Floor-support contacts are treated separately. For edges involving the floor, SAM~3~\cite{carion2025sam3} segments the visible floor in the original view. The resulting support region may be shared by multiple body parts, with their individual contact relationships maintained through the graph. Their precise placement is subsequently determined by body optimization.

\subsection{Grounding contact regions in 3D}
\label{sec:method-grounding}

The calibrated camera connects the image-space contact masks to the reconstructed scene. For a scene point $\mathbf{x}$ expressed in camera coordinates, its image location is
\begin{equation}
\pi_K(\mathbf{x})=\left(\frac{(K\mathbf{x})_1}{(K\mathbf{x})_3},\frac{(K\mathbf{x})_2}{(K\mathbf{x})_3}\right).
\end{equation}
We rasterize the scene to identify the visible mesh faces covered by each mask. Visibility handling prevents surfaces behind the observed scene surface from being selected. We also filter disconnected surface components to reduce the inclusion of unrelated geometry.

For each contact edge $e$, sampling the retained surface produces a scene target set $P_e$. This set is associated with the predefined body vertex subset $B_e$, establishing an explicit body-part-to-scene-region constraint. The target is a surface region within which the body can establish contact; it does not prescribe a fixed correspondence for every body vertex. All target sets remain fixed during optimization.

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

We minimize the objective with Adam in two stages. The first updates translation and scale while preserving the initial orientation and articulation. The second also optimizes orientation and body pose, allowing local adjustments to satisfy the contact constraints. The scene geometry, body shape, and target regions remain fixed throughout. The optimized vertices are returned to scene coordinates using
\begin{equation}
V_i^w=R_{cw}^{\top}\left(V_i^c-\mathbf{t}_{cw}\right).
\end{equation}

%-------------------------------------------------------------------------
\section{Experiments and Results}
\label{sec:experiments}

We evaluate whether explicit scene contact localization improves static human-scene interaction synthesis, whether agentic verification improves the localized regions, and how the complete method compares with existing approaches. The evaluation examines contact localization before body fitting, geometric accuracy after optimization, and the quality of rendered interactions.

\subsection{Experimental setup}
\label{sec:exp-setup}

\paragraph{Scenes and interactions.}
We evaluate on 23 interaction instructions across 15 reconstructed indoor scenes from ScanNet++~\cite{yeshwanth2023scannetpp}. Each interaction is specified by a language instruction, a calibrated RGB view, and the corresponding scene mesh. The instructions cover support interactions, such as sitting on furniture; functional interactions, such as operating a faucet or reaching for a lamp; and configurations requiring several simultaneous contacts, such as climbing a ladder or running on a treadmill. The geometric evaluation includes 82 body-part-to-scene contact relationships.

\paragraph{Reference contact annotations.}
We manually annotate the scene regions that participating body parts should contact. Annotations are drawn on the calibrated scene view, using the generated human reference to identify the intended contact configuration. The regions include a small margin around the apparent contact footprint to accommodate uncertainty in its extent. They are created independently of the predicted contact masks and used only for evaluation.

These annotations assess fidelity to the reference interaction. Since a language instruction can admit several valid poses, a different but plausible contact configuration may receive a larger localization error. We therefore interpret contact distance alongside semantic and rendered-interaction assessments.

\paragraph{Implementation.}
We generate five candidate human frames per interaction and use a vision-language model to select the reference according to the enhanced interaction prompt. Contact refinement runs for at most ten rounds. The selected reference also supplies the GVHMR initialization, and SAM~3 provides floor-support segmentation when required.

Body optimization uses Adam with a learning rate of $10^{-3}$ for 2,000 iterations. The first 400 iterations optimize translation and scale; the remaining 1,600 also update orientation and body pose. The contact, penetration, pose, orientation, height, and self-intersection weights are $800$, $60$, $250$, $100$, $1$, and $10^{-3}$, respectively. The self-intersection term is disabled during the first stage, and the penetration clearance margin is zero. Each scene contact target contains 2,048 surface samples. Body shape remains fixed throughout optimization.

\paragraph{Baselines.}
We compare against PROX~\cite{hassan2019prox}, PhySIC~\cite{muralidhar2025physic}, and GenZI~\cite{li2024genzi}. PROX and PhySIC are evaluated as scene-aware reconstruction baselines using the generated human reference as image evidence.

We adapt the baselines to the calibrated ScanNet++ setting to reduce differences arising from scene reconstruction and alignment. PhySIC receives the original scene image, calibrated camera parameters, and scene depth rendered from the reference mesh, with scene scale fixed during optimization. For PROX and GenZI, we construct scene TSDFs from mesh-rendered depth maps using registered ScanNet++ camera poses. GenZI additionally receives a manually specified target-object anchor for interaction localization.

\subsection{Evaluation metrics}
\label{sec:exp-metrics}

\paragraph{Image-space contact localization.}
We compare the first-round contact masks with the final masks after agentic refinement, using the same human reference and manual annotations. For predicted mask $\Omega^{\mathrm{pred}}$ and reference region $\Omega^{\mathrm{gt}}$, containment is defined as
\begin{equation}
C=\frac{|\Omega^{\mathrm{pred}}\cap\Omega^{\mathrm{gt}}|}{|\Omega^{\mathrm{pred}}|}.
\end{equation}
Containment measures the fraction of a prediction lying within the annotated region. It accommodates the annotation margin without requiring the prediction to fill the entire reference mask. However, containment alone can reward an excessively small prediction, so we also report centroid distance and the number of missing contacts.

Centroid distance measures the Euclidean separation between predicted and reference mask centroids in pixels. In the image-space results reported here, localization averages describe nonempty predictions, while omitted contacts are reported separately as misses. For bilateral contacts, the evaluation permits a left/right reassignment when it reduces the summed centroid distance by more than $2\%$ of the image diagonal. The same rule is applied to both settings.

\paragraph{Geometric contact accuracy.}
Manual contact masks are projected onto the reconstructed scene to obtain reference target regions. For contact edge $e$, we measure the directed mean distance from its body contact vertices $B_e$ to the sampled reference surface $S_e$:
\begin{equation}
D_e=\frac{1}{|B_e|}\sum_{i\in B_e}\min_{\mathbf{q}\in S_e}\|V_i-\mathbf{q}\|_2.
\end{equation}
Distances are averaged first over contact edges within an interaction and then over interactions. Every interaction therefore receives equal weight. We report contact distance in centimetres.

\paragraph{Scene penetration.}
We sample the scene surface within the evaluation crop and query the signed-distance field of the fitted body. Mean penetration is the average magnitude of negative signed distances, computed only over penetrating samples and set to zero when none penetrate. It measures collision severity rather than collision volume.

\paragraph{Diversity.}
We measure variation across the 23 synthesized interactions by clustering SMPL-X parameter vectors containing translation, global orientation, body pose, and shape. For each setting, feature dimensions are standardized across the evaluation set before $k$-means clustering with $k=15$. We report cluster-occupancy entropy, $H=-\sum_j p_j\log p_j$, where $p_j$ is the fraction of interactions assigned to cluster $j$, and cluster spread, the mean Euclidean distance from each standardized vector to its assigned cluster centre. Higher values indicate more even cluster occupancy and greater within-cluster variation, respectively. These set-level measures describe variation across different instructions and scenes, rather than alternative outputs for a single instruction, and are interpreted alongside interaction correctness.

\paragraph{Rendered-interaction quality.}
We evaluate multiple rendered views of each interaction using CLIP~\cite{radford2021clip} and a vision-language rubric. CLIP measures cosine similarity between normalized image and instruction embeddings, averaged across views and then interactions. The vision-language evaluator receives the instruction and rendered views jointly, assigning scores from 1 to 5 for target-object correctness, action correctness, contact and spatial relationships, and physical plausibility. The overall score is the arithmetic mean of these four criteria.

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

The qualitative examples show corrections to displaced regions, inappropriate mask extents, and missing contacts. These changes affect the geometric targets used during fitting. A mask placed beside a narrow rail may select a different scene surface when projected onto the mesh, so even a modest image-space displacement can produce a substantial error in the 3D target. The geometric evaluation below measures the resulting effect on body contact.

\subsection{Comparison with existing methods}
\label{sec:exp-comparison}

Table~\ref{tab:method-comparison} reports geometric, semantic, VLM, and diversity results for the complete method and the three baselines.

% Sources: 3DHSI/evaluation_comparison/average_metrics.csv;
% 3DHSI/06_Evaluate_Interaction/output/vlm.csv (full-method VLM).
% PROX CLIP: author-provided recalculation, 0.2710 (not reflected in the CSV).
\begin{table*}[t]
  \centering
  \caption{Comparison with existing methods over 23 interactions. Contact distance and penetration are averaged over interactions; penetration is conditioned on penetrating scene samples within each interaction. Diversity reports cluster-occupancy entropy and mean within-cluster spread in standardized parameter space. VLM criteria assess target object, action, contact, and physical plausibility; Overall is their mean. The best displayed values are bold.}
  \label{tab:method-comparison}
  \small
  \setlength{\tabcolsep}{3pt}
  \renewcommand{\arraystretch}{1.1}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrrr@{}}
    \toprule
    & \multicolumn{2}{c}{Geometry} & Semantics & \multicolumn{5}{c}{VLM evaluation (1 to 5)} & \multicolumn{2}{c}{Diversity} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-9}\cmidrule(l){10-11}
    Setting & Contact$\downarrow$ & Pen.$\downarrow$ & CLIP$\uparrow$ & Obj.$\uparrow$ & Act.$\uparrow$ & Cont.$\uparrow$ & Phys.$\uparrow$ & Overall$\uparrow$ & Entropy$\uparrow$ & Spread$\uparrow$ \\
    & (cm) & (mm) & & & & & & & & \\
    \midrule
    PROX~\cite{hassan2019prox} & 15.68 & 13.77 & 0.2710 & 4.52 & 3.57 & 3.04 & 3.35 & 3.62 & 2.427 & 2.574 \\
    PhySIC~\cite{muralidhar2025physic} & 16.56 & 13.97 & 0.2704 & 4.61 & 3.57 & 3.09 & 3.30 & 3.64 & 2.533 & 2.534 \\
    GenZI~\cite{li2024genzi} & 62.72 & 17.95 & 0.2592 & 1.91 & 1.22 & 1.04 & 1.61 & 1.45 & 2.487 & 2.411 \\
    Full method & \textbf{3.87} & \textbf{5.11} & \textbf{0.2725} & \textbf{5.00} & \textbf{3.91} & \textbf{3.52} & \textbf{3.87} & \textbf{4.08} & \textbf{2.548} & \textbf{2.681} \\
    \bottomrule
  \end{tabular*}
\end{table*}

The full method achieves the lowest mean contact distance, at 3.87~cm, compared with 15.68~cm for PROX, 16.56~cm for PhySIC, and 62.72~cm for GenZI. It also achieves the lowest mean penetration. The improvement over PhySIC remains present when the baseline receives reference scene geometry, supporting the value of explicitly localized contact targets beyond access to calibrated scene information.

The full method obtains the highest VLM score on each criterion, reaching an overall score of 4.08, compared with 3.64 for PhySIC, 3.62 for PROX, and 1.45 for GenZI. Its higher contact and physical-plausibility scores provide complementary evidence to the geometric measurements. GenZI's low object and action scores suggest that its errors in this evaluation extend beyond small contact offsets. These results characterize the evaluated configurations and interaction set.

The full method obtains the highest CLIP score, at 0.2725, compared with 0.2710 for PROX, 0.2704 for PhySIC, and 0.2592 for GenZI. The small differences among the first three methods contrast with their larger differences in contact accuracy, supporting the use of geometric measurements alongside semantic similarity.

The full method also obtains the highest diversity entropy and cluster spread among the compared methods, at 2.548 and 2.681, respectively. These measurements describe variation across the evaluated interactions; they do not establish the correctness of individual poses.

\subsection{Ablation studies}
\label{sec:exp-ablation}

We evaluate three settings to separate the effects of geometric fitting and contact refinement. \emph{GVHMR initialization} uses the recovered body without scene-aware optimization. \emph{Single-shot contacts} applies the optimizer using first-round contact estimates. \emph{Full method} uses the refined contact regions. The latter two settings share the human initialization and optimization configuration.

% Sources: 3DHSI/06_Evaluate_Interaction/ablation_metrics.csv (geometry, CLIP, and diversity);
% output_init/vlm.csv, output_round1/vlm.csv, and output/vlm.csv (VLM).
% GVHMR entropy: author-provided recalculation, 2.510 (not reflected in the CSV).
\begin{table*}[t]
  \centering
  \caption{Ablation of scene-aware optimization and agentic contact refinement over 23 interactions. Units and VLM criteria follow Table~\ref{tab:method-comparison}. The best displayed values are bold.}
  \label{tab:ablation}
  \small
  \setlength{\tabcolsep}{3pt}
  \renewcommand{\arraystretch}{1.1}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrrr@{}}
    \toprule
    & \multicolumn{2}{c}{Geometry} & Semantics & \multicolumn{5}{c}{VLM evaluation (1 to 5)} & \multicolumn{2}{c}{Diversity} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-9}\cmidrule(l){10-11}
    Setting & Contact$\downarrow$ & Pen.$\downarrow$ & CLIP$\uparrow$ & Obj.$\uparrow$ & Act.$\uparrow$ & Cont.$\uparrow$ & Phys.$\uparrow$ & Overall$\uparrow$ & Entropy$\uparrow$ & Spread$\uparrow$ \\
    & (cm) & (mm) & & & & & & & & \\
    \midrule
    GVHMR initialization & 16.30 & 17.11 & 0.2704 & 4.22 & 2.96 & 2.35 & 3.13 & 3.16 & 2.510 & 2.651 \\
    Single-shot contacts & 6.98 & 5.26 & 0.2712 & 4.70 & 3.61 & 3.04 & 3.43 & 3.70 & \textbf{2.548} & 2.575 \\
    Full method & \textbf{3.87} & \textbf{5.11} & \textbf{0.2725} & \textbf{5.00} & \textbf{3.91} & \textbf{3.52} & \textbf{3.87} & \textbf{4.08} & \textbf{2.548} & \textbf{2.681} \\
    \bottomrule
  \end{tabular*}
\end{table*}

Table~\ref{tab:ablation} combines geometric, semantic, VLM, and diversity measurements for these settings.

\paragraph{Effect of scene-aware optimization.}
Fitting with single-shot contacts reduces mean contact distance from 16.30 to 6.98~cm and mean penetration from 17.11 to 5.26~mm. A body recovered from the reference image therefore benefits substantially from explicit geometric constraints, even before contact refinement is introduced. The VLM contact score increases from 2.35 to 3.04 and the overall score from 3.16 to 3.70.

\paragraph{Effect of agentic refinement.}
Replacing first-round targets with refined targets further reduces contact distance from 6.98 to 3.87~cm. Mean penetration changes only slightly, from 5.26 to 5.11~mm. The full method also obtains the highest CLIP score. Agentic refinement improves all four VLM criteria: the contact score increases from 3.04 to 3.52, physical plausibility from 3.43 to 3.87, and the overall score from 3.70 to 4.08. Together with the image-space results, these findings support improved contact localization and rendered-interaction quality through refinement, while both fitted variants benefit from the shared collision objective.

Diversity remains similar across the ablations. Entropy increases from 2.510 for the initialization to 2.548 for both fitted settings, while cluster spread is 2.651, 2.575, and 2.681 for the initialization, single-shot contacts, and full method, respectively. The contact improvements are thus accompanied by comparable variation in the standardized body parameters.

The qualitative ablation is examined from multiple viewpoints. The initial body may appear aligned in the reference view while remaining separated from a contact surface in 3D. Optimization improves this alignment, but an incorrectly localized target can still attract a body part toward the wrong region. Correcting the target before fitting addresses this source of error.

\subsection{Limitations and failure cases}
\label{sec:exp-limitations}

Repeated structures remain challenging for contact localization. On a ladder, visually similar rungs can cause a predicted contact to be assigned to an adjacent rung. The fitted body may then form a plausible interaction at the wrong location. This failure distinguishes fidelity to the reference contact from general visual plausibility and shows that VLM acceptance does not guarantee correct localization.

The method also depends on the quality of the initial body estimate. In a lying-on-sofa example, monocular recovery fails to reproduce the pose shown in the generated reference. Because optimization regularizes the body toward its initialization, it cannot recover the intended configuration through local refinement alone.

More generally, scene targets depend on the visibility of the contacted surface and the quality of the reconstructed geometry. The method synthesizes a static configuration with fixed scene geometry and does not establish dynamic balance or simulate object manipulation. Finally, the evaluation covers a limited set of indoor interactions, and the automated VLM assessment does not replace a controlled human perceptual study.

%-------------------------------------------------------------------------
\section{Conclusion and Future Work}
\label{sec:conclusion}

We presented Agentic Contact, a zero-shot method for synthesizing static human-scene interactions from a language instruction, a calibrated RGB view, and a reconstructed scene mesh, without training on paired 3D interaction data. The method explicitly localizes scene regions for participating body parts, refines these regions through agentic verification, and grounds them in 3D to guide SMPL-X body optimization. Across the evaluated interactions, the full method achieves more accurate contact and higher VLM ratings of rendered interactions than the compared baselines. The ablations further support the contributions of scene-aware optimization and contact refinement. These results demonstrate the value of explicitly identifying where each body part should contact the scene when fitting a human body to an intended interaction.

Two directions follow from this work. First, contact verification could incorporate the positions of proposed regions on the 3D scene mesh and views from additional camera angles. This would provide further evidence for assessing whether a region lies on the intended surface, particularly when nearby or repeated structures are ambiguous in the reference view. Second, extending the method to temporal interactions would require representing how contacts are established, maintained, and released over time. Such an extension would need to coordinate contact transitions with continuous body motion and account for physical stability beyond the static configurations considered here.

%-------------------------------------------------------------------------
\bibliographystyle{eg-alpha-doi}
\bibliography{references}

\end{document}
