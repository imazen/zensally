# Catalogue, hero and detail crop purposes

Owner decision, 2026-11-08: product completeness is a per-request purpose.
Faces remain protected in every purpose. Purpose is independent of target
aspect ratio and Minimal/Maximal crop amount.

- Catalogue preserves the whole featured product plus all protected faces.
  If these regions cannot fit the target ratio, use contain/padding.
- Hero allows a deliberate product-detail composition while retaining protected
  faces and the selected story/identity context.
- Detail allows a deliberate crop into a selected product feature, retaining
  protected faces. It does not authorize clipping a face to show the feature.

Record purpose explicitly in new proposals, rendered-review tasks and cache
identities. Keep historical unspecified-purpose judgments unchanged; they are
not automatically valid for all three purposes. Purpose labels and preferred
feature/story regions need their own evidence. A face-only candidate bank does
not become a hero/detail composition model merely by changing its label.

The earlier eight-scene conservative whole-product/story profile is a catalogue
witness, not a global rule for every crop. It permits26 of176 crops and contains
150 under its approximate unreviewed regions. Preserve that measured baseline.
No purpose changes the face-coverage certification flag, admits unreviewed
regions as complete, or weakens the existing crop numerical/face guards.

Purpose-aware public API signatures still belong in the existing review PR.
The research purpose contract alone makes no claim of trained purpose-specific
composition quality.
