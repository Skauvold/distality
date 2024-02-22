### Computing relative intra-body location parameters

Suppose we have a shape (polygon) with a backbone through it. The
backbone is roughly a smoothed non-branching subset of the medial
axis. Intersections between the backbone and polygon together with
a direction the proximal and medial points. We then want to compute
four distance fields:

- Distance from edge $d_E$
- Distance from backbone $d_B$
- Distance from proximal point $d_P$
- Distance from distal point $d_D$

Next, based on these, we want to compute the intra-body location
parameters:

- Distality: $d_P / (d_P + d_D)$
- Peripherality: $d_B / (d_B + d_E)$