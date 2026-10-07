# P2 V3 canonical T* / T*-M evidence

Authority: the accepted Luna 16 finite-action package, reconciled to the Luna 11 final two-action statement and math audit. This file extracts the canonical result and boundaries; it does not replace or rewrite either package proof.

## Canonical finite-action proposition

For a fixed target partition P and finite action set A, assume each block B has a fixed additive risk vector r_B=(r_Ba). Its estimator is jointly Gaussian with mean r_B and covariance V_B/n. Ties use a deterministic priority fixed in advance. Let r*_B=min_a r_Ba and let p_Ba(n) be the Gaussian probability of the tie-priority-adjusted polyhedral region in which action a has the lowest estimated risk. Then:

R*(P)=sum over B in P of r*_B.

Rbar_n(P)=R*(P)+E_n(P), where E_n(P)=sum over B in P and a in A of (r_Ba-r*_B)p_Ba(n).

The terms in E_n(P) are nonnegative action-selection excess risk. A tied oracle action has zero loss gap and contributes zero excess even if it is selected with positive probability.

## Refinement comparison

For fixed nested partitions Q refining P, let G(P,Q)=R*(P)-R*(Q) be the oracle-risk reduction. Then:

Rbar_n(Q)-Rbar_n(P)=-G(P,Q)+E_n(Q)-E_n(P).

Therefore Q has strictly lower expected learned risk exactly when G(P,Q)>E_n(Q)-E_n(P). More resolution can reduce oracle risk while increasing estimation and action-selection cost. This comparison assumes additive deployment risk, a fixed action vocabulary, and prespecified partitions.

## Gaussian action-selection probability

For M actions, the event that action a is selected is a tie-priority-adjusted polyhedral region defined by pairwise linear inequalities between estimated action risks. Its probability p_Ba(n) is the corresponding multivariate-normal polyhedral probability formed from the M-1 dimensional risk-difference mean vector and covariance. For M>2 it generally requires numerical integration. Cross-block covariance drops out of expected risk only when block decisions are separate and total loss is additive; coupled losses can retain cross-block dependence.

## Exact M=2 reduction

With actions 0 and 1, target weights q_t, contrasts delta_t=L_t(0)-L_t(1), and d_B=sum_(t in B) q_t delta_t, let v_B=w_B' Sigma w_B for w_B(t)=q_t 1(t in B). The exact blockwise excess is:

e_(n,B)=|d_B| Phi(-sqrt(n)|d_B|/sqrt(v_B)) when d_B is nonzero and v_B>0; otherwise it is zero.

The two-action formula is an exact M=2 special case. Zero signal, zero scalar variance, singular positive-semidefinite covariance, and deterministic ties are handled without inverting Sigma.

## Crossover and manuscript boundary

The expected-risk difference need not be monotone in n and may have multiple crossings. The Sol two-target benchmark crosses at n=0.9922849575458863; its three-target adversarial benchmark has two roots, about 0.6539385 and 4.4479445. These are mathematical checks, not P2 outcomes.

Allowed placement: MAIN PROPOSITION / DECISION-THEORETIC SYNTHESIS. Present the standard Gaussian sign-decision penalty and additive partition result as an exact synthesis supporting the scientific target-resolution framework.

Do not claim a new general Gaussian choice theorem, a new sign-regret formula, a first target-resolution theory, a distribution-free guarantee, a unique sample-size threshold, monotone learned-risk improvement, an empirical population oracle/selection penalty, or that empirical P2 systems exactly satisfy the Gaussian model.

## P2 action-set mapping

The finite-action package maps RLC-1 to 3 policy actions, Radbill to 4, B1 to the final 8-channel vocabulary, and Awinda to its finite frequency-pair grid. It does not estimate empirical Gaussian covariance, fit a Gaussian model to P2, pool risk scales, or infer an empirical oracle. The earlier two-action expression remains exact; the eight-action B1 cardinality is now consistent with the later B13R2 eight-channel freeze.

## Provenance

- Finite-action canonical statement: TSTAR_M_MANUSCRIPT_CANONICAL_TEXT.md in P2_V3_TSTAR_MULTI_ACTION_LUNA16_2026-10-06.zip.
- Multi-action derivation, edge cases, and numerical checks: TSTAR_M_FULL_DERIVATION.md, TSTAR_M_EDGE_CASES.md, TSTAR_M_NUMERICAL_CHECKS.csv, and TSTAR_M_GATE_REPORT.md in that package.
- Exact M=2 formula and final placement: TSTAR_FINAL_CANONICAL_STATEMENT.md, TSTAR_FINAL_MATH_AUDIT.md, TSTAR_FINAL_GATE_REPORT.md, and TSTAR_FINAL_BENCHMARK_REPRODUCTION.csv in P2_V3_TSTAR_FINAL_RECONCILIATION_LUNA11_2026-10-06.zip.
