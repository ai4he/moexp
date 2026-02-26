import Lake
open Lake DSL

package MathScy where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib MathScy where
  srcDir := "MathScy"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.8.0"
