# composite (deprecated alias)

The composite skills (pick, place, open and close drawers, the look-at sweep) moved to
[phygs/skills/composite/](../skills/composite/). This package only forwards the old import paths
(`phygs.composite`, `.skills`, `.tuning`, `.geometry`) and warns with a `DeprecationWarning`; it will
be removed in a later release. Import from `phygs.skills.composite`.

Up: [phygs/README.md](../README.md). The skills' documentation: [docs/agents.md](../../docs/agents.md#composite-skills).
Tests: `tests/test_composite.py` (the skills), `tests/test_public_surface.py` (the alias).
