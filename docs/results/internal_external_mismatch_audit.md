# Internal vs External Mismatch Audit

## Purpose

This audit explains why mapped external performance plateaus even after external few-shot adaptation restores schema completeness.

It focuses on label-space mismatch, field entropy, and mapping purity differences between the in-domain reduced training set and the mapped external customer-support dataset.

## Dataset Snapshot

- internal train rows: `1993`
- external train rows: `5926`
- external test rows: `1698`

Internal source mix:
- `kameronb_it_callcenter_tickets`: `1599` (`0.802`)
- `console_ai_it_helpdesk_tickets`: `394` (`0.198`)

Entropy comparison:
- category entropy: internal `0.9168` vs external train `0.7904`
- priority entropy: internal `0.5489` vs external train `0.9997`
- component entropy: internal `0.7063` vs external train `0.9519`

## Label-Space Overlap

### category

- internal vocab: `5`
- external vocab: `4`
- overlap vocab: `4`
- external-only vocab: `0`
- external mass on overlapping labels: `1.0000`
- external mass on external-only labels: `0.0000`

### priority

- internal vocab: `4`
- external vocab: `4`
- overlap vocab: `4`
- external-only vocab: `0`
- external mass on overlapping labels: `1.0000`
- external mass on external-only labels: `0.0000`

### component

- internal vocab: `25`
- external vocab: `6`
- overlap vocab: `6`
- external-only vocab: `0`
- external mass on overlapping labels: `1.0000`
- external mass on external-only labels: `0.0000`

### action_template

- internal vocab: `5`
- external vocab: `4`
- overlap vocab: `4`
- external-only vocab: `0`
- external mass on overlapping labels: `1.0000`
- external mass on external-only labels: `0.0000`

### name

- internal vocab: `279`
- external vocab: `42`
- overlap vocab: `0`
- external-only vocab: `42`
- external mass on overlapping labels: `0.0000`
- external mass on external-only labels: `1.0000`

Top external-only labels:
- `roomba robot vacuum`: `170`
- `nest thermostat`: `165`
- `canon eos`: `165`
- `lg oled`: `160`
- `gopro hero`: `159`
- `apple airpods`: `155`
- `canon dslr camera`: `153`
- `garmin forerunner`: `152`
- `microsoft xbox controller`: `150`
- `lg washing machine`: `148`
- `sony xperia`: `147`
- `amazon echo`: `147`

## Top Label Distribution

### Internal train

- `category`: `task` (597), `bug` (497), `feature` (419), `incident` (413), `question` (67)
- `priority`: `medium` (1376), `high` (521), `urgent` (92), `low` (4)
- `component`: `configuration` (490), `installation` (385), `error` (326), `access` (199), `software` (183), `malfunction` (72), `account` (54), `network` (54)

### External train

- `category`: `task` (2412), `question` (2292), `bug` (1169), `incident` (53)
- `priority`: `medium` (1543), `urgent` (1498), `low` (1465), `high` (1420)
- `component`: `software` (1242), `deactivation` (1218), `request` (1194), `account` (1134), `error` (907), `hardware` (231)

## Mapping Purity

### Internal `name -> component`

- eligible keys: `205`
- weighted purity: `0.6587`
- median key purity: `0.5556`

Most ambiguous keys:
- `Piv Card`: purity `0.162`, support `74`, labels `activation (12), renewal (11), issue (11), request (11)`
- `SecureAccess`: purity `0.200`, support `5`, labels `installation (1), configuration (1), performance (1), access (1)`
- `InvoiceInsight`: purity `0.250`, support `4`, labels `malfunction (1), compatibility (1), installation (1), configuration (1)`
- `ProjectPulse`: purity `0.250`, support `4`, labels `access (1), configuration (1), installation (1), training (1)`
- `Trello`: purity `0.273`, support `11`, labels `configuration (3), access (3), error (3), training (1)`
- `Coursera for Business`: purity `0.286`, support `7`, labels `error (2), installation (2), access (1), malfunction (1)`
- `LoadLight`: purity `0.286`, support `7`, labels `error (2), access (2), installation (1), configuration (1)`
- `Microsoft Office 365`: purity `0.286`, support `7`, labels `access (2), configuration (2), installation (2), error (1)`

### External `name -> component`

- eligible keys: `42`
- weighted purity: `0.2546`
- median key purity: `0.2372`

Most ambiguous keys:
- `nintendo switch`: purity `0.210`, support `124`, labels `error (26), request (26), deactivation (25), software (25)`
- `gopro hero`: purity `0.214`, support `159`, labels `error (34), software (33), account (31), deactivation (31)`
- `nintendo switch pro controller`: purity `0.217`, support `129`, labels `deactivation (28), request (27), software (26), account (25)`
- `roomba robot vacuum`: purity `0.218`, support `170`, labels `account (37), request (36), error (35), software (34)`
- `canon eos`: purity `0.218`, support `165`, labels `software (36), error (36), request (32), deactivation (31)`
- `fitbit charge`: purity `0.219`, support `146`, labels `request (32), error (30), deactivation (29), software (28)`
- `macbook pro`: purity `0.220`, support `132`, labels `software (29), request (29), account (29), hardware (28)`
- `sony 4k hdr tv`: purity `0.222`, support `144`, labels `account (32), hardware (30), request (30), software (28)`

### Internal `summary -> category`

- eligible keys: `15`
- weighted purity: `0.9833`
- median key purity: `1.0000`

Most ambiguous keys:
- `Assistance Required for Configuring Multi-Factor Authentication`: purity `0.800`, support `5`, labels `task (4), question (1)`
- `Activation request for new PIV card`: purity `1.000`, support `6`, labels `task (6)`
- `Distribution List Creation Request`: purity `1.000`, support `6`, labels `task (6)`
- `Distribution List Modification Request`: purity `1.000`, support `5`, labels `task (5)`
- `Request for new PIV card issuance`: purity `1.000`, support `4`, labels `task (4)`
- `Request to deactivate PIV card immediately`: purity `1.000`, support `4`, labels `task (4)`
- `Shared Mailbox Creation Request`: purity `1.000`, support `4`, labels `task (4)`
- `Shared Mailbox Modification Request`: purity `1.000`, support `4`, labels `task (4)`

### External `summary -> category`

- eligible keys: `16`
- weighted purity: `0.4163`
- median key purity: `0.4139`

Most ambiguous keys:
- `battery life`: purity `0.394`, support `376`, labels `task (148), question (146), bug (82)`
- `product setup`: purity `0.394`, support `378`, labels `task (149), question (145), bug (80), incident (4)`
- `account access`: purity `0.398`, support `374`, labels `question (149), task (143), bug (75), incident (7)`
- `product recommendation`: purity `0.404`, support `369`, labels `question (149), task (148), bug (69), incident (3)`
- `software bug`: purity `0.404`, support `413`, labels `question (167), task (164), bug (82)`
- `delivery problem`: purity `0.405`, support `398`, labels `task (161), question (158), bug (72), incident (7)`
- `peripheral compatibility`: purity `0.405`, support `336`, labels `task (136), question (133), bug (63), incident (4)`
- `network problem`: purity `0.413`, support `351`, labels `question (145), task (130), bug (75), incident (1)`

## Main Conclusions

- `category` and `priority` mostly share the same coarse label space across domains; the problem is not raw label absence but changed conditional semantics.
- `component` is the main taxonomy mismatch field: external carries substantial mass on labels and label usages that are weakly supported or differently structured relative to the in-domain data.
- `action_template` mismatch is downstream of `category`; when category semantics shift, canonical action also shifts with it.
- `name -> component` mappings are much less reusable externally, which explains why in-domain deterministic `component <- name` rules do not transfer cleanly.
- external adaptation solves completeness, but the remaining plateau is best explained by conditional label mismatch rather than by formatting or missing fields.
